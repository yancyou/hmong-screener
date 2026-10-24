# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Any

from omegaconf import MISSING

import torch
import numpy as np
import torch.nn.functional as F

from fairseq.data import Dictionary, FairseqDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.audio_finetuning import LabelEncoder, label_len_fn

from fairseq.data.audio.gender_contrastive_dataset import (
    GenderContrastiveDataset,
    collate_gender_contrastive_samples,
)
from fairseq.tasks.tone_repulsive_task import (
    ToneRepulsiveDataset,
    collate_tone_repulsive_samples,
)
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset
from fairseq.data.add_target_dataset import AddTargetDataset
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.data.audio.augmentation import (
    AudioAugmentations,
    get_conservative_transforms,
    get_validation_transforms,
)


logger = logging.getLogger(__name__)


class _AsrAugmentingDataset(BaseWrapperDataset):
    """
    Lightweight wrapper that applies waveform augmentations to the ASR audio
    examples produced by FileAudioDataset.
    """

    def __init__(self, dataset, audio_transform, renormalize: bool = True):
        super().__init__(dataset)
        self.audio_transform = audio_transform
        self.renormalize = renormalize

    def __getitem__(self, index):
        item = self.dataset[index]
        src = item.get("source", None)
        if src is not None:
            # Convert to numpy, apply transform, convert back
            if isinstance(src, torch.Tensor):
                audio_np = src.numpy()
            else:
                audio_np = np.asarray(src, dtype=np.float32)

            augmented = self.audio_transform(audio_np)
            if not isinstance(augmented, np.ndarray):
                augmented = np.asarray(augmented, dtype=np.float32)
            aug_tensor = torch.from_numpy(augmented.astype(np.float32))

            if self.renormalize:
                with torch.no_grad():
                    aug_tensor = F.layer_norm(aug_tensor, aug_tensor.shape)

            item["source"] = aug_tensor

        return item


@dataclass
class GenderToneRepulsiveTaskConfig(FairseqDataclass):
    # Unified data root
    data: str = field(
        default=MISSING,
        metadata={"help": "path to data directory containing audio files"},
    )
    
    # Gender dataset
    gender_transcript_file: str = field(
        default="hmong_gender_transcript_train_cleaned.txt",
        metadata={"help": "transcript file for gender pairs (filename\tword)"},
    )
    gender_transcript_val_file: Optional[str] = field(
        default=None,
        metadata={"help": "validation transcript for gender (optional)"},
    )
    
    # Tone dataset (repulsive format)
    tone_metadata_file: str = field(
        default="hmong_tone_metadata_train.json",
        metadata={"help": "tone metadata JSON file (repulsive format)"},
    )
    tone_metadata_val_file: Optional[str] = field(
        default=None,
        metadata={"help": "validation tone metadata (optional)"},
    )
    
    # Audio settings
    sample_rate: int = field(default=16000, metadata={"help": "target sample rate"})
    max_audio_length: int = field(default=320000, metadata={"help": "max audio length in samples"})
    min_audio_length: int = field(default=1600, metadata={"help": "min audio length in samples"})
    normalize: bool = field(default=True, metadata={"help": "normalize audio"})
    seed: int = field(default=42, metadata={"help": "random seed"})
    
    # Gender contrastive settings
    num_negatives_gender: int = field(default=20, metadata={"help": "number of gender negatives"})
    
    # Tone repulsive settings
    num_positives_tone: int = field(default=2, metadata={"help": "number of tone positives (same tone)"})
    num_hard_negatives_tone: int = field(default=4, metadata={"help": "number of hard negatives (same word, diff tone)"})
    num_soft_negatives_tone: int = field(default=10, metadata={"help": "number of soft negatives (diff words)"})
    
    # Augmentation settings
    enable_augmentation: bool = field(default=True, metadata={"help": "enable augmentation globally"})
    augmentation_config: Optional[Any] = field(default=None, metadata={"help": "global augmentation config"})
    
    # Stream-specific augmentation
    enable_augmentation_gender: bool = field(default=True, metadata={"help": "enable augmentation for gender"})
    augmentation_gender_config: Optional[Any] = field(default=None, metadata={"help": "gender augmentation config"})
    
    enable_augmentation_tone: bool = field(default=True, metadata={"help": "enable augmentation for tone"})
    augmentation_tone_config: Optional[Any] = field(default=None, metadata={"help": "tone augmentation config"})
    
    enable_augmentation_asr: bool = field(default=True, metadata={"help": "enable augmentation for ASR"})
    augmentation_asr_config: Optional[Any] = field(default=None, metadata={"help": "ASR augmentation config"})
    
    # ASR auxiliary stream
    asr_enable: bool = field(default=False, metadata={"help": "enable ASR auxiliary stream"})
    asr_data_dir: Optional[str] = field(default=None, metadata={"help": "ASR data directory"})
    asr_dictionary_file: Optional[str] = field(default=None, metadata={"help": "ASR dictionary file"})
    asr_labels: str = field(default="ltr", metadata={"help": "ASR label type (ltr, wrd, etc.)"})
    asr_batch_ratio: float = field(default=0.25, metadata={"help": "probability of including ASR in batch"})
    asr_pad_audio: bool = field(default=True, metadata={"help": "pad ASR audio for padding_mask"})


@register_task("gender_tone_repulsive", dataclass=GenderToneRepulsiveTaskConfig)
class GenderToneRepulsiveTask(FairseqTask):
    """
    Task that jointly provides batches for:
    1. Gender contrastive learning (InfoNCE)
    2. Tone repulsive learning (attractive + repulsive + classification)
    3. ASR CTC (optional)
    
    Uses ToneRepulsiveDataset (new) instead of ToneContrastiveDataset (old).
    """

    cfg: GenderToneRepulsiveTaskConfig

    def __init__(self, cfg: GenderToneRepulsiveTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = Dictionary()
        self.blank_symbol = "<s>"

        logger.info("Gender+Tone repulsive task initialized")
        logger.info(f"Data directory: {cfg.data}")

        self._gender_ds = None
        self._tone_ds = None
        self._asr_ds = None

        # Load ASR dictionary if needed
        if self.cfg.asr_enable:
            dict_path = None
            if self.cfg.asr_dictionary_file and os.path.exists(self.cfg.asr_dictionary_file):
                dict_path = self.cfg.asr_dictionary_file
            elif self.cfg.asr_data_dir:
                cand = os.path.join(self.cfg.asr_data_dir, f"dict.{self.cfg.asr_labels}.txt")
                if os.path.exists(cand):
                    dict_path = cand
            if dict_path is not None:
                try:
                    self.dictionary = Dictionary.load(dict_path)
                    logger.info(f"Loaded ASR dictionary from {dict_path} (size={len(self.dictionary)})")
                except Exception as e:
                    logger.warning(f"Failed to load ASR dictionary: {e}")

    @classmethod
    def setup_task(cls, cfg: GenderToneRepulsiveTaskConfig, **kwargs):
        return cls(cfg)

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        # Select files for split
        is_train = (split == "train")
        
        gender_transcript_path = (
            self.cfg.gender_transcript_file if is_train
            else (self.cfg.gender_transcript_val_file or self.cfg.gender_transcript_file)
        )
        
        tone_metadata_path = (
            self.cfg.tone_metadata_file if is_train
            else (self.cfg.tone_metadata_val_file or self.cfg.tone_metadata_file)
        )

        # Augmentation settings
        enable_aug_gender = self.cfg.enable_augmentation and self.cfg.enable_augmentation_gender and is_train
        enable_aug_tone = self.cfg.enable_augmentation and self.cfg.enable_augmentation_tone and is_train
        enable_aug_asr = self.cfg.enable_augmentation and self.cfg.enable_augmentation_asr and is_train

        logger.info(f"Loading {split} dataset:")
        logger.info(f"  Gender transcript: {gender_transcript_path}")
        logger.info(f"  Tone metadata: {tone_metadata_path}")
        logger.info(f"  Gender augmentation: {enable_aug_gender}")
        logger.info(f"  Tone augmentation: {enable_aug_tone}")

        # Create gender dataset
        self._gender_ds = GenderContrastiveDataset(
            data_dir=self.cfg.data,
            transcript_file=gender_transcript_path,
            sample_rate=self.cfg.sample_rate,
            num_negatives=self.cfg.num_negatives_gender,
            max_audio_length=self.cfg.max_audio_length,
            min_audio_length=self.cfg.min_audio_length,
            normalize=self.cfg.normalize,
            seed=self.cfg.seed + hash((split, "gender")),
            enable_augmentation=enable_aug_gender,
            augmentation_config=(
                self.cfg.augmentation_gender_config
                if self.cfg.augmentation_gender_config is not None
                else self.cfg.augmentation_config
            ),
        )

        # Create tone repulsive dataset
        # Load tone metadata JSON
        import json
        with open(tone_metadata_path, 'r', encoding='utf-8') as f:
            tone_metadata = json.load(f)
        
        self._tone_ds = ToneRepulsiveDataset(
            data_dir=self.cfg.data,
            tone_metadata=tone_metadata,  # Pass loaded dictionary
            sample_rate=self.cfg.sample_rate,
            num_positives=self.cfg.num_positives_tone,
            num_hard_negatives=self.cfg.num_hard_negatives_tone,
            num_soft_negatives=self.cfg.num_soft_negatives_tone,
            max_audio_length=self.cfg.max_audio_length,
            min_audio_length=self.cfg.min_audio_length,
            normalize=self.cfg.normalize,
            seed=self.cfg.seed + hash((split, "tone")),
            enable_augmentation=enable_aug_tone,
            augmentation_config=(
                self.cfg.augmentation_tone_config
                if self.cfg.augmentation_tone_config is not None
                else self.cfg.augmentation_config
            ),
        )

        # Optionally create ASR dataset
        self._asr_ds = None
        if self.cfg.asr_enable and self.cfg.asr_data_dir:
            manifest_path = os.path.join(self.cfg.asr_data_dir, f"{split}.tsv")
            if os.path.exists(manifest_path):
                logger.info(f"Loading ASR dataset from {manifest_path}")
                
                # Load raw audio
                asr_audio_ds = FileAudioDataset(
                    manifest_path=manifest_path,
                    sample_rate=self.cfg.sample_rate,
                    max_sample_size=self.cfg.max_audio_length,
                    min_sample_size=self.cfg.min_audio_length,
                    pad=self.cfg.asr_pad_audio,
                    normalize=self.cfg.normalize,
                )

                # Apply augmentation if enabled
                if enable_aug_asr:
                    aug_cfg = (
                        self.cfg.augmentation_asr_config
                        if self.cfg.augmentation_asr_config is not None
                        else self.cfg.augmentation_config
                    )
                    if aug_cfg is not None:
                        from fairseq.data.audio.augmentation import AudioAugmentations
                        audio_transform = AudioAugmentations(
                            sample_rate=self.cfg.sample_rate,
                            training=True,
                            aug_prob=aug_cfg.get("aug_prob", 0.3),
                            noise_factor_range=tuple(aug_cfg.get("noise_factor_range", [0.002, 0.01])),
                            time_stretch_range=tuple(aug_cfg.get("time_stretch_range", [0.98, 1.02])),
                            pitch_shift_range=tuple(aug_cfg.get("pitch_shift_range", [-0.1, 0.1])),
                            volume_gain_range=tuple(aug_cfg.get("volume_gain_range", [-1.5, 1.5])),
                            enable_noise=aug_cfg.get("enable_noise", True),
                            enable_time_stretch=aug_cfg.get("enable_time_stretch", True),
                            enable_pitch_shift=aug_cfg.get("enable_pitch_shift", False),
                            enable_volume=aug_cfg.get("enable_volume", True),
                        )
                        asr_audio_ds = _AsrAugmentingDataset(asr_audio_ds, audio_transform, renormalize=True)

                # Add targets
                label_path = os.path.join(self.cfg.asr_data_dir, f"{split}.{self.cfg.asr_labels}")
                if os.path.exists(label_path):
                    from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
                    
                    # Use no compression (same as gender_tone_contrastive_task)
                    text_compressor = TextCompressor(level=TextCompressionLevel.none)
                    
                    with open(label_path, "r", encoding="utf-8") as f:
                        labels = [text_compressor.compress(l.rstrip("\n")) for l in f]
                    
                    process_label = LabelEncoder(self.dictionary)
                    self._asr_ds = AddTargetDataset(
                        asr_audio_ds,
                        labels,
                        pad=self.dictionary.pad(),
                        eos=self.dictionary.eos(),
                        batch_targets=True,
                        process_label=process_label,
                        label_len_fn=label_len_fn,
                        add_to_input=False,
                    )
                    logger.info(f"Loaded {len(self._asr_ds)} ASR samples")

        # Wrap in proxy dataset
        proxy = _JointGenderToneRepulsiveProxy(
            gender_ds=self._gender_ds,
            tone_ds=self._tone_ds,
            asr_ds=self._asr_ds,
            asr_ratio=self.cfg.asr_batch_ratio if self.cfg.asr_enable else 0.0,
        )

        self.datasets[split] = proxy
        logger.info(f"Loaded {len(proxy)} samples for {split}")

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class _JointGenderToneRepulsiveProxy(FairseqDataset):
    """
    Proxy dataset that combines gender and tone repulsive datasets.
    Each sample contains both gender and tone data.
    """

    def __init__(self, gender_ds, tone_ds, asr_ds=None, asr_ratio=0.0):
        self.gender_ds = gender_ds
        self.tone_ds = tone_ds
        self.asr_ds = asr_ds
        self.asr_ratio = asr_ratio
        
        # Use the smaller dataset size to avoid index errors
        self._len = min(len(gender_ds), len(tone_ds))

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        # Get samples from both datasets
        gender_sample = self.gender_ds[index % len(self.gender_ds)]
        tone_sample = self.tone_ds[index % len(self.tone_ds)]

        # Combine into single sample
        sample = {
            'gender': gender_sample,
            'tone': tone_sample,
            'metadata': {
                'word_gender': gender_sample.get('base_word', ''),
                'word_tone': tone_sample.get('base_word', ''),
            }
        }

        # Optionally add ASR
        if self.asr_ds is not None and random.random() < self.asr_ratio:
            asr_index = random.randint(0, len(self.asr_ds) - 1)
            sample['asr'] = self.asr_ds[asr_index]

        return sample

    def collate_fn(self, samples):
        """Collate samples into batch"""
        # Separate gender and tone samples
        gender_samples = []
        tone_samples = []
        asr_samples = []

        for s in samples:
            # Gender sample is raw from dataset (has 'female_audio', 'male_positive_audio', etc.)
            gender_samples.append(s['gender'])

            # Tone sample is raw from dataset
            tone_samples.append(s['tone'])

            # ASR sample (if present)
            if 'asr' in s:
                asr_samples.append(s['asr'])

        # Collate each stream
        g_batch = collate_gender_contrastive_samples(gender_samples)
        t_batch = collate_tone_repulsive_samples(tone_samples)

        # Merge net_inputs
        net_input = {}
        
        # Gender stream
        net_input.update({
            'female_audio': g_batch['net_input']['female_audio'],
            'male_positive_audio': g_batch['net_input']['male_positive_audio'],
            'male_negative_audio': g_batch['net_input']['male_negative_audio'],
            'female_padding_mask': g_batch['net_input']['female_padding_mask'],
            'male_positive_padding_mask': g_batch['net_input']['male_positive_padding_mask'],
            'male_negative_padding_mask': g_batch['net_input']['male_negative_padding_mask'],
        })
        
        # Tone stream (repulsive format)
        net_input.update({
            'anchor_audio': t_batch['net_input']['anchor_audio'],
            'positive_audio': t_batch['net_input']['positive_audio'],
            'hard_negative_audio': t_batch['net_input']['hard_negative_audio'],
            'soft_negative_audio': t_batch['net_input']['soft_negative_audio'],
            'anchor_padding_mask': t_batch['net_input']['anchor_padding_mask'],
            'positive_padding_mask': t_batch['net_input']['positive_padding_mask'],
            'hard_negative_padding_mask': t_batch['net_input']['hard_negative_padding_mask'],
            'soft_negative_padding_mask': t_batch['net_input']['soft_negative_padding_mask'],
            'positive_valid_mask': t_batch['net_input']['positive_valid_mask'],
            'hard_negative_valid_mask': t_batch['net_input']['hard_negative_valid_mask'],
            'soft_negative_valid_mask': t_batch['net_input']['soft_negative_valid_mask'],
            'anchor_tone_label': t_batch['net_input']['anchor_tone_label'],
        })

        # ASR stream (if present)
        if asr_samples:
            # Collate ASR samples
            asr_sources = [s['source'] for s in asr_samples]
            asr_targets = [s['label'] for s in asr_samples]  # AddTargetDataset uses 'label', not 'target'
            
            max_source_len = max(src.size(0) for src in asr_sources)
            max_target_len = max(tgt.size(0) for tgt in asr_targets)
            
            asr_audio = torch.zeros(len(asr_sources), max_source_len)
            asr_padding_mask = torch.ones(len(asr_sources), max_source_len, dtype=torch.bool)
            asr_target = torch.zeros(len(asr_targets), max_target_len, dtype=torch.long)
            asr_target_lengths = torch.zeros(len(asr_targets), dtype=torch.long)
            
            for i, (src, tgt) in enumerate(zip(asr_sources, asr_targets)):
                asr_audio[i, :src.size(0)] = src
                asr_padding_mask[i, :src.size(0)] = False
                asr_target[i, :tgt.size(0)] = tgt
                asr_target_lengths[i] = tgt.size(0)
            
            net_input.update({
                'asr_audio': asr_audio,
                'asr_padding_mask': asr_padding_mask,
                'asr_target': asr_target,
                'asr_target_lengths': asr_target_lengths,
            })

        total_ntokens = g_batch.get('ntokens', 0) + t_batch.get('ntokens', 0)
        nsentences = max(g_batch.get('nsentences', 0), t_batch.get('nsentences', 0))

        return {
            'net_input': net_input,
            'ntokens': total_ntokens,
            'nsentences': nsentences,
        }

    def collater(self, samples):
        return self.collate_fn(samples)

    def num_tokens(self, index):
        return self.gender_ds.num_tokens(index % len(self.gender_ds))

    def size(self, index):
        return self.gender_ds.size(index % len(self.gender_ds))

    def ordered_indices(self):
        return np.arange(len(self))

    def supports_prefetch(self):
        return False
    
    def prefetch(self, indices):
        """Prefetch data for given indices (no-op for this dataset)"""
        pass
