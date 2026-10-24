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
from fairseq.tasks.tone_contrastive_task import ToneContrastiveTaskConfig
from fairseq.tasks.gender_contrastive_task import GenderContrastiveTaskConfig
from fairseq.tasks.audio_finetuning import LabelEncoder, label_len_fn

from fairseq.data.audio.gender_contrastive_dataset import (
    GenderContrastiveDataset,
    collate_gender_contrastive_samples,
)
from fairseq.data.audio.tone_contrastive_dataset import (
    ToneContrastiveDataset,
    collate_tone_contrastive_samples,
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
    examples produced by FileAudioDataset. Uses the existing audio augmentation
    pipeline to keep behavior consistent with contrastive datasets.
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
class GenderToneContrastiveTaskConfig(FairseqDataclass):
    # Unified data root and manifests/metadata
    data: str = field(
        default=MISSING,
        metadata={"help": "path to data directory containing audio files"},
    )
    # For gender dataset
    gender_transcript_file: str = field(
        default="hmong_gender_transcript_train_cleaned.txt",
        metadata={"help": "transcript file for gender pairs (filename\tword)"},
    )
    # For tone dataset
    tone_transcript_file: str = field(
        default="hmong_new_wordlist_transcripts.txt",
        metadata={"help": "transcript file for tone groups (filename\tword)"},
    )
    tone_metadata_file: str = field(
        default="hmong_tone_metadata.json",
        metadata={"help": "tone metadata JSON with tone groups"},
    )

    # Optional explicit validation overrides (to align with external ASR split)
    gender_transcript_val_file: Optional[str] = field(
        default="",
        metadata={"help": "explicit gender validation transcript file (filename\tword)"},
    )
    tone_transcript_val_file: Optional[str] = field(
        default="",
        metadata={"help": "explicit tone validation transcript file (filename\tword)"},
    )
    tone_metadata_val_file: Optional[str] = field(
        default="",
        metadata={"help": "explicit tone validation metadata JSON"},
    )

    # Audio/common settings
    sample_rate: int = field(default=16000)
    num_negatives_gender: int = field(default=100)
    num_negatives_tone: int = field(default=20)
    max_audio_length: Optional[int] = field(default=320000)
    min_audio_length: Optional[int] = field(default=1600)
    normalize: bool = field(default=True)
    seed: int = field(default=42)

    # Augmentation (global toggle + optional per-stream overrides)
    enable_augmentation: bool = field(default=False)
    augmentation_config: Optional[Any] = field(default=None)
    # Stream-specific toggles (fall back to enable_augmentation if None)
    enable_augmentation_gender: Optional[bool] = field(default=None)
    enable_augmentation_tone: Optional[bool] = field(default=None)
    enable_augmentation_asr: Optional[bool] = field(default=None)
    # Stream-specific configs (fall back to augmentation_config if None)
    augmentation_gender_config: Optional[Any] = field(default=None)
    augmentation_tone_config: Optional[Any] = field(default=None)
    augmentation_asr_config: Optional[Any] = field(default=None)

    # Optional ASR auxiliary stream
    asr_enable: bool = field(default=False, metadata={"help": "enable ASR auxiliary stream"})
    asr_data_dir: Optional[str] = field(default=None, metadata={"help": "path to ASR manifests dir (train.tsv, valid.tsv, *.ltr, dict.ltr.txt)"})
    asr_dictionary_file: Optional[str] = field(default=None, metadata={"help": "explicit path to dict.ltr.txt; overrides asr_data_dir"})
    asr_labels: str = field(default="ltr", metadata={"help": "label type for ASR (usually ltr)"})
    asr_batch_ratio: float = field(default=0.25, metadata={"help": "probability to include ASR in a batch"})
    asr_pad_audio: bool = field(default=True, metadata={"help": "pad ASR audio within batch to enable padding_mask"})


@register_task("gender_tone_contrastive", dataclass=GenderToneContrastiveTaskConfig)
class GenderToneContrastiveTask(FairseqTask):
    """
    Task that jointly provides batches for gender and tone contrastive learning.

    It instantiates both the GenderContrastiveDataset and ToneContrastiveDataset
    under the hood and returns a combined sample where `net_input` contains the
    union of fields needed by the combined criterion.
    """

    cfg: GenderToneContrastiveTaskConfig

    def __init__(self, cfg: GenderToneContrastiveTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg
        # Default dictionary; may be replaced by ASR dict
        self.dictionary = Dictionary()
        # Use BOS as CTC blank (index 0) to align with wav2vec_ctc convention
        self.blank_symbol = "<s>"

        logger.info("Gender+Tone contrastive task initialized")
        logger.info(f"Data directory: {cfg.data}")

        self._gender_ds = None
        self._tone_ds = None
        self._asr_ds = None

        # Try to load ASR dictionary upfront (so model output dim is correct)
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
                    logger.warning(f"Failed to load ASR dictionary at {dict_path}: {e}")

    @classmethod
    def setup_task(cls, cfg: GenderToneContrastiveTaskConfig, **kwargs):
        return cls(cfg)

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        # Select transcript/metadata for split (reuse train for simplicity; extend as needed)
        if split in ("val", "valid"):
            # Prefer explicit overrides when provided; otherwise fall back to name substitution
            if self.cfg.gender_transcript_val_file:
                gender_transcript = self.cfg.gender_transcript_val_file
            else:
                gender_transcript = self.cfg.gender_transcript_file.replace("_train", "_val")

            if self.cfg.tone_transcript_val_file:
                tone_transcript = self.cfg.tone_transcript_val_file
            else:
                tone_transcript = self.cfg.tone_transcript_file

            if self.cfg.tone_metadata_val_file:
                tone_metadata = self.cfg.tone_metadata_val_file
            else:
                tone_metadata = self.cfg.tone_metadata_file.replace("_train.json", "_val.json")
        else:
            gender_transcript = self.cfg.gender_transcript_file
            tone_transcript = self.cfg.tone_transcript_file
            tone_metadata = self.cfg.tone_metadata_file

        gender_transcript_path = gender_transcript if os.path.isabs(gender_transcript) else os.path.join(self.cfg.data, gender_transcript)
        tone_transcript_path = tone_transcript if os.path.isabs(tone_transcript) else os.path.join(self.cfg.data, tone_transcript)
        tone_metadata_path = tone_metadata if os.path.isabs(tone_metadata) else os.path.join(self.cfg.data, tone_metadata)

        if not os.path.exists(gender_transcript_path):
            raise FileNotFoundError(f"Gender transcript not found: {gender_transcript_path}")
        if not os.path.exists(tone_transcript_path):
            raise FileNotFoundError(f"Tone transcript not found: {tone_transcript_path}")
        if not os.path.exists(tone_metadata_path):
            raise FileNotFoundError(f"Tone metadata not found: {tone_metadata_path}")

        # Resolve per-stream augmentation toggles
        enable_aug_global = self.cfg.enable_augmentation and split == "train"
        enable_aug_gender = (
            self.cfg.enable_augmentation_gender
            if self.cfg.enable_augmentation_gender is not None
            else self.cfg.enable_augmentation
        ) and split == "train"
        enable_aug_tone = (
            self.cfg.enable_augmentation_tone
            if self.cfg.enable_augmentation_tone is not None
            else self.cfg.enable_augmentation
        ) and split == "train"

        # Instantiate sub-datasets
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

        self._tone_ds = ToneContrastiveDataset(
            data_dir=self.cfg.data,
            transcript_file=tone_transcript_path,
            tone_metadata_file=tone_metadata_path,
            sample_rate=self.cfg.sample_rate,
            num_negatives=self.cfg.num_negatives_tone,
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

        # Optionally instantiate ASR dataset
        self._asr_ds = None
        if self.cfg.asr_enable and self.cfg.asr_data_dir:
            split_name = "valid" if split in ("val", "valid") else "train"
            asr_manifest = os.path.join(self.cfg.asr_data_dir, f"{split_name}.tsv")
            asr_labels_path = os.path.join(self.cfg.asr_data_dir, f"{split_name}.{self.cfg.asr_labels}")
            if os.path.exists(asr_manifest) and os.path.exists(asr_labels_path):
                base_asr = FileAudioDataset(
                    manifest_path=asr_manifest,
                    sample_rate=self.cfg.sample_rate,
                    pad=self.cfg.asr_pad_audio,
                    normalize=self.cfg.normalize,
                    num_buckets=0,
                )
                # Apply augmentation policy to ASR stream (train only), allow per-stream override
                asr_enable_aug = (
                    self.cfg.enable_augmentation_asr
                    if self.cfg.enable_augmentation_asr is not None
                    else self.cfg.enable_augmentation
                ) and split == "train"
                if asr_enable_aug:
                    asr_aug_cfg = (
                        self.cfg.augmentation_asr_config
                        if self.cfg.augmentation_asr_config is not None
                        else self.cfg.augmentation_config
                    )
                    if asr_aug_cfg is None:
                        audio_transform = get_conservative_transforms(sample_rate=self.cfg.sample_rate)
                    else:
                        # Allow custom augmentation config
                        audio_transform = AudioAugmentations(
                            sample_rate=self.cfg.sample_rate,
                            training=True,
                            **asr_aug_cfg,
                        )
                    base_asr = _AsrAugmentingDataset(base_asr, audio_transform, renormalize=True)
                else:
                    # Wrap with a no-op transform for consistent behavior
                    audio_transform = get_validation_transforms()
                    base_asr = _AsrAugmentingDataset(base_asr, audio_transform, renormalize=True)
                # Read labels and (optionally) compress for AddTargetDataset
                text_compressor = TextCompressor(level=TextCompressionLevel.none)
                with open(asr_labels_path, "r", encoding="utf-8") as f:
                    labels = [text_compressor.compress(l.rstrip("\n")) for l in f]
                # Ensure dictionary is loaded
                if len(self.dictionary) <= 0:
                    dict_path = self.cfg.asr_dictionary_file or os.path.join(self.cfg.asr_data_dir, f"dict.{self.cfg.asr_labels}.txt")
                    if os.path.exists(dict_path):
                        self.dictionary = Dictionary.load(dict_path)
                        logger.info(f"Loaded ASR dictionary from {dict_path} (size={len(self.dictionary)})")
                    else:
                        logger.warning(f"ASR dictionary not found at {dict_path}; CTC head may be misconfigured")
                self._asr_ds = AddTargetDataset(
                    base_asr,
                    labels,
                    pad=self.dictionary.pad(),
                    eos=self.dictionary.eos(),
                    batch_targets=True,
                    process_label=LabelEncoder(self.dictionary),
                    label_len_fn=label_len_fn,
                    add_to_input=False,
                    text_compression_level=TextCompressor(level=TextCompressionLevel.none).level,
                )
                logger.info(f"Loaded ASR dataset: {split_name} n={len(self._asr_ds)}")
            else:
                logger.warning(f"ASR files missing for split={split}: {asr_manifest}, {asr_labels_path}")

        # Wrap into a lightweight proxy that fetches aligned pairs by index cycling
        self.datasets[split] = _JointGenderToneProxy(self._gender_ds, self._tone_ds, self._asr_ds, self.cfg.asr_batch_ratio)
        logger.info(f"Loaded joint dataset: gender={len(self._gender_ds)}, tone={len(self._tone_ds)}")

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        from fairseq.data import iterators
        import numpy as np
        import torch
        import random

        if max_tokens is None:
            if max_sentences is None:
                max_sentences = 1
            batch_size = max_sentences
        else:
            batch_size = max_sentences or 1

        logger.info(f"Creating joint batch iterator with batch size: {batch_size}")

        indices = list(range(len(dataset)))
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.shuffle(indices)

        batches = []
        for i in range(0, len(indices), batch_size):
            batches.append(indices[i:i + batch_size])

        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_sampler=batches,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def max_positions(self):
        return self.cfg.max_audio_length or float('inf')


class _JointGenderToneProxy(FairseqDataset):
    """
    Minimal proxy dataset that pairs a gender sample and a tone sample by cycling
    through the shorter dataset. Collates into a single `net_input` dict containing
    all fields required by the combined criterion.
    """

    def __init__(self, gender_ds, tone_ds, asr_ds=None, asr_ratio: float = 0.0):
        self.gender_ds = gender_ds
        self.tone_ds = tone_ds
        self.asr_ds = asr_ds
        self.asr_ratio = max(0.0, min(1.0, asr_ratio))

    def __len__(self):
        return max(len(self.gender_ds), len(self.tone_ds))

    def set_epoch(self, epoch):
        if hasattr(self.gender_ds, 'set_epoch'):
            self.gender_ds.set_epoch(epoch)
        if hasattr(self.tone_ds, 'set_epoch'):
            self.tone_ds.set_epoch(epoch)

    def __getitem__(self, index):
        g = self.gender_ds[index % len(self.gender_ds)]
        t = self.tone_ds[index % len(self.tone_ds)]

        # Merge fields into a single sample
        sample = {
            'net_input': {
                # Gender group
                'female_audio': g['female_audio'],
                'male_positive_audio': g['male_positive_audio'],
                # Keep negatives as a list; collate will pad and stack
                'male_negative_audio': g['male_negative_audios'],
                'female_padding_mask': None,  # filled by collate
                'male_positive_padding_mask': None,
                'male_negative_padding_mask': None,

                # Tone group
                'anchor_audio': t['anchor_audio'],
                # Keep positives/negatives as lists; collate will pad and stack
                'positive_audio': t['positive_audios'] if len(t['positive_audios']) > 0 else [t['anchor_audio']],
                'negative_audio': t['negative_audios'],
                'anchor_padding_mask': None,
                'positive_padding_mask': None,
                'negative_padding_mask': None,
                'positive_valid_mask': None,
            },
            'metadata': {
                'word_gender': g.get('word'),
                'word_tone': t.get('base_word'),
            },
        }
        return sample

    def collate_fn(self, samples):
        # We will reuse existing collate functions to build masks, then merge
        # First, rebuild minimal lists expected by collators
        if not samples:
            return {}

        gender_samples = []
        tone_samples = []
        for s in samples:
            # Reconstruct sample dicts compatible with existing collators
            # Handle negatives that may be list[T] or Tensor[N, T]
            male_negs = s['net_input']['male_negative_audio']
            if isinstance(male_negs, torch.Tensor):
                male_neg_list = [x for x in male_negs]
                male_neg_count = male_negs.size(0)
            else:
                male_neg_list = list(male_negs)
                male_neg_count = len(male_neg_list)

            g = {
                'female_audio': s['net_input']['female_audio'],
                'male_positive_audio': s['net_input']['male_positive_audio'],
                'male_negative_audios': male_neg_list,
                'word': s['metadata']['word_gender'],
                'female_path': '',
                'male_positive_path': '',
                'male_negative_paths': [''] * male_neg_count,
            }
            # Handle positives/negatives that may be list[T] or Tensor[P/N, T]
            pos = s['net_input']['positive_audio']
            if isinstance(pos, torch.Tensor):
                pos_list = [x for x in pos]
                pos_count = pos.size(0)
            else:
                pos_list = list(pos)
                pos_count = len(pos_list)

            neg = s['net_input']['negative_audio']
            if isinstance(neg, torch.Tensor):
                neg_list = [x for x in neg]
                neg_count = neg.size(0)
            else:
                neg_list = list(neg)
                neg_count = len(neg_list)

            t = {
                'anchor_audio': s['net_input']['anchor_audio'],
                'positive_audios': pos_list,
                'negative_audios': neg_list,
                'base_word': s['metadata']['word_tone'],
                'anchor_tone': '',
                'positive_tones': [''] * pos_count,
                'anchor_path': '',
                'positive_paths': [''] * pos_count,
                'negative_paths': [''] * neg_count,
            }
            gender_samples.append(g)
            tone_samples.append(t)

        g_batch = collate_gender_contrastive_samples(gender_samples)
        t_batch = collate_tone_contrastive_samples(tone_samples)

        # Merge net_inputs
        net_input = {}
        net_input.update({
            'female_audio': g_batch['net_input']['female_audio'],
            'male_positive_audio': g_batch['net_input']['male_positive_audio'],
            'male_negative_audio': g_batch['net_input']['male_negative_audio'],
            'female_padding_mask': g_batch['net_input']['female_padding_mask'],
            'male_positive_padding_mask': g_batch['net_input']['male_positive_padding_mask'],
            'male_negative_padding_mask': g_batch['net_input']['male_negative_padding_mask'],
        })
        net_input.update({
            'anchor_audio': t_batch['net_input']['anchor_audio'],
            'positive_audio': t_batch['net_input']['positive_audio'],
            'negative_audio': t_batch['net_input']['negative_audio'],
            'anchor_padding_mask': t_batch['net_input']['anchor_padding_mask'],
            'positive_padding_mask': t_batch['net_input']['positive_padding_mask'],
            'negative_padding_mask': t_batch['net_input']['negative_padding_mask'],
            'positive_valid_mask': t_batch['net_input']['positive_valid_mask'],
        })

        # Optionally add ASR stream (with probability asr_ratio)
        total_ntokens = g_batch.get('ntokens', 0) + t_batch.get('ntokens', 0)
        nsentences = max(g_batch.get('nsentences', 0), t_batch.get('nsentences', 0))
        if self.asr_ds is not None and random.random() < self.asr_ratio:
            # Sample ASR examples (match batch size)
            batch_size = nsentences if nsentences and nsentences > 0 else len(samples)
            if batch_size <= 0:
                batch_size = 1
            idxs = [random.randrange(len(self.asr_ds)) for _ in range(batch_size)]
            asr_samples = [self.asr_ds[i] for i in idxs]
            asr_batch = self.asr_ds.collater(asr_samples)
            if asr_batch:
                net_input.update({
                    'asr_audio': asr_batch['net_input']['source'],
                    'asr_padding_mask': asr_batch['net_input'].get('padding_mask'),
                })
                # Targets
                if 'target' in asr_batch and 'target_lengths' in asr_batch:
                    net_input.update({
                        'asr_target': asr_batch['target'],
                        'asr_target_lengths': asr_batch['target_lengths'],
                    })
                total_ntokens += asr_batch.get('ntokens', 0)
                nsentences = max(nsentences, asr_batch.get('nsentences', batch_size))

        batch = {
            'net_input': net_input,
            'ntokens': total_ntokens,
            'nsentences': nsentences,
            'id': g_batch.get('id', None),
        }
        return batch


