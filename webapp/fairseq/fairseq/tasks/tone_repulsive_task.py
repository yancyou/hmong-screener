# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import json
import random
import numpy as np
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any

from omegaconf import MISSING
import torch
import torchaudio

from fairseq import utils
from fairseq.data import Dictionary, FairseqDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)


@dataclass
class ToneRepulsiveTaskConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={"help": "path to data directory containing audio files"}
    )
    tone_metadata_file: str = field(
        default=MISSING,
        metadata={"help": "path to tone metadata JSON file"}
    )
    sample_rate: int = field(
        default=16000,
        metadata={"help": "target sample rate"}
    )
    num_positives: int = field(
        default=2,
        metadata={"help": "number of positive samples (same word+tone augmentations)"}
    )
    num_hard_negatives: int = field(
        default=4,
        metadata={"help": "number of hard negatives (same word, different tone)"}
    )
    num_soft_negatives: int = field(
        default=10,
        metadata={"help": "number of soft negatives (different words)"}
    )
    max_audio_length: Optional[int] = field(
        default=320000,
        metadata={"help": "maximum audio length in samples"}
    )
    min_audio_length: Optional[int] = field(
        default=1600,
        metadata={"help": "minimum audio length in samples"}
    )
    normalize: bool = field(
        default=True,
        metadata={"help": "whether to normalize audio"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "random seed"}
    )
    enable_augmentation: bool = field(
        default=False,
        metadata={"help": "whether to enable audio augmentation during training"}
    )
    augmentation_config: Optional[Any] = field(
        default=None,
        metadata={"help": "augmentation configuration parameters"}
    )


@register_task("tone_repulsive", dataclass=ToneRepulsiveTaskConfig)
class ToneRepulsiveTask(FairseqTask):
    """
    Task for tone-discriminative contrastive learning.
    Pulls same word+tone together, pushes different tones apart.
    """

    cfg: ToneRepulsiveTaskConfig

    def __init__(self, cfg: ToneRepulsiveTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = Dictionary()

    @classmethod
    def setup_task(cls, cfg: ToneRepulsiveTaskConfig, **kwargs):
        return cls(cfg)

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        """Load dataset for given split"""
        # Load tone metadata
        metadata_path = self.cfg.tone_metadata_file
        if split == "valid" or split == "val":
            metadata_path = metadata_path.replace("_train.json", "_val.json")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Enable augmentation only for training
        enable_augmentation = self.cfg.enable_augmentation and split == "train"

        dataset = ToneRepulsiveDataset(
            data_dir=self.cfg.data,
            tone_metadata=metadata,
            sample_rate=self.cfg.sample_rate,
            num_positives=self.cfg.num_positives,
            num_hard_negatives=self.cfg.num_hard_negatives,
            num_soft_negatives=self.cfg.num_soft_negatives,
            max_audio_length=self.cfg.max_audio_length,
            min_audio_length=self.cfg.min_audio_length,
            normalize=self.cfg.normalize,
            seed=self.cfg.seed + hash(split),
            enable_augmentation=enable_augmentation,
            augmentation_config=self.cfg.augmentation_config,
        )

        self.datasets[split] = dataset
        logger.info(f"Loaded {len(dataset)} samples for {split} (augmentation: {enable_augmentation})")

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class ToneRepulsiveDataset(FairseqDataset):
    """
    Dataset for tone repulsive learning.
    Each sample contains:
    - anchor: one audio file
    - positives: same word+tone (augmented versions or same files)
    - hard_negatives: same base word, different tones
    - soft_negatives: different base words
    """

    def __init__(self, data_dir, tone_metadata, sample_rate=16000,
                 num_positives=2, num_hard_negatives=4, num_soft_negatives=10,
                 max_audio_length=320000, min_audio_length=1600,
                 normalize=True, seed=42, enable_augmentation=False, augmentation_config=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.num_positives = num_positives
        self.num_hard_negatives = num_hard_negatives
        self.num_soft_negatives = num_soft_negatives
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.normalize = normalize
        self.rng = random.Random(seed)
        
        # Setup audio augmentation
        self.enable_augmentation = enable_augmentation
        if enable_augmentation:
            from fairseq.data.audio.augmentation import AudioAugmentations
            if augmentation_config is None:
                # Conservative defaults for tone (no pitch shift!)
                augmentation_config = {
                    'aug_prob': 0.3,
                    'noise_factor_range': (0.002, 0.01),
                    'time_stretch_range': (0.98, 1.02),
                    'volume_gain_range': (-1.5, 1.5),
                    'enable_pitch_shift': False,  # Critical: disable for tonal languages!
                    'enable_noise': True,
                    'enable_time_stretch': True,
                    'enable_volume': True,
                }
            self.audio_transform = AudioAugmentations(
                sample_rate=sample_rate,
                training=True,
                **augmentation_config
            )
            logger.info(f"Augmentation enabled for tone repulsive dataset")
        else:
            self.audio_transform = None
            logger.info(f"Augmentation disabled for tone repulsive dataset")

        # Build index from tone_metadata
        self.tone_groups = tone_metadata['tone_groups']
        self.samples = []  # List of (group_idx, tone_key, filename)

        for group_idx, group in enumerate(self.tone_groups):
            for tone_key, tone_info in group['tone_variants'].items():
                filename = tone_info['filename']
                self.samples.append((group_idx, tone_key, filename))

    def __len__(self):
        return len(self.samples)
    
    def num_tokens(self, index):
        """Return the number of tokens in a sample. Required by FairseqDataset."""
        return self.max_audio_length if self.max_audio_length else 320000
    
    def size(self, index):
        """Return an example's size as a float or tuple. Required by FairseqDataset."""
        return self.max_audio_length if self.max_audio_length else 320000
    
    def ordered_indices(self):
        """Return an ordered list of indices. Required by FairseqDataset."""
        return np.arange(len(self), dtype=np.int64)
    
    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching. Required by FairseqDataset."""
        return False
    
    def collater(self, samples):
        """Collate a list of samples into a batch. Required by FairseqDataset."""
        return collate_tone_repulsive_samples(samples)

    def find_same_tone_variants(self, base_filename):
        """
        Find real positive variants (same word+tone, e.g., male/female versions).
        Returns list of variant filenames.
        
        Example:
        base_filename = "Hmong_Word_List_4_Full_Recording_word099"
        variants = ["Hmong_Word_List_4_Full_Recording_word099_tone_male",
                    "Hmong_Word_List_4_Full_Recording_word099_tone_female"]
        """
        variants = []
        possible_suffixes = ['_tone_male', '_tone_female', '_tone__male', '_tone__female']
        
        for suffix in possible_suffixes:
            variant_filename = base_filename + suffix
            # Check if file exists with any extension
            for ext in ['.wav', '.flac', '.mp3']:
                audio_path = os.path.join(self.data_dir, variant_filename + ext)
                if os.path.exists(audio_path):
                    variants.append(variant_filename)
                    break
        
        return variants

    def load_audio(self, filename, apply_augmentation=False):
        """Load and preprocess audio, optionally with augmentation"""
        # Try common extensions
        audio_path = None
        for ext in ['.wav', '.flac', '.mp3']:
            candidate_path = os.path.join(self.data_dir, filename + ext)
            if os.path.exists(candidate_path):
                audio_path = candidate_path
                break
        
        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found: {filename}")
        
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # [T]
        
        # Apply augmentation if requested (before normalization)
        if apply_augmentation and self.audio_transform is not None:
            waveform_np = waveform.numpy()
            waveform_np = self.audio_transform(waveform_np)
            waveform = torch.from_numpy(waveform_np)

        # Normalize
        if self.normalize and waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        # Trim or pad
        if self.max_audio_length and waveform.size(0) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        if self.min_audio_length and waveform.size(0) < self.min_audio_length:
            pad_len = self.min_audio_length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform

    def __getitem__(self, idx):
        group_idx, anchor_tone_key, anchor_filename = self.samples[idx]
        group = self.tone_groups[group_idx]

        # Load anchor (without augmentation for anchor)
        anchor_audio = self.load_audio(anchor_filename, apply_augmentation=False)

        # Sample positives: same word+tone
        # Strategy 1: Find real variants (e.g., male/female versions)
        # Strategy 2: Apply augmentation to create more positives
        positives = []
        
        # Try to find real same-tone variants
        real_variants = self.find_same_tone_variants(anchor_filename)
        for variant_filename in real_variants:
            if len(positives) >= self.num_positives:
                break
            try:
                variant_audio = self.load_audio(variant_filename, apply_augmentation=False)
                positives.append(variant_audio)
            except FileNotFoundError:
                pass
        
        # Fill remaining with augmented versions
        while len(positives) < self.num_positives:
            if self.enable_augmentation and self.audio_transform is not None:
                # Load with augmentation
                aug_audio = self.load_audio(anchor_filename, apply_augmentation=True)
                positives.append(aug_audio)
            else:
                # Fallback: add small noise
                pos_audio = anchor_audio + torch.randn_like(anchor_audio) * 0.001
                positives.append(pos_audio)

        # Sample hard negatives: same base word, different tones (no augmentation)
        hard_negatives = []
        available_tones = [k for k in group['tone_variants'].keys() if k != anchor_tone_key]
        sampled_hard = self.rng.sample(available_tones, min(self.num_hard_negatives, len(available_tones)))
        for tone_key in sampled_hard:
            hard_neg_filename = group['tone_variants'][tone_key]['filename']
            hard_neg_audio = self.load_audio(hard_neg_filename, apply_augmentation=False)
            hard_negatives.append(hard_neg_audio)

        # Pad hard negatives if needed
        while len(hard_negatives) < self.num_hard_negatives:
            hard_negatives.append(torch.zeros_like(anchor_audio))

        # Sample soft negatives: different base words (no augmentation)
        soft_negatives = []
        available_groups = [i for i in range(len(self.tone_groups)) if i != group_idx]
        sampled_soft_groups = self.rng.sample(available_groups, min(self.num_soft_negatives, len(available_groups)))
        for soft_group_idx in sampled_soft_groups:
            soft_group = self.tone_groups[soft_group_idx]
            soft_tone_key = self.rng.choice(list(soft_group['tone_variants'].keys()))
            soft_neg_filename = soft_group['tone_variants'][soft_tone_key]['filename']
            soft_neg_audio = self.load_audio(soft_neg_filename, apply_augmentation=False)
            soft_negatives.append(soft_neg_audio)

        # Pad soft negatives if needed
        while len(soft_negatives) < self.num_soft_negatives:
            soft_negatives.append(torch.zeros_like(anchor_audio))

        # Get tone label from anchor_tone_key
        # anchor_tone_key is like "tone1", "tone2", ..., "tone7"
        # Extract the tone number (1-7) and convert to 0-indexed (0-6)
        tone_number = group['tone_variants'][anchor_tone_key]['tone_number']  # 1-7
        tone_label = tone_number - 1  # Convert to 0-indexed (0-6)

        return {
            'anchor_audio': anchor_audio,
            'positive_audios': positives,
            'hard_negative_audios': hard_negatives,
            'soft_negative_audios': soft_negatives,
            'anchor_tone_label': tone_label,
        }


def collate_tone_repulsive_samples(samples):
    """Collate function for batching"""
    if len(samples) == 0:
        return {}

    # Find max lengths
    max_anchor_len = max(s['anchor_audio'].size(0) for s in samples)
    max_pos_len = max(max(p.size(0) for p in s['positive_audios']) for s in samples)
    max_hard_len = max(max(h.size(0) for h in s['hard_negative_audios']) for s in samples)
    max_soft_len = max(max(sn.size(0) for sn in s['soft_negative_audios']) for s in samples)

    batch_size = len(samples)
    num_pos = len(samples[0]['positive_audios'])
    num_hard = len(samples[0]['hard_negative_audios'])
    num_soft = len(samples[0]['soft_negative_audios'])

    # Initialize tensors
    anchor_audio = torch.zeros(batch_size, max_anchor_len)
    anchor_padding_mask = torch.ones(batch_size, max_anchor_len, dtype=torch.bool)

    positive_audio = torch.zeros(batch_size, num_pos, max_pos_len)
    positive_padding_mask = torch.ones(batch_size, num_pos, max_pos_len, dtype=torch.bool)
    positive_valid_mask = torch.zeros(batch_size, num_pos, dtype=torch.bool)

    hard_negative_audio = torch.zeros(batch_size, num_hard, max_hard_len)
    hard_negative_padding_mask = torch.ones(batch_size, num_hard, max_hard_len, dtype=torch.bool)
    hard_negative_valid_mask = torch.zeros(batch_size, num_hard, dtype=torch.bool)

    soft_negative_audio = torch.zeros(batch_size, num_soft, max_soft_len)
    soft_negative_padding_mask = torch.ones(batch_size, num_soft, max_soft_len, dtype=torch.bool)
    soft_negative_valid_mask = torch.zeros(batch_size, num_soft, dtype=torch.bool)

    # Tone labels
    anchor_tone_labels = torch.zeros(batch_size, dtype=torch.long)

    # Fill tensors
    for i, sample in enumerate(samples):
        # Anchor
        anchor_len = sample['anchor_audio'].size(0)
        anchor_audio[i, :anchor_len] = sample['anchor_audio']
        anchor_padding_mask[i, :anchor_len] = False

        # Positives
        for j, pos in enumerate(sample['positive_audios']):
            pos_len = pos.size(0)
            positive_audio[i, j, :pos_len] = pos
            positive_padding_mask[i, j, :pos_len] = False
            if pos.abs().sum() > 0:  # Valid if not all zeros
                positive_valid_mask[i, j] = False

        # Hard negatives
        for j, hard in enumerate(sample['hard_negative_audios']):
            hard_len = hard.size(0)
            hard_negative_audio[i, j, :hard_len] = hard
            hard_negative_padding_mask[i, j, :hard_len] = False
            if hard.abs().sum() > 0:
                hard_negative_valid_mask[i, j] = False

        # Soft negatives
        for j, soft in enumerate(sample['soft_negative_audios']):
            soft_len = soft.size(0)
            soft_negative_audio[i, j, :soft_len] = soft
            soft_negative_padding_mask[i, j, :soft_len] = False
            if soft.abs().sum() > 0:
                soft_negative_valid_mask[i, j] = False
        
        # Tone label
        anchor_tone_labels[i] = sample['anchor_tone_label']

    return {
        'net_input': {
            'anchor_audio': anchor_audio,
            'anchor_padding_mask': anchor_padding_mask,
            'positive_audio': positive_audio,
            'positive_padding_mask': positive_padding_mask,
            'positive_valid_mask': positive_valid_mask,
            'hard_negative_audio': hard_negative_audio,
            'hard_negative_padding_mask': hard_negative_padding_mask,
            'hard_negative_valid_mask': hard_negative_valid_mask,
            'soft_negative_audio': soft_negative_audio,
            'soft_negative_padding_mask': soft_negative_padding_mask,
            'soft_negative_valid_mask': soft_negative_valid_mask,
            'anchor_tone_label': anchor_tone_labels,
        },
        'ntokens': batch_size,
        'nsentences': batch_size,
    }

