# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset

from fairseq.data import FairseqDataset, data_utils

# Import our audio augmentation pipeline (local copy)
from .augmentation import AudioAugmentations, get_conservative_transforms, get_validation_transforms

logger = logging.getLogger(__name__)


class ToneContrastiveDataset(FairseqDataset):
    """
    Dataset for tone contrastive learning.
    
    For each sample, provides:
    - Anchor audio (one tone variant)
    - Positive audios (other tone variants of the same base word)
    - Negative audios (different base words)
    
    Uses tone metadata with 7-tone groups where every 7 words = same base word.
    """
    
    def __init__(
        self,
        data_dir: str,
        transcript_file: str,
        tone_metadata_file: str,
        sample_rate: int = 16000,
        num_negatives: int = 20,
        max_audio_length: Optional[int] = None,
        min_audio_length: Optional[int] = None,
        normalize: bool = True,
        seed: int = 42,
        # Audio augmentation parameters
        enable_augmentation: bool = False,
        augmentation_config: Optional[Dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transcript_file = transcript_file
        self.tone_metadata_file = tone_metadata_file
        self.sample_rate = sample_rate
        self.num_negatives = num_negatives
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.normalize = normalize
        # Force deterministic seed for distributed training consistency
        self.seed = 42
        if seed is not None and seed != 42:
            logger.info(f"Overriding provided seed {seed} with fixed seed 42 for deterministic training")
        
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Setup audio augmentation pipeline
        self.enable_augmentation = enable_augmentation
        if enable_augmentation:
            if augmentation_config is None:
                # Use conservative defaults for tone contrastive learning
                # Disable pitch shifting since it can affect tonal information
                conservative_config = {
                    'aug_prob': 0.3,
                    'noise_factor_range': (0.002, 0.01),
                    'time_stretch_range': (0.98, 1.02),
                    'volume_gain_range': (-1.0, 1.0),
                    'enable_pitch_shift': False,  # Critical: disable for tonal languages!
                }
                self.audio_transform = AudioAugmentations(
                    sample_rate=sample_rate,
                    training=True,
                    **conservative_config
                )
                logger.info("Using conservative audio augmentations for tone contrastive learning (pitch shift disabled)")
            else:
                # Custom augmentation config
                self.audio_transform = AudioAugmentations(
                    sample_rate=sample_rate,
                    training=True,
                    **augmentation_config
                )
                logger.info(f"Using custom audio augmentations: {augmentation_config}")
        else:
            # No augmentation (validation mode)
            self.audio_transform = get_validation_transforms()
            logger.info("Audio augmentation disabled")
        
        # Load tone metadata and create tone groups
        self.tone_groups = self._load_tone_metadata()
        self.tone_samples = self._create_tone_samples()
        self.negative_pool = self._build_negative_pool()
        
        logger.info(f"Loaded {len(self.tone_groups)} tone groups")
        logger.info(f"Created {len(self.tone_samples)} tone contrastive samples")
        logger.info(f"Negative pool size: {len(self.negative_pool)}")
    
    def train(self):
        """Set dataset to training mode (enables augmentations)"""
        if hasattr(self.audio_transform, 'train'):
            self.audio_transform.train()
        return self
    
    def eval(self):
        """Set dataset to evaluation mode (disables augmentations)"""
        if hasattr(self.audio_transform, 'eval'):
            self.audio_transform.eval()
        return self
    
    def _load_tone_metadata(self) -> List[Dict[str, Any]]:
        """Load tone metadata from JSON file"""
        try:
            with open(self.tone_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            tone_groups = metadata['tone_groups']
            logger.info(f"Loaded {len(tone_groups)} tone groups from {self.tone_metadata_file}")
            
            return tone_groups
        
        except Exception as e:
            logger.error(f"Error loading tone metadata from {self.tone_metadata_file}: {e}")
            return []
    
    def _find_audio_file(self, filename: str) -> Optional[Path]:
        """Find audio file with various extensions and gender suffixes"""
        possible_extensions = ['.wav', '.mp3', '.flac', '']
        possible_suffixes = ['', '_tone_male', '_tone_female', '_tone__male', '_tone__female']
        
        for suffix in possible_suffixes:
            for ext in possible_extensions:
                audio_path = self.data_dir / f"{filename}{suffix}{ext}"
                if audio_path.exists():
                    return audio_path
        
        return None
    
    def _create_tone_samples(self) -> List[Dict[str, Any]]:
        """Create training samples from tone groups"""
        tone_samples = []
        
        for group in self.tone_groups:
            base_word = group['base_word']
            tone_variants = group['tone_variants']
            
            # Check which tone variants have existing audio files
            valid_variants = {}
            for tone_name, tone_data in tone_variants.items():
                filename = tone_data['filename']
                audio_path = self._find_audio_file(filename)
                
                if audio_path is not None:
                    valid_variants[tone_name] = {
                        'word': tone_data['word'],
                        'filename': filename,
                        'audio_path': str(audio_path),
                        'tone_number': tone_data['tone_number'],
                        'tone_marker': tone_data['tone_marker']
                    }
                else:
                    logger.warning(f"Audio file not found for {filename}")
            
            # Need at least 2 tone variants to create positive pairs
            if len(valid_variants) < 2:
                logger.warning(f"Skipping tone group '{base_word}' - only {len(valid_variants)} valid variants")
                continue
            
            # Create samples: each tone variant can be an anchor
            for anchor_tone, anchor_data in valid_variants.items():
                # Get positive samples (other tone variants of same base word)
                positive_variants = {k: v for k, v in valid_variants.items() if k != anchor_tone}
                
                if len(positive_variants) > 0:
                    tone_samples.append({
                        'group_id': group['group_id'],
                        'base_word': base_word,
                        'anchor_tone': anchor_tone,
                        'anchor_data': anchor_data,
                        'positive_variants': positive_variants,
                    })
        
        return tone_samples
    
    def _build_negative_pool(self) -> List[Dict[str, Any]]:
        """Build pool of all audio files for negative sampling"""
        negative_pool = []
        
        for group in self.tone_groups:
            for tone_name, tone_data in group['tone_variants'].items():
                filename = tone_data['filename']
                audio_path = self._find_audio_file(filename)
                
                if audio_path is not None:
                    negative_pool.append({
                        'group_id': group['group_id'],
                        'base_word': group['base_word'],
                        'word': tone_data['word'],
                        'filename': filename,
                        'audio_path': str(audio_path),
                        'tone_name': tone_name,
                        'tone_number': tone_data['tone_number']
                    })
        
        return negative_pool
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file with optional augmentation"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Apply length constraints
            if self.min_audio_length and len(audio) < self.min_audio_length:
                # Pad short audio
                padding = self.min_audio_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
            
            if self.max_audio_length and len(audio) > self.max_audio_length:
                # Truncate long audio
                audio = audio[:self.max_audio_length]
            
            # Apply audio augmentations (torchvision-style transform)
            audio = self.audio_transform(audio)
            
            # Normalize audio (after augmentation to account for volume changes)
            if self.normalize:
                audio = audio / (np.abs(audio).max() + 1e-8)
            
            return audio.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            return np.array([], dtype=np.float32)
    
    def _sample_negatives(self, anchor_group_id: int, num_negatives: int, seed: int = None) -> List[Dict[str, Any]]:
        """Sample negative audio examples (different base words)"""
        # Get all audios from different base words (different groups)
        negative_pool = [item for item in self.negative_pool if item['group_id'] != anchor_group_id]
        
        # Use a deterministic seed for consistent sampling across distributed processes
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        if len(negative_pool) < num_negatives:
            # Sample with replacement if we don't have enough unique negatives
            negatives = rng.choices(negative_pool, k=num_negatives)
            logger.warning(f"Not enough unique negatives for group {anchor_group_id}, sampling with replacement")
        else:
            # Sample without replacement
            negatives = rng.sample(negative_pool, num_negatives)
        
        return negatives
    
    def __len__(self) -> int:
        return len(self.tone_samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Returns:
            Dict containing:
            - anchor_audio: torch.Tensor [T]
            - positive_audios: List[torch.Tensor] [num_positives x T]
            - negative_audios: List[torch.Tensor] [num_negatives x T]
            - base_word: str
            - anchor_tone: str
            - positive_tones: List[str]
            - anchor_path: str
            - positive_paths: List[str]
            - negative_paths: List[str]
        """
        sample = self.tone_samples[index]
        base_word = sample['base_word']
        group_id = sample['group_id']
        anchor_data = sample['anchor_data']
        positive_variants = sample['positive_variants']
        
        # Load anchor audio
        anchor_audio = self._load_audio(anchor_data['audio_path'])
        
        # Load positive audios (other tone variants of same base word)
        positive_audios = []
        positive_paths = []
        positive_tones = []
        
        for tone_name, tone_data in positive_variants.items():
            pos_audio = self._load_audio(tone_data['audio_path'])
            if len(pos_audio) > 0:  # Only add if audio loaded successfully
                positive_audios.append(torch.from_numpy(pos_audio))
                positive_paths.append(tone_data['audio_path'])
                positive_tones.append(tone_name)
        
        # Sample and load negative audios (different base words)
        # Use index as seed to ensure consistent sampling across distributed processes
        deterministic_seed = (abs(self.seed) + index) % (2**32 - 1)
        negative_samples = self._sample_negatives(group_id, self.num_negatives, seed=deterministic_seed)
        negative_audios = []
        negative_paths = []
        
        for neg_sample in negative_samples:
            neg_audio = self._load_audio(neg_sample['audio_path'])
            if len(neg_audio) > 0:  # Only add if audio loaded successfully
                negative_audios.append(torch.from_numpy(neg_audio))
                negative_paths.append(neg_sample['audio_path'])
        
        # Ensure we have enough negatives
        while len(negative_audios) < self.num_negatives:
            # Pad with duplicates if necessary
            if len(negative_audios) > 0:
                negative_audios.append(negative_audios[-1].clone())
                negative_paths.append(negative_paths[-1])
            else:
                # Fallback: create zero audio
                zero_audio = torch.zeros_like(torch.from_numpy(anchor_audio))
                negative_audios.append(zero_audio)
                negative_paths.append("ZERO_AUDIO")
        
        # Ensure we have at least one positive
        if len(positive_audios) == 0:
            # Fallback: duplicate anchor as positive
            positive_audios.append(torch.from_numpy(anchor_audio))
            positive_paths.append(anchor_data['audio_path'])
            positive_tones.append(sample['anchor_tone'])
        
        return {
            'anchor_audio': torch.from_numpy(anchor_audio),
            'positive_audios': positive_audios,
            'negative_audios': negative_audios[:self.num_negatives],
            'base_word': base_word,
            'anchor_tone': sample['anchor_tone'],
            'positive_tones': positive_tones,
            'anchor_path': anchor_data['audio_path'],
            'positive_paths': positive_paths,
            'negative_paths': negative_paths[:self.num_negatives],
        }


def collate_tone_contrastive_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for tone contrastive dataset.
    
    Handles variable-length audio sequences and variable number of positives.
    """
    if not samples:
        return {}
    
    batch_size = len(samples)
    num_negatives = len(samples[0]['negative_audios'])
    
    # Find maximum number of positives and maximum audio lengths
    max_num_positives = max(len(sample['positive_audios']) for sample in samples)
    
    anchor_max_len = max(sample['anchor_audio'].shape[0] for sample in samples)
    pos_max_len = max(
        max(pos_audio.shape[0] for pos_audio in sample['positive_audios'])
        for sample in samples if len(sample['positive_audios']) > 0
    )
    neg_max_len = max(
        max(neg_audio.shape[0] for neg_audio in sample['negative_audios'])
        for sample in samples
    )
    
    # Use the global maximum for consistent padding
    max_len = max(anchor_max_len, pos_max_len, neg_max_len)
    
    # Initialize tensors
    anchor_audios = torch.zeros(batch_size, max_len)
    positive_audios = torch.zeros(batch_size, max_num_positives, max_len)
    negative_audios = torch.zeros(batch_size, num_negatives, max_len)
    
    # Padding masks (True for padding positions)
    anchor_padding_masks = torch.ones(batch_size, max_len, dtype=torch.bool)
    positive_padding_masks = torch.ones(batch_size, max_num_positives, max_len, dtype=torch.bool)
    negative_padding_masks = torch.ones(batch_size, num_negatives, max_len, dtype=torch.bool)
    
    # Mask for valid positives (False for valid, True for padding)
    positive_valid_mask = torch.ones(batch_size, max_num_positives, dtype=torch.bool)
    
    # Fill tensors and create masks
    base_words = []
    anchor_tones = []
    positive_tones = []
    anchor_paths = []
    positive_paths = []
    negative_paths = []
    
    for i, sample in enumerate(samples):
        # Anchor audio
        anchor_len = sample['anchor_audio'].shape[0]
        anchor_audios[i, :anchor_len] = sample['anchor_audio']
        anchor_padding_masks[i, :anchor_len] = False
        
        # Positive audios
        for j, pos_audio in enumerate(sample['positive_audios']):
            pos_len = pos_audio.shape[0]
            positive_audios[i, j, :pos_len] = pos_audio
            positive_padding_masks[i, j, :pos_len] = False
            positive_valid_mask[i, j] = False  # Mark as valid
        
        # Negative audios
        for j, neg_audio in enumerate(sample['negative_audios']):
            neg_len = neg_audio.shape[0]
            negative_audios[i, j, :neg_len] = neg_audio
            negative_padding_masks[i, j, :neg_len] = False
        
        # Metadata
        base_words.append(sample['base_word'])
        anchor_tones.append(sample['anchor_tone'])
        positive_tones.append(sample['positive_tones'])
        anchor_paths.append(sample['anchor_path'])
        positive_paths.append(sample['positive_paths'])
        negative_paths.append(sample['negative_paths'])
    
    return {
        'net_input': {
            'anchor_audio': anchor_audios,
            'positive_audio': positive_audios,
            'negative_audio': negative_audios,
            'anchor_padding_mask': anchor_padding_masks,
            'positive_padding_mask': positive_padding_masks,
            'negative_padding_mask': negative_padding_masks,
            'positive_valid_mask': positive_valid_mask,  # Which positives are valid
        },
        'base_word_labels': base_words,
        'anchor_tones': anchor_tones,
        'positive_tones': positive_tones,
        'anchor_paths': anchor_paths,
        'positive_paths': positive_paths,
        'negative_paths': negative_paths,
        'ntokens': batch_size * max_len,  # Total number of audio frames
        'nsentences': batch_size,
        'id': torch.arange(batch_size),
    }
