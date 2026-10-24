# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any, List

from omegaconf import MISSING

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.audio.tone_contrastive_dataset import (
    ToneContrastiveDataset, 
    collate_tone_contrastive_samples
)

logger = logging.getLogger(__name__)


@dataclass
class ToneContrastiveTaskConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, 
        metadata={"help": "path to data directory containing audio files"}
    )
    transcript_file: str = field(
        default="hmong_new_wordlist_transcripts.txt",
        metadata={"help": "path to transcript file with filename and word pairs"}
    )
    tone_metadata_file: str = field(
        default="hmong_tone_metadata.json",
        metadata={"help": "path to tone metadata JSON file with tone group information"}
    )
    sample_rate: int = field(
        default=16000, 
        metadata={"help": "target sample rate. audio files will be up/down sampled to this rate"}
    )
    num_negatives: int = field(
        default=20,
        metadata={"help": "number of negative samples for contrastive learning"}
    )
    max_audio_length: Optional[int] = field(
        default=320000,  # 20 seconds at 16kHz
        metadata={"help": "maximum audio length in samples"}
    )
    min_audio_length: Optional[int] = field(
        default=1600,  # 0.1 seconds at 16kHz
        metadata={"help": "minimum audio length in samples"}
    )
    normalize: bool = field(
        default=True,
        metadata={"help": "whether to normalize audio"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "random seed for reproducibility"}
    )
    # Audio augmentation parameters
    enable_augmentation: bool = field(
        default=False,
        metadata={"help": "whether to enable audio augmentation during training"}
    )
    augmentation_config: Optional[Any] = field(
        default=None,
        metadata={"help": "augmentation configuration parameters"}
    )


@register_task("tone_contrastive", dataclass=ToneContrastiveTaskConfig)
class ToneContrastiveTask(FairseqTask):
    """
    Task for tone contrastive learning.
    
    This task loads tone variant audio data (every 7 words = same base word with different tones)
    and trains a model to pull same-base-word embeddings together while
    pushing different-base-word embeddings apart.
    """
    
    cfg: ToneContrastiveTaskConfig
    
    def __init__(self, cfg: ToneContrastiveTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Create a dummy dictionary (required by Fairseq but not used for audio tasks)
        self.dictionary = Dictionary()
        
        logger.info(f"Tone contrastive task initialized")
        logger.info(f"Data directory: {cfg.data}")
        logger.info(f"Transcript file: {cfg.transcript_file}")
        logger.info(f"Tone metadata file: {cfg.tone_metadata_file}")
        logger.info(f"Sample rate: {cfg.sample_rate}")
        logger.info(f"Number of negatives: {cfg.num_negatives}")
    
    @classmethod
    def setup_task(cls, cfg: ToneContrastiveTaskConfig, **kwargs):
        """Setup the task (called once per experiment)."""
        return cls(cfg)
    
    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        """
        Load a given dataset split.
        
        Args:
            split (str): name of the split (e.g. "train", "valid", "test")
            combine (bool): whether to combine multiple datasets
            **kwargs: additional arguments
        """
        # Choose files based on split
        if split == "train":
            transcript_file = self.cfg.transcript_file
            tone_metadata_file = self.cfg.tone_metadata_file
        elif split == "val" or split == "valid":
            # Use validation files
            transcript_file = self.cfg.transcript_file  # Keep same transcript for tone
            tone_metadata_file = self.cfg.tone_metadata_file.replace("_train.json", "_val.json")
        else:
            # For other splits, use the main files
            transcript_file = self.cfg.transcript_file
            tone_metadata_file = self.cfg.tone_metadata_file
        
        # Find transcript file
        transcript_path = os.path.join(self.cfg.data, transcript_file)
        if not os.path.exists(transcript_path):
            # Try the transcript file as an absolute path
            if os.path.exists(transcript_file):
                transcript_path = transcript_file
            else:
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        # Find tone metadata file
        tone_metadata_path = os.path.join(self.cfg.data, tone_metadata_file)
        if not os.path.exists(tone_metadata_path):
            # Try the metadata file as an absolute path
            if os.path.exists(tone_metadata_file):
                tone_metadata_path = tone_metadata_file
            else:
                raise FileNotFoundError(f"Tone metadata file not found: {tone_metadata_path}")
        
        logger.info(f"Loading {split} dataset from {self.cfg.data}")
        logger.info(f"Using transcript file: {transcript_path}")
        logger.info(f"Using tone metadata file: {tone_metadata_path}")
        
        # Disable augmentation for validation
        enable_augmentation = self.cfg.enable_augmentation and split == "train"
        
        dataset = ToneContrastiveDataset(
            data_dir=self.cfg.data,
            transcript_file=transcript_path,
            tone_metadata_file=tone_metadata_path,
            sample_rate=self.cfg.sample_rate,
            num_negatives=self.cfg.num_negatives,
            max_audio_length=self.cfg.max_audio_length,
            min_audio_length=self.cfg.min_audio_length,
            normalize=self.cfg.normalize,
            seed=self.cfg.seed + hash(split),  # Different seed for each split
            enable_augmentation=enable_augmentation,
            augmentation_config=self.cfg.augmentation_config,
        )
        
        self.datasets[split] = dataset
        logger.info(f"Loaded {len(dataset)} samples for {split} split (augmentation: {enable_augmentation})")
    
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
        """
        Get an iterator that yields batches of the given size.
        """
        from fairseq.data import iterators
        
        # Use max_sentences from task config if not provided
        if max_tokens is None:
            if max_sentences is None:
                max_sentences = 1
            batch_size = max_sentences
        else:
            # For audio tasks, max_tokens doesn't make as much sense
            # Use max_sentences as batch size
            batch_size = max_sentences or 1
        
        logger.info(f"Creating batch iterator with batch size: {batch_size}")
        
        # Create batch sampler
        indices = list(range(len(dataset)))
        
        # Shuffle indices if this is training
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)
        
        import numpy as np
        import torch
        import random
        
        # Set all random seeds for deterministic behavior
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        
        np.random.shuffle(indices)
        
        # Create batches
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batches.append(batch_indices)
        
        # Create iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=collate_tone_contrastive_samples,
            batch_sampler=batches,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
    
    @property
    def source_dictionary(self):
        """Return the source dictionary (dummy for audio tasks)."""
        return self.dictionary
    
    @property
    def target_dictionary(self):
        """Return the target dictionary (dummy for audio tasks)."""
        return self.dictionary
    
    def max_positions(self):
        """Return the max input length allowed by the task."""
        return self.cfg.max_audio_length or float('inf')
    
    def filter_indices_by_size(self, indices, dataset, max_positions=None, ignore_invalid_inputs=False):
        """
        Filter examples that are too large.
        """
        if max_positions is None:
            max_positions = self.max_positions()
        
        if max_positions == float('inf'):
            return indices
        
        # Filter by audio length
        filtered_indices = []
        for idx in indices:
            sample = dataset[idx]
            anchor_len = len(sample['anchor_audio'])
            max_pos_len = max(len(pos_audio) for pos_audio in sample['positive_audios']) if sample['positive_audios'] else 0
            max_neg_len = max(len(neg_audio) for neg_audio in sample['negative_audios']) if sample['negative_audios'] else 0
            
            max_sample_len = max(anchor_len, max_pos_len, max_neg_len)
            
            if max_sample_len <= max_positions:
                filtered_indices.append(idx)
            elif not ignore_invalid_inputs:
                logger.warning(
                    f"Sample {idx} has length {max_sample_len} > max_positions {max_positions}, skipping"
                )
        
        return filtered_indices
