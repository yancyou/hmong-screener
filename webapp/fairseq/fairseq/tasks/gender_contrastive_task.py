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
from fairseq.data.audio.gender_contrastive_dataset import (
    GenderContrastiveDataset, 
    collate_gender_contrastive_samples
)

logger = logging.getLogger(__name__)


@dataclass
class GenderContrastiveTaskConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, 
        metadata={"help": "path to data directory containing audio files"}
    )
    transcript_file: str = field(
        default="hmong_wordlist_transcripts.txt",
        metadata={"help": "path to transcript file with filename and word pairs"}
    )
    sample_rate: int = field(
        default=16000, 
        metadata={"help": "target sample rate. audio files will be up/down sampled to this rate"}
    )
    num_negatives: int = field(
        default=100,
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


@register_task("gender_contrastive", dataclass=GenderContrastiveTaskConfig)
class GenderContrastiveTask(FairseqTask):
    """
    Task for gender contrastive learning.
    
    This task loads paired male/female audio data for the same words
    and trains a model to pull same-word embeddings together while
    pushing different-word embeddings apart.
    """
    
    cfg: GenderContrastiveTaskConfig
    
    def __init__(self, cfg: GenderContrastiveTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Create a dummy dictionary (required by Fairseq but not used for audio tasks)
        self.dictionary = Dictionary()
        
        logger.info(f"Gender contrastive task initialized")
        logger.info(f"Data directory: {cfg.data}")
        logger.info(f"Transcript file: {cfg.transcript_file}")
        logger.info(f"Sample rate: {cfg.sample_rate}")
        logger.info(f"Number of negatives: {cfg.num_negatives}")
    
    @classmethod
    def setup_task(cls, cfg: GenderContrastiveTaskConfig, **kwargs):
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
        # Choose transcript file based on split
        if split == "train":
            transcript_file = self.cfg.transcript_file
        elif split == "val" or split == "valid":
            # Use validation transcript file
            transcript_file = self.cfg.transcript_file.replace("_train.txt", "_val.txt")
        else:
            # For other splits, use the main transcript file
            transcript_file = self.cfg.transcript_file
        
        # Try to find the transcript file
        transcript_path = os.path.join(self.cfg.data, transcript_file)
        if not os.path.exists(transcript_path):
            # Try the transcript file as an absolute path
            if os.path.exists(transcript_file):
                transcript_path = transcript_file
            else:
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        logger.info(f"Loading {split} dataset from {self.cfg.data}")
        logger.info(f"Using transcript file: {transcript_path}")
        
        # Disable augmentation for validation
        enable_augmentation = self.cfg.enable_augmentation and split == "train"
        
        dataset = GenderContrastiveDataset(
            data_dir=self.cfg.data,
            transcript_file=transcript_path,
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
            collate_fn=collate_gender_contrastive_samples,
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
            female_len = len(sample['female_audio'])
            male_pos_len = len(sample['male_positive_audio'])
            max_neg_len = max(len(neg_audio) for neg_audio in sample['male_negative_audios'])
            
            max_sample_len = max(female_len, male_pos_len, max_neg_len)
            
            if max_sample_len <= max_positions:
                filtered_indices.append(idx)
            elif not ignore_invalid_inputs:
                logger.warning(
                    f"Sample {idx} has length {max_sample_len} > max_positions {max_positions}, skipping"
                )
        
        return filtered_indices 