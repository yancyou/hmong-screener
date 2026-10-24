# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import re
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


class GenderContrastiveDataset(FairseqDataset):
    """
    Dataset for gender contrastive learning.
    
    For each sample, provides:
    - Female audio (anchor)
    - Male audio of the same word (positive)
    - Multiple male audios of different words (negatives)
    """
    
    def __init__(
        self,
        data_dir: str,
        transcript_file: str,
        sample_rate: int = 16000,
        num_negatives: int = 100,
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
        self.sample_rate = sample_rate
        self.num_negatives = num_negatives
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.normalize = normalize
        # Force deterministic seed for distributed training consistency
        # Always use a fixed seed to ensure identical behavior across workers
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
                # Use conservative defaults for gender contrastive learning
                self.audio_transform = get_conservative_transforms(sample_rate=sample_rate)
                logger.info("Using conservative audio augmentations for gender contrastive learning")
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
        
        # Load and process transcript data
        self.speaker_to_id = {}
        self.word_pairs = self._load_word_pairs()
        self.male_audio_pool = self._build_male_audio_pool()
        
        logger.info(f"Loaded {len(self.word_pairs)} word pairs for gender contrastive learning")
        logger.info(f"Male audio pool size: {len(self.male_audio_pool)}")
    
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
    
    def _load_transcript_file(self) -> pd.DataFrame:
        """Load transcript file and extract gender, word, and file path information"""
        try:
            # Load the transcript file (tab-separated)
            df = pd.read_csv(self.transcript_file, sep='\t', header=None, names=['filename', 'word'])
            
            # Extract gender from filename
            def extract_gender(filename):
                if 'Female' in filename or 'female' in filename:
                    return 'female'
                elif 'Male' in filename or 'male' in filename:
                    return 'male'
                else:
                    return 'unknown'
            
            df['gender'] = df['filename'].apply(extract_gender)

            # Extract speaker id from filename/path for GRL speaker-adversarial baselines
            # We keep this generic across:
            # - hmong_data_16k_gendered/ctNNN_... (ctNNN)
            # - cxNNN... (cxNNN)
            # - all_data_gender/06_Female_... (06)
            # - chinese_data_gendered/..._FV1_... (FV1)
            def extract_speaker(filename):
                if not isinstance(filename, str):
                    return None
                s = filename.replace("\\", "/")
                base = os.path.basename(s)
                base_noext = base.rsplit(".", 1)[0] if "." in base else base

                # all_data_gender: treat as coarse speaker groups (user-defined)
                # - 02..09_Male and Hmong_Word_List_*_Male => SAME speaker
                # - 02..09_Female (REAL, no extra suffix after wordNN) => SAME speaker
                # - 02..09_*_(male|female)_tonal / *_male_female / *_female_male (VC, i.e. extra suffix after wordNN)
                #   AND Hmong_Word_List_*_Female (and *_tone_female) => SAME speaker
                #
                # Note: the numeric prefix (02/06/...) is NOT a speaker id.
                if "/all_data_gender/" in s or s.startswith("all_data_gender/") or "/all_data_gendered/" in s or s.startswith("all_data_gendered/"):
                    # wordlist male merges into male-group
                    if re.match(r"^Hmong_Word_List_.*_Male$", base_noext, flags=re.IGNORECASE):
                        return "all_data_gender_male"

                    # wordlist tone_female belongs to the FEMALE speaker group (speaker=..._tone_female)
                    if re.match(r"^Hmong_Word_List_.*_tone_female$", base_noext, flags=re.IGNORECASE):
                        return "all_data_gender_female"

                    # wordlist Female belongs to the TONAL_FEMALE speaker group (per user spec)
                    if re.match(r"^Hmong_Word_List_.*_Female$", base_noext, flags=re.IGNORECASE):
                        return "all_data_gender_tonal_female"

                    # Only suffix *_tonal counts as "tonal speaker"
                    if re.search(r"_tonal$", base_noext, flags=re.IGNORECASE):
                        return "all_data_gender_tonal_female"

                    # Real male/female groups
                    if re.match(r"^0[2-9]_Male_", base_noext):
                        return "all_data_gender_male"
                    if re.match(r"^0[2-9]_Female_", base_noext):
                        # Includes *_male_female / *_female_male conversions per user (still NOT tonal)
                        return "all_data_gender_female"

                    # Fallback: unknown within all_data_gender
                    return None

                # Hmong ct speakers: ctNNN (may be surrounded by '_' so avoid \b)
                m = re.search(r"(ct\d{3})", s, flags=re.IGNORECASE)
                if m is not None:
                    return m.group(1).lower()

                # cx speakers: cxNNN
                m = re.search(r"(cx\d{3})", s, flags=re.IGNORECASE)
                if m is not None:
                    return m.group(1).lower()

                # Chinese: FV1 / MV2 etc
                m = re.search(r"([FM]V\d+)", s)
                if m is not None:
                    return m.group(1)

                return None

            df['speaker'] = df['filename'].apply(extract_speaker)
            
            # Remove rows with unknown gender
            df = df[df['gender'] != 'unknown']

            # Build a deterministic speaker id mapping (stable across runs/workers)
            unique_speakers = sorted([s for s in df['speaker'].dropna().unique().tolist() if s])
            self.speaker_to_id = {spk: i for i, spk in enumerate(unique_speakers)}
            df['speaker_id'] = df['speaker'].map(self.speaker_to_id).fillna(-1).astype(int)
            
            logger.info(f"Loaded {len(df)} entries from {self.transcript_file}")
            logger.info(f"Female entries: {len(df[df['gender'] == 'female'])}")
            logger.info(f"Male entries: {len(df[df['gender'] == 'male'])}")
            if len(self.speaker_to_id) > 0:
                logger.info(f"Detected {len(self.speaker_to_id)} unique speakers for adversarial baseline")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading transcript file {self.transcript_file}: {e}")
            return pd.DataFrame()
    
    def _find_audio_file(self, filename: str) -> Optional[Path]:
        """Find audio file with various extensions"""
        possible_extensions = ['.wav', '.mp3', '.flac', '']
        
        for ext in possible_extensions:
            audio_path = self.data_dir / f"{filename}{ext}"
            if audio_path.exists():
                return audio_path
            else:
                curr_dirs = os.listdir(self.data_dir)
                for curr_dir in curr_dirs:
                    audio_path = self.data_dir / curr_dir / f"{filename}{ext}"
                    if audio_path.exists():
                        return audio_path
                    else:
                        continue
        
        return None
    
    def _load_word_pairs(self) -> List[Dict[str, Any]]:
        """Find all word pairs with their file paths"""
        df = self._load_transcript_file()
        
        if df.empty:
            logger.error("Failed to load transcript file!")
            return []
        
        # Get male and female entries
        male_df = df[df['gender'] == 'male']
        female_df = df[df['gender'] == 'female']
        
        male_words = set(male_df['word'].str.strip())
        female_words = set(female_df['word'].str.strip())
        
        # Find common words
        common_words = male_words & female_words
        logger.info(f"Found {len(common_words)} common words between male and female speakers")
        
        if not common_words:
            logger.error("No common words found!")
            return []
        
        # Create word pairs with their corresponding file paths
        word_pairs = []
        
        for word in common_words:
            # Get male and female entries for this word
            male_entries = male_df[male_df['word'].str.strip() == word]
            female_entries = female_df[female_df['word'].str.strip() == word]
            
            if len(male_entries) == 0 or len(female_entries) == 0:
                continue
            
            # Create pairs for ALL combinations of male/female entries (supports multiple speakers)
            for _, male_row in male_entries.iterrows():
                male_filename = male_row['filename']
                male_speaker_id = int(male_row.get('speaker_id', -1))
                male_path = self._find_audio_file(male_filename)
                
                if male_path is None:
                    logger.warning(f"Male audio file not found: {male_filename}")
                    continue
                
                for _, female_row in female_entries.iterrows():
                    female_filename = female_row['filename']
                    female_speaker_id = int(female_row.get('speaker_id', -1))
                    female_path = self._find_audio_file(female_filename)
                    
                    if female_path is None:
                        logger.warning(f"Female audio file not found: {female_filename}")
                        continue
                    
                    word_pairs.append({
                        'word': word,
                        'male_path': str(male_path),
                        'female_path': str(female_path),
                        'male_filename': male_filename,
                        'female_filename': female_filename,
                        # Speaker IDs for GRL speaker-adversarial training (optional)
                        'male_speaker_id': male_speaker_id,
                        'female_speaker_id': female_speaker_id,
                    })
        
        logger.info(f"Found {len(word_pairs)} word pairs with existing audio files")
        return word_pairs
    
    def _build_male_audio_pool(self) -> List[Dict[str, Any]]:
        """Build a pool of all male audio files for negative sampling"""
        male_pool = []
        for pair in self.word_pairs:
            male_pool.append({
                'word': pair['word'],
                'path': pair['male_path'],
                'filename': pair['male_filename']
            })
        return male_pool
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file with optional augmentation"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Check if audio is empty or too short
            if len(audio) == 0:
                logger.error(f"Audio file is empty: {audio_path}")
                return np.array([], dtype=np.float32)
            
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
            
            # Check again after augmentation
            if len(audio) == 0:
                logger.error(f"Audio became empty after augmentation: {audio_path}")
                return np.array([], dtype=np.float32)
            
            # Normalize audio (after augmentation to account for volume changes)
            if self.normalize:
                audio = audio / (np.abs(audio).max() + 1e-8)
            
            return audio.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return np.array([], dtype=np.float32)
    
    def _sample_negatives(self, target_word: str, num_negatives: int, seed: int = None) -> List[Dict[str, Any]]:
        """Sample negative male audio examples (different words)"""
        # Get all male audios except the target word
        negative_pool = [item for item in self.male_audio_pool if item['word'] != target_word]
        
        # Use a deterministic seed for consistent sampling across distributed processes
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        if len(negative_pool) < num_negatives:
            # Sample with replacement if we don't have enough unique negatives
            negatives = rng.choices(negative_pool, k=num_negatives)
            logger.warning(f"Not enough unique negatives for word '{target_word}', sampling with replacement")
        else:
            # Sample without replacement
            negatives = rng.sample(negative_pool, num_negatives)
        
        return negatives
    
    def __len__(self) -> int:
        return len(self.word_pairs)
    
    def num_tokens(self, index: int) -> int:
        """Return the number of tokens (audio frames) for the given index."""
        # Estimate based on max_audio_length or return a fixed value
        if self.max_audio_length is not None:
            return self.max_audio_length
        return 160000  # Default max length
    
    def size(self, index: int) -> int:
        """Return the size of the sample at the given index."""
        # Return the same as num_tokens for audio data
        return self.num_tokens(index)
    
    def ordered_indices(self):
        """Return an ordered list of indices. Defaults to sequential."""
        return np.arange(len(self), dtype=np.int64)
    
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Returns:
            Dict containing:
            - female_audio: torch.Tensor [T]
            - male_positive_audio: torch.Tensor [T] 
            - male_negative_audios: List[torch.Tensor] [num_negatives x T]
            - word: str
            - female_path: str
            - male_positive_path: str
            - male_negative_paths: List[str]
        """
        pair = self.word_pairs[index]
        word = pair['word']
        female_path = pair['female_path']
        male_positive_path = pair['male_path']
        female_speaker_id = int(pair.get('female_speaker_id', -1))
        male_speaker_id = int(pair.get('male_speaker_id', -1))
        
        # Load female audio (anchor)
        female_audio = self._load_audio(female_path)
        
        # Load male positive audio (same word)
        male_positive_audio = self._load_audio(male_positive_path)
        
        # Sample and load negative male audios (different words)
        # Use index as seed to ensure consistent sampling across distributed processes
        # Ensure the seed is always in valid range [0, 2^32-1]
        deterministic_seed = (abs(self.seed) + index) % (2**32 - 1)
        negative_samples = self._sample_negatives(word, self.num_negatives, seed=deterministic_seed)
        male_negative_audios = []
        male_negative_paths = []
        
        for neg_sample in negative_samples:
            neg_audio = self._load_audio(neg_sample['path'])
            if len(neg_audio) > 0:  # Only add if audio loaded successfully
                male_negative_audios.append(torch.from_numpy(neg_audio))
                male_negative_paths.append(neg_sample['path'])
        
        # Ensure we have enough negatives
        while len(male_negative_audios) < self.num_negatives:
            # Pad with duplicates if necessary
            if len(male_negative_audios) > 0:
                male_negative_audios.append(male_negative_audios[-1].clone())
                male_negative_paths.append(male_negative_paths[-1])
            else:
                # Fallback: create zero audio
                zero_audio = torch.zeros_like(torch.from_numpy(female_audio))
                male_negative_audios.append(zero_audio)
                male_negative_paths.append("ZERO_AUDIO")
        
        return {
            'female_audio': torch.from_numpy(female_audio),
            'male_positive_audio': torch.from_numpy(male_positive_audio),
            'male_negative_audios': male_negative_audios[:self.num_negatives],
            'word': word,
            'female_path': female_path,
            'male_positive_path': male_positive_path,
            'male_negative_paths': male_negative_paths[:self.num_negatives],
            'female_speaker_id': female_speaker_id,
            'male_positive_speaker_id': male_speaker_id,
        }


def collate_gender_contrastive_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for gender contrastive dataset.
    
    Handles variable-length audio sequences by padding to the maximum length in the batch.
    """
    if not samples:
        return {}
    
    batch_size = len(samples)
    num_negatives = len(samples[0]['male_negative_audios'])
    
    # Find maximum audio lengths
    female_max_len = max(sample['female_audio'].shape[0] for sample in samples)
    male_pos_max_len = max(sample['male_positive_audio'].shape[0] for sample in samples)
    male_neg_max_len = max(
        max(neg_audio.shape[0] for neg_audio in sample['male_negative_audios'])
        for sample in samples
    )
    
    # Use the global maximum for consistent padding
    max_len = max(female_max_len, male_pos_max_len, male_neg_max_len)
    
    # Initialize tensors
    female_audios = torch.zeros(batch_size, max_len)
    male_positive_audios = torch.zeros(batch_size, max_len)
    male_negative_audios = torch.zeros(batch_size, num_negatives, max_len)
    
    # Padding masks (True for padding positions)
    female_padding_masks = torch.ones(batch_size, max_len, dtype=torch.bool)
    male_positive_padding_masks = torch.ones(batch_size, max_len, dtype=torch.bool)
    male_negative_padding_masks = torch.ones(batch_size, num_negatives, max_len, dtype=torch.bool)
    
    # Fill tensors and create masks
    words = []
    female_paths = []
    male_positive_paths = []
    male_negative_paths = []
    female_speaker_ids = torch.full((batch_size,), -1, dtype=torch.long)
    male_positive_speaker_ids = torch.full((batch_size,), -1, dtype=torch.long)
    
    for i, sample in enumerate(samples):
        # Female audio
        female_len = sample['female_audio'].shape[0]
        female_audios[i, :female_len] = sample['female_audio']
        female_padding_masks[i, :female_len] = False
        
        # Male positive audio
        male_pos_len = sample['male_positive_audio'].shape[0]
        male_positive_audios[i, :male_pos_len] = sample['male_positive_audio']
        male_positive_padding_masks[i, :male_pos_len] = False
        
        # Male negative audios
        for j, neg_audio in enumerate(sample['male_negative_audios']):
            neg_len = neg_audio.shape[0]
            male_negative_audios[i, j, :neg_len] = neg_audio
            male_negative_padding_masks[i, j, :neg_len] = False
        
        # Metadata
        words.append(sample['word'])
        female_paths.append(sample['female_path'])
        male_positive_paths.append(sample['male_positive_path'])
        male_negative_paths.append(sample['male_negative_paths'])
        if 'female_speaker_id' in sample:
            female_speaker_ids[i] = int(sample.get('female_speaker_id', -1))
        if 'male_positive_speaker_id' in sample:
            male_positive_speaker_ids[i] = int(sample.get('male_positive_speaker_id', -1))
    
    return {
        'net_input': {
            'female_audio': female_audios,
            'male_positive_audio': male_positive_audios,
            'male_negative_audio': male_negative_audios,
            'female_padding_mask': female_padding_masks,
            'male_positive_padding_mask': male_positive_padding_masks,
            'male_negative_padding_mask': male_negative_padding_masks,
            # Speaker IDs for optional speaker-adversarial (GRL) baseline
            'female_speaker_id': female_speaker_ids,
            'male_positive_speaker_id': male_positive_speaker_ids,
        },
        'word_labels': words,
        'female_paths': female_paths,
        'male_positive_paths': male_positive_paths,
        'male_negative_paths': male_negative_paths,
        'ntokens': batch_size * max_len,  # Total number of audio frames
        'nsentences': batch_size,
        'id': torch.arange(batch_size),
    } 