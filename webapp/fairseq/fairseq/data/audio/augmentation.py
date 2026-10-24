# Audio Augmentation Pipeline for Contrastive Learning
# Similar to torchvision.transforms for images

import numpy as np
import librosa
import torch
import random
from typing import Optional, List, Union, Callable
import logging

logger = logging.getLogger(__name__)


class AudioAugmentations:
    """
    Audio augmentation pipeline similar to torchvision.transforms.
    
    Can be used during training by applying augmentations to raw waveforms
    before they're fed to the model.
    
    Example usage:
        transform = AudioAugmentations(
            sample_rate=16000,
            training=True,
            aug_prob=0.5
        )
        
        # In dataset __getitem__
        augmented_audio = transform(raw_audio)
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        training: bool = True,
        aug_prob: float = 0.5,
        noise_factor_range: tuple = (0.005, 0.02),
        time_stretch_range: tuple = (0.9, 1.1),
        pitch_shift_range: tuple = (-1.0, 1.0),
        volume_gain_range: tuple = (-3.0, 3.0),
        enable_noise: bool = True,
        enable_time_stretch: bool = True,
        enable_pitch_shift: bool = True,
        enable_volume: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize audio augmentation pipeline.
        
        Args:
            sample_rate: Audio sample rate
            training: Whether in training mode (augmentations only applied during training)
            aug_prob: Probability of applying each individual augmentation
            noise_factor_range: Range for Gaussian noise factor
            time_stretch_range: Range for time stretching (0.9 = 10% slower, 1.1 = 10% faster)
            pitch_shift_range: Range for pitch shifting in semitones
            volume_gain_range: Range for volume gain in dB
            enable_*: Enable/disable specific augmentations
            seed: Random seed for reproducibility (if None, uses current random state)
        """
        self.sr = sample_rate
        self.training = training
        self.aug_prob = aug_prob
        self.seed = seed
        
        # Augmentation ranges
        self.noise_factor_range = noise_factor_range
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.volume_gain_range = volume_gain_range
        
        # Enable/disable flags
        self.enable_noise = enable_noise
        self.enable_time_stretch = enable_time_stretch
        self.enable_pitch_shift = enable_pitch_shift
        self.enable_volume = enable_volume
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to audio waveform.
        
        Args:
            audio: Input audio waveform (numpy array)
            
        Returns:
            Augmented audio waveform
        """
        if not self.training:
            return audio
            
        return self.apply_augmentations(audio)
    
    def apply_augmentations(self, audio: np.ndarray) -> np.ndarray:
        """Apply random combination of augmentations"""
        try:
            # Apply each augmentation with probability aug_prob
            if self.enable_noise and np.random.random() < self.aug_prob:
                audio = self.add_noise(audio)
            
            if self.enable_time_stretch and np.random.random() < self.aug_prob:
                audio = self.time_stretch(audio)
                
            if self.enable_pitch_shift and np.random.random() < self.aug_prob:
                audio = self.pitch_shift(audio)
                
            if self.enable_volume and np.random.random() < self.aug_prob:
                audio = self.volume_perturbation(audio)
                
            return audio
            
        except Exception as e:
            logger.warning(f"Audio augmentation failed: {e}. Returning original audio.")
            return audio
    
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to audio"""
        noise_factor = np.random.uniform(*self.noise_factor_range)
        noise = np.random.randn(len(audio)) * noise_factor
        return audio + noise
    
    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Apply time stretching (speed change without pitch change)"""
        stretch_factor = np.random.uniform(*self.time_stretch_range)
        try:
            return librosa.effects.time_stretch(audio, rate=stretch_factor)
        except Exception as e:
            logger.warning(f"Time stretch failed: {e}")
            return audio
    
    def pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shifting"""
        n_steps = np.random.uniform(*self.pitch_shift_range)
        try:
            return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio
    
    def volume_perturbation(self, audio: np.ndarray) -> np.ndarray:
        """Apply volume changes"""
        gain_db = np.random.uniform(*self.volume_gain_range)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def train(self):
        """Set to training mode (enables augmentations)"""
        self.training = True
        return self
    
    def eval(self):
        """Set to evaluation mode (disables augmentations)"""
        self.training = False
        return self


class Compose:
    """
    Compose multiple audio transformations together.
    Similar to torchvision.transforms.Compose
    """
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            audio = transform(audio)
        return audio


class RandomChoice:
    """
    Apply one of the given transformations randomly.
    Similar to torchvision.transforms.RandomChoice
    """
    
    def __init__(self, transforms: List[Callable], p: Optional[List[float]] = None):
        self.transforms = transforms
        self.p = p
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        transform = np.random.choice(self.transforms, p=self.p)
        return transform(audio)


class RandomApply:
    """
    Apply given transformation with probability p.
    Similar to torchvision.transforms.RandomApply
    """
    
    def __init__(self, transforms: List[Callable], p: float = 0.5):
        self.transforms = transforms
        self.p = p
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            for transform in self.transforms:
                audio = transform(audio)
        return audio


# Individual augmentation functions (for fine-grained control)
class AddNoise:
    """Add Gaussian noise"""
    
    def __init__(self, noise_factor_range: tuple = (0.005, 0.02)):
        self.noise_factor_range = noise_factor_range
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        noise_factor = np.random.uniform(*self.noise_factor_range)
        noise = np.random.randn(len(audio)) * noise_factor
        return audio + noise


class TimeStretch:
    """Time stretching"""
    
    def __init__(self, stretch_range: tuple = (0.9, 1.1)):
        self.stretch_range = stretch_range
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        stretch_factor = np.random.uniform(*self.stretch_range)
        try:
            return librosa.effects.time_stretch(audio, rate=stretch_factor)
        except Exception as e:
            logger.warning(f"Time stretch failed: {e}")
            return audio


class PitchShift:
    """Pitch shifting"""
    
    def __init__(self, pitch_range: tuple = (-1.0, 1.0), sample_rate: int = 16000):
        self.pitch_range = pitch_range
        self.sr = sample_rate
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        n_steps = np.random.uniform(*self.pitch_range)
        try:
            return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio


class VolumeChange:
    """Volume perturbation"""
    
    def __init__(self, gain_range: tuple = (-3.0, 3.0)):
        self.gain_range = gain_range
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(*self.gain_range)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear


# Example usage patterns (similar to torchvision)
def get_training_transforms(sample_rate: int = 16000) -> AudioAugmentations:
    """Get standard training augmentations"""
    return AudioAugmentations(
        sample_rate=sample_rate,
        training=True,
        aug_prob=0.5,
        noise_factor_range=(0.005, 0.02),
        time_stretch_range=(0.95, 1.05),  # Conservative for speech
        pitch_shift_range=(-0.5, 0.5),   # Conservative for tonal languages
        volume_gain_range=(-2.0, 2.0),   # Conservative volume changes
    )


def get_aggressive_transforms(sample_rate: int = 16000) -> AudioAugmentations:
    """Get more aggressive augmentations for robust training"""
    return AudioAugmentations(
        sample_rate=sample_rate,
        training=True,
        aug_prob=0.7,
        noise_factor_range=(0.01, 0.03),
        time_stretch_range=(0.9, 1.1),
        pitch_shift_range=(-1.0, 1.0),
        volume_gain_range=(-3.0, 3.0),
    )


def get_conservative_transforms(sample_rate: int = 16000) -> AudioAugmentations:
    """Get conservative augmentations (good for tonal languages)"""
    return AudioAugmentations(
        sample_rate=sample_rate,
        training=True,
        aug_prob=0.3,
        noise_factor_range=(0.002, 0.01),
        time_stretch_range=(0.98, 1.02),
        pitch_shift_range=(-0.2, 0.2),   # Very conservative for tones
        volume_gain_range=(-1.0, 1.0),
        enable_pitch_shift=False,  # Disable pitch shift for tonal languages
    )


def get_validation_transforms() -> AudioAugmentations:
    """Get transforms for validation (no augmentation)"""
    return AudioAugmentations(training=False)


# Custom compose example for fine-grained control
def get_custom_compose_transforms(sample_rate: int = 16000):
    """Example of using individual transforms with Compose"""
    return Compose([
        RandomApply([AddNoise(noise_factor_range=(0.005, 0.015))], p=0.5),
        RandomApply([VolumeChange(gain_range=(-2.0, 2.0))], p=0.4),
        RandomChoice([
            TimeStretch(stretch_range=(0.95, 1.05)),
            PitchShift(pitch_range=(-0.5, 0.5), sample_rate=sample_rate),
        ], p=[0.7, 0.3]),  # Prefer time stretch over pitch shift
    ])


def test_real_audio_augmentations(audio_path: str, output_dir: str = "augmentation_test_outputs"):
    """
    Test individual augmentations on real audio file and save outputs.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save augmented audio files
    """
    import os
    import soundfile as sf
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🎵 Testing augmentations on: {audio_path}")
    print(f"📁 Output directory: {output_path}")
    
    # Load audio
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"✅ Loaded audio: {audio.shape} samples at {sr}Hz ({len(audio)/sr:.2f}s)")
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return
    
    # Save original audio
    original_path = output_path / "00_original.wav"
    sf.write(original_path, audio, sr)
    print(f"💾 Saved original: {original_path}")
    
    # Test individual augmentations
    augmentations = [
        ("01_noise_light", AddNoise(noise_factor_range=(0.005, 0.005))),  # Fixed value for consistency
        ("02_noise_moderate", AddNoise(noise_factor_range=(0.015, 0.015))),
        ("03_time_stretch_slow", TimeStretch(stretch_range=(0.9, 0.9))),  # 10% slower
        ("04_time_stretch_fast", TimeStretch(stretch_range=(1.1, 1.1))),  # 10% faster
        ("05_pitch_down", PitchShift(pitch_range=(-1.0, -1.0), sample_rate=sr)),  # 1 semitone down
        ("06_pitch_up", PitchShift(pitch_range=(1.0, 1.0), sample_rate=sr)),      # 1 semitone up
        ("07_volume_quiet", VolumeChange(gain_range=(-3.0, -3.0))),       # -3dB
        ("08_volume_loud", VolumeChange(gain_range=(3.0, 3.0))),          # +3dB
    ]
    
    # Apply each augmentation
    for name, aug_func in augmentations:
        try:
            print(f"🔄 Applying {name}...")
            augmented = aug_func(audio.copy())  # Copy to avoid modifying original
            
            # Save augmented audio
            output_file = output_path / f"{name}.wav"
            sf.write(output_file, augmented, sr)
            
            # Print stats
            orig_rms = np.sqrt(np.mean(audio**2))
            aug_rms = np.sqrt(np.mean(augmented**2))
            length_change = len(augmented) - len(audio)
            
            print(f"   ✅ {name}: {audio.shape} -> {augmented.shape} ({length_change:+d} samples)")
            print(f"      RMS: {orig_rms:.4f} -> {aug_rms:.4f} ({aug_rms/orig_rms:.3f}x)")
            print(f"      💾 Saved: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Failed {name}: {e}")
    
    # Test combined augmentations (presets)
    print(f"\n🎛️  Testing combined augmentation presets...")
    
    presets = [
        ("09_training_preset", get_training_transforms(sr)),
        ("10_aggressive_preset", get_aggressive_transforms(sr)),
        ("11_conservative_preset", get_conservative_transforms(sr)),
        ("12_custom_compose", get_custom_compose_transforms(sr)),
    ]
    
    for name, transform in presets:
        try:
            print(f"🔄 Applying {name}...")
            # Apply multiple times to show randomness
            for i in range(3):
                augmented = transform(audio.copy())
                output_file = output_path / f"{name}_sample_{i+1}.wav"
                sf.write(output_file, augmented, sr)
                
                orig_rms = np.sqrt(np.mean(audio**2))
                aug_rms = np.sqrt(np.mean(augmented**2))
                length_change = len(augmented) - len(audio)
                
                print(f"   ✅ Sample {i+1}: {audio.shape} -> {augmented.shape} ({length_change:+d} samples)")
                print(f"      RMS: {orig_rms:.4f} -> {aug_rms:.4f} ({aug_rms/orig_rms:.3f}x)")
                print(f"      💾 Saved: {output_file}")
                
        except Exception as e:
            print(f"   ❌ Failed {name}: {e}")
    
    print(f"\n🎉 Augmentation testing complete!")
    print(f"📁 All outputs saved in: {output_path}")
    print(f"🎧 You can now listen to the augmented audio files to evaluate quality.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test audio augmentations')
    parser.add_argument('--audio-path', type=str, default=None,
                       help='Path to audio file to test (if not provided, uses synthetic audio)')
    parser.add_argument('--output-dir', type=str, default='augmentation_test_outputs',
                       help='Output directory for augmented audio files')
    
    args = parser.parse_args()
    
    if args.audio_path:
        # Test with real audio
        test_real_audio_augmentations(args.audio_path, args.output_dir)
    else:
        # Original synthetic audio test
        print("Testing audio augmentation pipeline with synthetic audio...")
        print("💡 Use --audio-path to test with real audio files")
        print()
        
        # Create dummy audio
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Test different transform configurations
        transforms = {
            "training": get_training_transforms(sample_rate),
            "aggressive": get_aggressive_transforms(sample_rate),
            "conservative": get_conservative_transforms(sample_rate),
            "validation": get_validation_transforms(),
            "custom": get_custom_compose_transforms(sample_rate),
        }
        
        for name, transform in transforms.items():
            augmented = transform(audio)
            print(f"{name}: {audio.shape} -> {augmented.shape}")
            print(f"  Original RMS: {np.sqrt(np.mean(audio**2)):.4f}")
            print(f"  Augmented RMS: {np.sqrt(np.mean(augmented**2)):.4f}")
            print()