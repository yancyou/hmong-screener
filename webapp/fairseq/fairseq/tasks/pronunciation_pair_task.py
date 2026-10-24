# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pronunciation pair task
=======================

Goal: learn a 3-way classifier from a (reference_audio, speaker_audio) pair.

Labels (int — kept as 1/2/3 in data, remapped to 0/1/2 in the criterion):
  1: exactly correct
  2: correct base word, wrong pronunciation
  3: wrong base word, wrong pronunciation

Label 0 (missing) rows are silently skipped during manifest loading.

Dataset manifest formats supported:
  - TSV/CSV with header. Must contain at least:
      speaker_audio, reference_audio, label
    OR:
      wav_file, word, label   (reference_audio resolved from tone_metadata_file)
  - JSON / JSONL (list of dicts)
"""

import csv
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from fairseq.data import Dictionary, FairseqDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from omegaconf import MISSING

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def _pick_col(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Return the first key from *candidates* that exists in *row*."""
    for c in candidates:
        if c in row:
            return c
    return None


def _is_probably_tsv(path: str) -> bool:
    return path.lower().endswith(".tsv")


def _is_probably_json(path: str) -> bool:
    return path.lower().endswith(".json")


def _is_probably_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl")


_KNOWN_TONE_SUFFIXES = {"b", "m", "j", "v", "s", "g"}
_KNOWN_GENDER_SUFFIXES = {"male", "female", "m", "f"}


def _normalize_audio_stem(filename: str) -> str:
    """
    Strip extension and known tone/gender suffixes from an audio filename
    to get a canonical stem for matching.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.rsplit("_", 2)
    while len(parts) > 1 and parts[-1].lower() in (_KNOWN_TONE_SUFFIXES | _KNOWN_GENDER_SUFFIXES):
        parts = parts[:-1]
    return "_".join(parts)


def _try_resolve_audio_path(data_dir: str, p: Optional[str]) -> Optional[str]:
    """
    Resolve a possibly-relative audio path, with extension probing.
    Returns absolute path if exists, else None.

    Search order for a relative path like ``spk_list1/word001.wav``:
      1. ``data_dir/spk_list1/word001.wav``          (direct join)
      2. ``data_dir/<sub>/word001.wav``                (flat one-level: filename only)
      3. ``data_dir/<sub>/spk_list1/word001.wav``     (one-level + full relative path)
    """
    if p is None:
        return None
    p = str(p).strip()
    if not p:
        return None

    # Keep original relative path before joining
    p = os.path.expandvars(os.path.expanduser(p))
    rel_path = p  # save for step-3 search
    if not os.path.isabs(p):
        p = os.path.join(data_dir, p)

    def _subdir_search(filename: str, rel: Optional[str] = None) -> Optional[str]:
        """Search data_dir/<sub>/filename, and optionally data_dir/<sub>/rel."""
        try:
            for sub in os.listdir(data_dir):
                sub_dir = os.path.join(data_dir, sub)
                if not os.path.isdir(sub_dir):
                    continue
                # flat: data_dir/<sub>/filename
                cand = os.path.join(sub_dir, filename)
                if os.path.exists(cand):
                    return cand
                # nested: data_dir/<sub>/<full_relative_path>
                if rel is not None and rel != filename:
                    cand2 = os.path.join(sub_dir, rel)
                    if os.path.exists(cand2):
                        return cand2
        except Exception:
            pass
        return None

    # If already has extension, check directly
    root, ext = os.path.splitext(p)
    if ext:
        if os.path.exists(p):
            return p
        found = _subdir_search(os.path.basename(p), rel_path)
        if found:
            return found
        return None

    # Probe common audio extensions
    for e in (".wav", ".flac", ".mp3"):
        cand = root + e
        if os.path.exists(cand):
            return cand
        found = _subdir_search(os.path.basename(root) + e, rel_path + e if rel_path else None)
        if found:
            return found

    # As-is fallback (maybe directory / odd suffix)
    return p if os.path.exists(p) else None


# ============================================================
# Task config + registration
# ============================================================

@dataclass
class PronunciationPairTaskConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={"help": "root directory for audio paths (also used for relative paths)"},
    )

    # Manifest paths
    manifest_train: str = field(
        default="manifests/pron/train.tsv",
        metadata={"help": "path to training manifest (tsv/csv/json/jsonl, relative to data)"},
    )
    manifest_val: str = field(
        default="manifests/pron/val.tsv",
        metadata={"help": "path to validation manifest"},
    )

    # Audio processing
    sample_rate: int = field(default=16000, metadata={"help": "expected sample rate"})
    max_audio_length: int = field(default=160000, metadata={"help": "max waveform length (samples)"})
    min_audio_length: int = field(default=1600, metadata={"help": "min waveform length (samples)"})
    normalize: bool = field(default=True, metadata={"help": "normalize waveform to max abs = 1"})
    seed: int = field(default=42, metadata={"help": "random seed"})

    # Augmentation
    enable_augmentation: bool = field(default=False, metadata={"help": "enable waveform augmentation"})
    augmentation_config: Optional[Any] = field(
        default=None,
        metadata={"help": "dict of augmentation params (noise, time_stretch, volume, etc.)"},
    )
    augment_speaker: bool = field(default=True, metadata={"help": "augment speaker audio"})
    augment_reference: bool = field(default=False, metadata={"help": "augment reference audio"})

    # Audio path strictness
    strict_audio_paths: bool = field(
        default=True,
        metadata={
            "help": "if true, skip samples when audio path can't be resolved (with warning)"
        },
    )

    # Tone metadata for reference bank + synthetic data
    tone_metadata_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to hmong_tone_metadata_train.json (used for reference bank and synth)"},
    )

    # Training-time synthetic label construction
    synth_enable: bool = field(
        default=False,
        metadata={"help": "enable synthetic label2/label3 generation from tone metadata"},
    )
    synth_source_labels: Optional[Any] = field(
        default=None,
        metadata={"help": "which true labels to use as sources for synthetic generation (default: [1])"},
    )
    synth_num_label2_per_source: int = field(
        default=1,
        metadata={"help": "number of synthetic label-2 (same base, different tone) per source sample"},
    )
    synth_num_label3_per_source: int = field(
        default=1,
        metadata={"help": "number of synthetic label-3 (different base word) per source sample"},
    )


@register_task("pronunciation_pair", dataclass=PronunciationPairTaskConfig)
class PronunciationPairTask(FairseqTask):
    cfg: PronunciationPairTaskConfig

    def __init__(self, cfg: PronunciationPairTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = Dictionary()
        logger.info("PronunciationPairTask initialized")
        logger.info(f"Audio root: {cfg.data}")

    @classmethod
    def setup_task(cls, cfg: PronunciationPairTaskConfig, **kwargs):
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs):
        cfg = self.cfg
        if split == "train":
            manifest = cfg.manifest_train
            enable_augmentation = cfg.enable_augmentation
        else:
            manifest = cfg.manifest_val
            enable_augmentation = False

        # Resolve manifest path
        if not os.path.isabs(manifest):
            manifest = os.path.join(cfg.data, manifest)

        # Resolve tone metadata path
        tone_meta = cfg.tone_metadata_file
        if tone_meta and not os.path.isabs(tone_meta):
            tone_meta = os.path.join(cfg.data, tone_meta)

        ds = PronunciationPairDataset(
            data_dir=cfg.data,
            manifest_path=manifest,
            sample_rate=cfg.sample_rate,
            max_audio_length=cfg.max_audio_length,
            min_audio_length=cfg.min_audio_length,
            normalize=cfg.normalize,
            seed=cfg.seed,
            enable_augmentation=enable_augmentation,
            augmentation_config=cfg.augmentation_config,
            augment_speaker=cfg.augment_speaker,
            augment_reference=cfg.augment_reference,
            strict_audio_paths=cfg.strict_audio_paths,
            tone_metadata_file=tone_meta,
            synth_enable=cfg.synth_enable if split == "train" else False,
            synth_source_labels=cfg.synth_source_labels,
            synth_num_label2_per_source=cfg.synth_num_label2_per_source,
            synth_num_label3_per_source=cfg.synth_num_label3_per_source,
        )

        self.datasets[split] = ds
        logger.info(
            f"Loaded {len(ds)} samples for {split} (augmentation: {enable_augmentation})"
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class PronunciationPairDataset(FairseqDataset):
    """
    Each item yields:
      - speaker_audio: Tensor [T]
      - reference_audio: Tensor [T]
      - pron_label: int in {1,2,3}
    """

    def __init__(
        self,
        data_dir: str,
        manifest_path: str,
        sample_rate: int = 16000,
        max_audio_length: int = 320000,
        min_audio_length: int = 1600,
        normalize: bool = True,
        seed: int = 42,
        enable_augmentation: bool = False,
        augmentation_config: Optional[Any] = None,
        augment_speaker: bool = True,
        augment_reference: bool = False,
        strict_audio_paths: bool = True,
        tone_metadata_file: Optional[str] = None,
        synth_enable: bool = False,
        synth_source_labels: Optional[Any] = (1,),
        synth_num_label2_per_source: int = 1,
        synth_num_label3_per_source: int = 1,
    ):
        # Make data_dir absolute to avoid Hydra cwd surprises
        self.data_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(data_dir)))
        self.manifest_path = manifest_path
        self.sample_rate = int(sample_rate)
        self.max_audio_length = int(max_audio_length)
        self.min_audio_length = int(min_audio_length)
        self.normalize = bool(normalize)
        self.seed = int(seed)

        self.enable_augmentation = bool(enable_augmentation)
        self.augment_speaker = bool(augment_speaker)
        self.augment_reference = bool(augment_reference)
        self.rng = random.Random(int(seed))
        self.strict_audio_paths = bool(strict_audio_paths)

        self.synth_enable = bool(synth_enable)
        self.tone_metadata_file = tone_metadata_file
        self.synth_num_label2_per_source = max(0, int(synth_num_label2_per_source))
        self.synth_num_label3_per_source = max(0, int(synth_num_label3_per_source))
        self.synth_source_labels = self._coerce_int_set(synth_source_labels, default={1})

        # Build audio transform if augmentation enabled
        self.audio_transform = None
        if self.enable_augmentation and augmentation_config is not None:
            self.audio_transform = self._build_audio_transform(augmentation_config)
            logger.info("PronunciationPairDataset: augmentation enabled")

        # Load manifest
        self.samples = self._load_manifest(manifest_path)
        logger.info(f"Loaded {len(self.samples)} rows from {manifest_path}")

        # ===== Reference bank (tone metadata) =====
        # We use it for two things:
        #  1) resolving reference_audio from prompt "word" (label CSVs often only have wav_file+word+label)
        #  2) constructing synthetic label2/label3 samples (training only)
        self._ref_by_base: Dict[str, Dict[str, str]] = {}
        self._ref_by_word: Dict[str, Tuple[str, str, str]] = {}  # word -> (base_word, tone_key, ref_path)
        self._ref_key_to_base_tone: Dict[str, Tuple[str, str]] = {}  # audio-stem key -> (base_word, tone_key)
        self._tone_marker_to_tone_key: Dict[str, str] = {}  # marker -> tone_key
        self._tone_markers_sorted: List[str] = []  # non-empty markers sorted by length desc
        self._all_bases: List[str] = []

        # Do we need the tone reference bank?
        need_ref_from_word = any(
            (s.get("reference_audio") in (None, "")) and (s.get("word") not in (None, ""))
            for s in self.samples
        )
        need_bank = bool(self.synth_enable or need_ref_from_word)
        if need_bank:
            self._build_reference_bank_from_tone_metadata(require=True)

        # Resolve audio paths + validate true labels
        self._resolved: List[Tuple[Optional[str], Optional[str], int]] = []
        self._prompt_word: List[Optional[str]] = []
        self._prompt_base: List[Optional[str]] = []
        self._prompt_tone: List[Optional[str]] = []

        _valid_labels = {1, 2, 3}
        skipped_label0 = 0
        skipped_no_ref = 0

        true_counts = {1: 0, 2: 0, 3: 0}
        for s in self.samples:
            y = int(s.get("label"))

            # Skip label 0 (missing) — not used in classifier training
            if y == 0:
                skipped_label0 += 1
                continue

            if y not in _valid_labels:
                raise ValueError(f"Invalid label {y} (expected 1..3) in {manifest_path}")

            sp_raw = s.get("speaker_audio")
            rp_raw = s.get("reference_audio")
            prompt_word = s.get("word", None)
            if prompt_word is not None:
                prompt_word = str(prompt_word).strip()
                if prompt_word == "":
                    prompt_word = None

            sp = _try_resolve_audio_path(self.data_dir, sp_raw)

            # Resolve reference audio:
            # - If manifest provides reference_audio, use it
            # - Else, resolve from prompt word via tone metadata
            if rp_raw not in (None, ""):
                rp = _try_resolve_audio_path(self.data_dir, rp_raw)
            else:
                rp = self._resolve_reference_audio_from_word(prompt_word)

            # Infer prompt base/tone (used for synthetic label2/3 swapping)
            bw, tk = self._infer_prompt_base_tone(prompt_word=prompt_word, reference_path=rp)
            self._prompt_word.append(prompt_word)
            self._prompt_base.append(bw)
            self._prompt_tone.append(tk)

            # Skip samples with unresolvable audio (word not in tone metadata, etc.)
            if rp is None:
                msg = (
                    f"Unresolved reference_audio for word={prompt_word} "
                    f"(reference_audio={rp_raw}, data_dir={self.data_dir})"
                )
                if self.strict_audio_paths:
                    logger.warning(msg + " -> skipping sample")
                    skipped_no_ref += 1
                    continue
                else:
                    logger.warning(msg + " -> will use silence tensor")
            if sp is None:
                msg = f"Unresolved speaker_audio for label={y}: {sp_raw} (data_dir={self.data_dir})"
                if self.strict_audio_paths:
                    logger.warning(msg + " -> skipping sample")
                    skipped_no_ref += 1
                    continue
                else:
                    logger.warning(msg + " -> will use silence tensor")

            true_counts[y] += 1
            self._resolved.append((sp, rp, y))

        if skipped_label0 > 0:
            logger.info(f"Skipped {skipped_label0} label-0 (missing) samples")
        if skipped_no_ref > 0:
            logger.info(
                f"Skipped {skipped_no_ref} samples with unresolvable audio paths "
                f"(word not in tone metadata or file missing)"
            )

        self.true_label_counts = dict(true_counts)
        self.label_counts = dict(true_counts)  # will be updated if synth is enabled
        logger.info(f"PronunciationPairDataset true label counts: {self.true_label_counts}")

        # Virtual indices over (true + synthetic) samples.
        # mode: 0=true; 2=synthetic label2 (swap tone); 3=synthetic label3 (swap base word)
        self._virtual_index: List[Tuple[int, int]] = [(i, 0) for i in range(len(self._resolved))]
        if self.synth_enable:
            self._expand_virtual_indices_with_synth()

    @staticmethod
    def _coerce_int_set(val: Optional[Any], default: set) -> set:
        if val is None:
            return set(default)
        if isinstance(val, (list, tuple, set)):
            out = set()
            for x in val:
                try:
                    out.add(int(x))
                except Exception:
                    pass
            return out if out else set(default)
        if isinstance(val, str):
            val = val.strip().strip("[](){}")
            out = set()
            for part in val.split(","):
                part = part.strip()
                if part:
                    try:
                        out.add(int(part))
                    except Exception:
                        pass
            return out if out else set(default)
        try:
            return {int(val)}
        except Exception:
            return set(default)

    @staticmethod
    def _build_audio_transform(config):
        """Build a simple numpy-based audio transform from config dict."""
        if config is None:
            return None

        aug_prob = float(config.get("aug_prob", 0.3))
        noise_range = config.get("noise_factor_range", [0.002, 0.01])
        stretch_range = config.get("time_stretch_range", [0.98, 1.02])
        volume_range = config.get("volume_gain_range", [-1.5, 1.5])
        enable_noise = config.get("enable_noise", True)
        enable_stretch = config.get("enable_time_stretch", True)
        enable_volume = config.get("enable_volume", True)

        def transform(wav_np):
            if random.random() > aug_prob:
                return wav_np
            if enable_noise and random.random() < 0.5:
                noise = np.random.randn(*wav_np.shape).astype(wav_np.dtype)
                factor = random.uniform(float(noise_range[0]), float(noise_range[1]))
                wav_np = wav_np + factor * noise
            if enable_stretch and random.random() < 0.5:
                rate = random.uniform(float(stretch_range[0]), float(stretch_range[1]))
                indices = np.round(np.arange(0, len(wav_np), rate)).astype(int)
                indices = indices[indices < len(wav_np)]
                wav_np = wav_np[indices]
            if enable_volume and random.random() < 0.5:
                gain_db = random.uniform(float(volume_range[0]), float(volume_range[1]))
                wav_np = wav_np * (10.0 ** (gain_db / 20.0))
            return wav_np

        return transform

    # ===== Reference bank from tone metadata =====

    def _build_reference_bank_from_tone_metadata(self, require: bool = False):
        """
        Build lookup tables from the tone metadata JSON:
          - _ref_by_base[base_word][tone_key] -> resolved reference audio path
          - _ref_by_word[word] -> (base_word, tone_key, ref_path)
          - _tone_marker_to_tone_key[marker] -> tone_key
        """
        meta_path = self.tone_metadata_file
        if meta_path is None:
            if require:
                raise FileNotFoundError(
                    "tone_metadata_file is required for reference bank but not set"
                )
            return

        if not os.path.isabs(meta_path):
            meta_path = os.path.join(self.data_dir, meta_path)

        if not os.path.exists(meta_path):
            if require:
                raise FileNotFoundError(f"tone_metadata_file not found: {meta_path}")
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Build tone_marker -> tone_key mapping from the tone_system
        tone_system = meta.get("tone_system", {}).get("tone_markers", {})
        for tk, marker in tone_system.items():
            self._tone_marker_to_tone_key[marker] = tk
        self._tone_markers_sorted = sorted(
            [m for m in self._tone_marker_to_tone_key.keys() if m], key=len, reverse=True
        )

        tone_groups = meta.get("tone_groups", [])
        missing_ref = 0
        for g in tone_groups:
            base_word = g.get("base_word", None)
            if base_word is None:
                continue
            tone_variants = g.get("tone_variants", {})
            for tone_key, variant in tone_variants.items():
                word = variant.get("word", None)
                fn = variant.get("filename", None)
                if word is None or fn is None:
                    continue
                rp = _try_resolve_audio_path(self.data_dir, fn)
                if rp is None:
                    missing_ref += 1
                    continue

                # _ref_by_base
                if base_word not in self._ref_by_base:
                    self._ref_by_base[base_word] = {}
                self._ref_by_base[base_word][tone_key] = rp

                # _ref_by_word
                self._ref_by_word[word] = (base_word, tone_key, rp)

                # _ref_key_to_base_tone
                stem = _normalize_audio_stem(fn)
                self._ref_key_to_base_tone[stem] = (base_word, tone_key)

        self._all_bases = sorted(self._ref_by_base.keys())

        if not self._all_bases:
            msg = f"Reference bank is empty after loading {meta_path}"
            if require:
                raise RuntimeError(msg)
            logger.warning(msg)
            return

        if missing_ref:
            logger.info(f"Reference bank: {missing_ref} tone variants had missing audio paths (skipped).")

        logger.info(
            f"Reference bank ready: bases={len(self._all_bases)}, "
            f"word_entries={len(self._ref_by_word)} "
            f"(tone_meta={meta_path})"
        )

    def _resolve_reference_audio_from_word(self, prompt_word: Optional[str]) -> Optional[str]:
        """Look up the reference audio for a prompt word using the reference bank."""
        if prompt_word is None or not self._ref_by_word:
            return None
        entry = self._ref_by_word.get(prompt_word, None)
        if entry is not None:
            return entry[2]  # ref_path
        return None

    def _infer_prompt_base_tone(
        self, prompt_word: Optional[str] = None, reference_path: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Infer (base_word, tone_key) for a prompt from word or reference path."""
        # Try from word lookup first
        if prompt_word is not None and prompt_word in self._ref_by_word:
            bw, tk, _ = self._ref_by_word[prompt_word]
            return bw, tk

        # Try stripping tone marker from prompt word
        if prompt_word is not None and self._tone_markers_sorted:
            for marker in self._tone_markers_sorted:
                if prompt_word.endswith(marker):
                    base = prompt_word[: -len(marker)]
                    tk = self._tone_marker_to_tone_key.get(marker)
                    if base and tk and base in self._ref_by_base:
                        return base, tk
            # Tone 2 has empty marker
            empty_tk = self._tone_marker_to_tone_key.get("", None)
            if empty_tk and prompt_word in self._ref_by_base:
                return prompt_word, empty_tk

        # Try from reference path stem
        if reference_path is not None:
            stem = _normalize_audio_stem(reference_path)
            if stem in self._ref_key_to_base_tone:
                return self._ref_key_to_base_tone[stem]

        return None, None

    def _expand_virtual_indices_with_synth(self):
        """
        Expand _virtual_index with synthetic label2/label3 samples.
        Assumes prompt base/tone are already inferred for each true sample.
        """
        # Ensure reference bank exists (required for swapping)
        if not self._ref_by_base or not self._all_bases:
            try:
                self._build_reference_bank_from_tone_metadata(require=True)
            except Exception as e:
                logger.warning(f"Synth disabled: failed to build reference bank: {e}")
                self.synth_enable = False
                return

        if self.synth_num_label2_per_source <= 0 and self.synth_num_label3_per_source <= 0:
            logger.info("Synth enabled but synth_num_label2/3_per_source are both 0; nothing to add.")
            return

        syn2 = 0
        syn3 = 0
        eligible = 0

        for base_idx, (sp, rp, y) in enumerate(self._resolved):
            if int(y) not in self.synth_source_labels:
                continue
            if sp is None or rp is None:
                continue
            bw = self._prompt_base[base_idx]
            tk = self._prompt_tone[base_idx]
            if bw is None or tk is None:
                continue

            # label2 requires an alternate tone for the same baseword
            tones = list(self._ref_by_base.get(bw, {}).keys())
            alt_tones = [t for t in tones if t != tk]
            has_l2 = len(alt_tones) > 0

            # label3 requires another baseword in the bank
            has_l3 = len(self._all_bases) > 1

            if not has_l2 and not has_l3:
                continue

            eligible += 1
            if has_l2 and self.synth_num_label2_per_source > 0:
                for _ in range(self.synth_num_label2_per_source):
                    self._virtual_index.append((base_idx, 2))
                    syn2 += 1
            if has_l3 and self.synth_num_label3_per_source > 0:
                for _ in range(self.synth_num_label3_per_source):
                    self._virtual_index.append((base_idx, 3))
                    syn3 += 1

        # Update counts to reflect synthetic additions
        self.label_counts = dict(self.true_label_counts)
        self.label_counts[2] = int(self.label_counts.get(2, 0)) + int(syn2)
        self.label_counts[3] = int(self.label_counts.get(3, 0)) + int(syn3)

        logger.info(
            "Synth summary: "
            f"eligible_sources={eligible}, added_syn2={syn2}, added_syn3={syn3}, "
            f"final_len={len(self._virtual_index)}"
        )
        logger.info(f"PronunciationPairDataset label counts (with synth): {self.label_counts}")

    def _load_manifest(self, path: str) -> List[Dict[str, Any]]:
        def _normalize_sample(obj: Dict[str, Any]) -> Dict[str, Any]:
            # Normalize keys and handle common aliases
            speaker_key = _pick_col(
                obj,
                [
                    "speaker_audio",
                    "speaker_path",
                    "speaker",
                    "v",
                    "audio_speaker",
                    "speaker_wav",
                    # Common label-file naming
                    "wav_file",
                    "wav_path",
                    "wav",
                ],
            )
            ref_key = _pick_col(
                obj,
                [
                    "reference_audio",
                    "reference_path",
                    "reference",
                    "u",
                    "audio_reference",
                    "ref_wav",
                ],
            )
            label_key = _pick_col(obj, ["label", "y", "target"])

            if label_key is None:
                raise ValueError(f"Missing label column in manifest row: {obj}")

            out = {
                "speaker_audio": obj.get(speaker_key) if speaker_key else None,
                "reference_audio": obj.get(ref_key) if ref_key else None,
                "label": obj.get(label_key),
            }
            # keep extra fields for debugging if needed
            for extra in ("base_word", "word", "utt_id", "id"):
                if extra in obj:
                    out[extra] = obj[extra]
            return out

        if _is_probably_jsonl(path):
            out = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(_normalize_sample(json.loads(line)))
            return out

        if _is_probably_json(path):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                if "samples" in obj and isinstance(obj["samples"], list):
                    obj = obj["samples"]
                elif "data" in obj and isinstance(obj["data"], list):
                    obj = obj["data"]
                else:
                    raise ValueError(f"JSON manifest must be a list or {{samples: [...]}}: {path}")
            if not isinstance(obj, list):
                raise ValueError(f"JSON manifest must be a list: {path}")
            return [_normalize_sample(x) for x in obj]

        # TSV/CSV fallback
        delimiter = "\t" if _is_probably_tsv(path) else ","
        out = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header row: {path}")
            header_set = set(reader.fieldnames)
            for row in reader:
                # Skip duplicate header rows (from concatenated CSVs)
                if set(row.values()) == header_set:
                    continue
                out.append(_normalize_sample(row))
        return out

    def __len__(self) -> int:
        return len(self._virtual_index)

    def num_tokens(self, index):
        # Rough estimate for batching; keep consistent with other audio datasets
        return self.max_audio_length

    def size(self, index):
        return self.max_audio_length

    def ordered_indices(self):
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        return False

    def collater(self, samples):
        return collate_pronunciation_pair_samples(samples)

    def _load_audio(self, path: Optional[str], apply_augmentation: bool) -> torch.Tensor:
        if path is None:
            # Missing audio: return silence (still normalized/padded later)
            return torch.zeros(self.min_audio_length, dtype=torch.float32)

        waveform, sr = torchaudio.load(path)
        # to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # [T]

        # resample
        if int(sr) != int(self.sample_rate):
            waveform = torchaudio.functional.resample(waveform, int(sr), int(self.sample_rate))

        # augmentation
        if apply_augmentation and self.audio_transform is not None:
            wav_np = waveform.detach().cpu().numpy()
            wav_np = self.audio_transform(wav_np)
            waveform = torch.from_numpy(np.asarray(wav_np, dtype=np.float32))

        # normalize
        if self.normalize and waveform.numel() > 0:
            mx = waveform.abs().max()
            if mx > 0:
                waveform = waveform / mx

        # trim/pad
        if self.max_audio_length and waveform.size(0) > self.max_audio_length:
            waveform = waveform[: self.max_audio_length]
        if self.min_audio_length and waveform.size(0) < self.min_audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.min_audio_length - waveform.size(0)))

        return waveform.float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_idx, mode = self._virtual_index[int(index)]
        sp, rp_true, y_true = self._resolved[int(base_idx)]

        # Decide label + reference path
        if int(mode) == 0:
            rp = rp_true
            y = int(y_true)
        elif int(mode) == 2:
            # same baseword, different tone
            bw = self._prompt_base[int(base_idx)]
            tk = self._prompt_tone[int(base_idx)]
            if bw is None or tk is None:
                rp = rp_true
                y = int(y_true)
            else:
                tones = list(self._ref_by_base.get(bw, {}).keys())
                alt_tones = [t for t in tones if t != tk]
                if not alt_tones:
                    rp = rp_true
                    y = int(y_true)
                else:
                    # deterministic choice by index
                    det_seed = (abs(self.seed) + 10007 * int(index) + 17 * int(base_idx)) % (2**32 - 1)
                    rng = random.Random(det_seed)
                    chosen_tone = rng.choice(alt_tones)
                    rp = self._ref_by_base[bw][chosen_tone]
                    y = 2
        elif int(mode) == 3:
            # different baseword
            bw = self._prompt_base[int(base_idx)]
            if bw is None or len(self._all_bases) <= 1:
                rp = rp_true
                y = int(y_true)
            else:
                det_seed = (abs(self.seed) + 10009 * int(index) + 31 * int(base_idx)) % (2**32 - 1)
                rng = random.Random(det_seed)
                other_bases = [b for b in self._all_bases if b != bw]
                if not other_bases:
                    rp = rp_true
                    y = int(y_true)
                else:
                    chosen_bw = rng.choice(other_bases)
                    tone_keys = list(self._ref_by_base.get(chosen_bw, {}).keys())
                    if not tone_keys:
                        rp = rp_true
                        y = int(y_true)
                    else:
                        chosen_tk = rng.choice(tone_keys)
                        rp = self._ref_by_base[chosen_bw][chosen_tk]
                        y = 3
        else:
            rp = rp_true
            y = int(y_true)

        speaker_audio = self._load_audio(sp, apply_augmentation=(self.enable_augmentation and self.augment_speaker))
        reference_audio = self._load_audio(rp, apply_augmentation=(self.enable_augmentation and self.augment_reference))

        return {
            "speaker_audio": speaker_audio,
            "reference_audio": reference_audio,
            "pron_label": int(y),
            "speaker_path": sp or "",
            "reference_path": rp or "",
            "is_synthetic": bool(int(mode) in (2, 3)),
        }


def collate_pronunciation_pair_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}

    B = len(samples)
    speaker_max_len = max(s["speaker_audio"].size(0) for s in samples)
    ref_max_len = max(s["reference_audio"].size(0) for s in samples)

    speaker_audio = torch.zeros(B, speaker_max_len, dtype=torch.float32)
    speaker_padding_mask = torch.ones(B, speaker_max_len, dtype=torch.bool)
    reference_audio = torch.zeros(B, ref_max_len, dtype=torch.float32)
    reference_padding_mask = torch.ones(B, ref_max_len, dtype=torch.bool)

    labels = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(samples):
        sa = s["speaker_audio"]
        ra = s["reference_audio"]

        speaker_audio[i, : sa.size(0)] = sa
        speaker_padding_mask[i, : sa.size(0)] = False

        reference_audio[i, : ra.size(0)] = ra
        reference_padding_mask[i, : ra.size(0)] = False

        labels[i] = int(s["pron_label"])

    return {
        "net_input": {
            "speaker_audio": speaker_audio,
            "speaker_padding_mask": speaker_padding_mask,
            "reference_audio": reference_audio,
            "reference_padding_mask": reference_padding_mask,
            "pron_label": labels,
        },
        "nsentences": B,
        "ntokens": B,
    }
