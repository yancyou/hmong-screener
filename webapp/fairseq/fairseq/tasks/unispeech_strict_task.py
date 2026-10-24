import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from fairseq.data import Dictionary
from fairseq.data.add_target_dataset import AddTargetDataset
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.audio_finetuning import LabelEncoder, label_len_fn


logger = logging.getLogger(__name__)


@dataclass
class UniSpeechStrictTaskConfig(FairseqDataclass):
    asr_data_dir: str = field(
        default=MISSING,
        metadata={"help": "ASR manifest dir containing {split}.tsv and {split}.ltr"},
    )
    asr_dictionary_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to dict.ltr.txt (defaults to asr_data_dir/dict.{labels}.txt)"},
    )
    asr_labels: str = field(default="ltr", metadata={"help": "label extension (ltr, wrd, ...)"})

    sample_rate: int = field(default=16000, metadata={"help": "audio sample rate"})
    max_audio_length: int = field(default=320000, metadata={"help": "max waveform length in samples"})
    min_audio_length: int = field(default=1600, metadata={"help": "min waveform length in samples"})
    normalize: bool = field(default=True, metadata={"help": "layer-norm waveform"})
    pad_audio: bool = field(default=True, metadata={"help": "pad audio and create padding_mask"})


@register_task("unispeech_strict", dataclass=UniSpeechStrictTaskConfig)
class UniSpeechStrictTask(FairseqTask):
    """
    ASR-only task for UniSpeech-strict baseline:
    - Provides labeled audio batches (waveform + CTC targets).
    - Works with a wav2vec2-style pretraining model (masking + quantizer).
    """

    cfg: UniSpeechStrictTaskConfig

    def __init__(self, cfg: UniSpeechStrictTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = Dictionary()
        self.blank_symbol = "<s>"

        dict_path = None
        if cfg.asr_dictionary_file and os.path.exists(cfg.asr_dictionary_file):
            dict_path = cfg.asr_dictionary_file
        else:
            cand = os.path.join(cfg.asr_data_dir, f"dict.{cfg.asr_labels}.txt")
            if os.path.exists(cand):
                dict_path = cand

        if dict_path is None:
            raise FileNotFoundError(
                f"Could not find ASR dictionary. Set task.asr_dictionary_file or ensure "
                f"{os.path.join(cfg.asr_data_dir, f'dict.{cfg.asr_labels}.txt')} exists."
            )

        self.dictionary = Dictionary.load(dict_path)
        logger.info(f"[unispeech_strict_task] Loaded dictionary from {dict_path} (size={len(self.dictionary)})")

    @classmethod
    def setup_task(cls, cfg: UniSpeechStrictTaskConfig, **kwargs):
        return cls(cfg)

    def _resolve_split_paths(self, split: str):
        manifest = os.path.join(self.cfg.asr_data_dir, f"{split}.tsv")
        labels = os.path.join(self.cfg.asr_data_dir, f"{split}.{self.cfg.asr_labels}")
        if os.path.exists(manifest):
            return manifest, labels

        # Common naming: valid <-> val
        if split == "valid":
            alt = "val"
        elif split == "val":
            alt = "valid"
        else:
            alt = None

        if alt is not None:
            manifest2 = os.path.join(self.cfg.asr_data_dir, f"{alt}.tsv")
            labels2 = os.path.join(self.cfg.asr_data_dir, f"{alt}.{self.cfg.asr_labels}")
            if os.path.exists(manifest2):
                return manifest2, labels2

        return manifest, labels

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        manifest_path, label_path = self._resolve_split_paths(split)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing labels: {label_path}")

        is_train = split.startswith("train")

        audio_ds = FileAudioDataset(
            manifest_path=manifest_path,
            sample_rate=self.cfg.sample_rate,
            max_sample_size=self.cfg.max_audio_length,
            min_sample_size=self.cfg.min_audio_length,
            shuffle=is_train,
            pad=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
        )

        text_compressor = TextCompressor(level=TextCompressionLevel.none)
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [text_compressor.compress(l.rstrip("\n")) for l in f]

        process_label = LabelEncoder(self.dictionary)
        ds = AddTargetDataset(
            audio_ds,
            labels,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            label_len_fn=label_len_fn,
            add_to_input=False,
        )

        self.datasets[split] = ds
        logger.info(f"[unispeech_strict_task] Loaded {len(ds)} samples for split={split}")

    def max_positions(self):
        # AddTargetDataset.size() returns (audio_size, label_size), so
        # max_positions must also be a tuple for the comparison in
        # _filter_by_size_dynamic to work.
        return (self.cfg.max_audio_length, sys.maxsize)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

