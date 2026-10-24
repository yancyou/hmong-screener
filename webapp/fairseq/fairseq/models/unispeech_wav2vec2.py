import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

import torch

from fairseq import checkpoint_utils, models
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, register_model


logger = logging.getLogger(__name__)


@dataclass
class UniSpeechWav2Vec2Config(FairseqDataclass):
    """
    Thin wrapper model that:
    - Loads a pretrained fairseq `wav2vec2` checkpoint (keeps quantizer + masking modules).
    - Delegates forward/extract_features/get_logits/get_targets/get_extra_losses to the underlying
      `Wav2Vec2Model`, so a custom criterion can implement UniSpeech-style objectives.
    """

    w2v_path: str = field(
        default="",
        metadata={"help": "path to pretrained wav2vec2 / XLS-R checkpoint (.pt)"},
    )
    data: str = field(
        default="",
        metadata={"help": "data root placeholder used when building from checkpoint cfg"},
    )

    # Masking hyperparameters (UniSpeech uses low mask prob, e.g., p=0.05, len=10)
    apply_mask: bool = field(default=True, metadata={"help": "apply time masking"})
    mask_prob: float = field(default=0.05, metadata={"help": "mask probability"})
    mask_length: int = field(default=10, metadata={"help": "mask span length"})
    mask_selection: str = field(default="static", metadata={"help": "mask selection strategy"})
    mask_other: float = field(default=0.0, metadata={"help": "mask selection 'other' param"})
    no_mask_overlap: bool = field(default=False, metadata={"help": "disallow mask overlaps"})
    mask_min_space: int = field(default=1, metadata={"help": "min space between masks"})

    # Channel masking (usually disabled for UniSpeech)
    mask_channel_prob: float = field(default=0.0, metadata={"help": "channel mask prob"})
    mask_channel_length: int = field(default=10, metadata={"help": "channel mask length"})
    mask_channel_selection: str = field(default="static", metadata={"help": "channel mask selection"})
    mask_channel_other: float = field(default=0.0, metadata={"help": "channel mask 'other' param"})
    no_mask_channel_overlap: bool = field(default=False, metadata={"help": "disallow channel mask overlap"})
    mask_channel_min_space: int = field(default=1, metadata={"help": "min space between channel masks"})

    # Misc
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if set, do not load weights from w2v_path (build architecture only)"},
    )


@register_model("unispeech_wav2vec2", dataclass=UniSpeechWav2Vec2Config)
class UniSpeechWav2Vec2Model(BaseFairseqModel):
    """
    Wrapper around fairseq `Wav2Vec2Model` that keeps pretraining modules.
    """

    def __init__(self, cfg: UniSpeechWav2Vec2Config, w2v_model: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_model = w2v_model

        # Apply UniSpeech-style masking overrides on the underlying model
        self._apply_mask_overrides()

    @classmethod
    def build_model(cls, cfg: UniSpeechWav2Vec2Config, task=None):
        if not cfg.w2v_path:
            raise ValueError("model.w2v_path must be set for unispeech_wav2vec2")

        state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides=None)
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        elif not hasattr(w2v_args, "model"):
            # Some older checkpoints store cfg as a dict-like container.
            w2v_args = convert_namespace_to_omegaconf(w2v_args)

        # Ensure we can build the model even if checkpoint task has a data key.
        try:
            if hasattr(w2v_args, "task") and hasattr(w2v_args.task, "data") and cfg.data:
                w2v_args.task.data = cfg.data
        except Exception:
            pass

        w2v_model = models.build_model(w2v_args.model, task=None, from_checkpoint=True)

        if not cfg.no_pretrained_weights:
            sd = state.get("model", None)
            if sd is None:
                raise ValueError(f"Checkpoint at {cfg.w2v_path} has no 'model' state dict")
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            missing, unexpected = w2v_model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"[unispeech_wav2vec2] Missing keys when loading: {missing[:10]} (and {max(0, len(missing)-10)} more)")
            if unexpected:
                logger.warning(f"[unispeech_wav2vec2] Unexpected keys when loading: {unexpected[:10]} (and {max(0, len(unexpected)-10)} more)")

        return cls(cfg, w2v_model)

    def _apply_mask_overrides(self) -> None:
        # Underlying `Wav2Vec2Model` caches these as attributes used in apply_mask()
        m = self.w2v_model
        m.mask_prob = float(self.cfg.mask_prob)
        m.mask_selection = str(self.cfg.mask_selection)
        m.mask_other = float(self.cfg.mask_other)
        m.mask_length = int(self.cfg.mask_length)
        m.no_mask_overlap = bool(self.cfg.no_mask_overlap)
        m.mask_min_space = int(self.cfg.mask_min_space)

        m.mask_channel_prob = float(self.cfg.mask_channel_prob)
        m.mask_channel_selection = str(self.cfg.mask_channel_selection)
        m.mask_channel_other = float(self.cfg.mask_channel_other)
        m.mask_channel_length = int(self.cfg.mask_channel_length)
        m.no_mask_channel_overlap = bool(self.cfg.no_mask_channel_overlap)
        m.mask_channel_min_space = int(self.cfg.mask_channel_min_space)

    def _estimate_num_frames(self, wav_len: int) -> int:
        """Approximate number of frames after the CNN feature extractor."""
        n = wav_len
        conv_layers = eval(self.w2v_model.cfg.conv_feature_layers)
        for _, kernel, stride in conv_layers:
            n = (n - kernel) // stride + 1
            if n <= 0:
                return 0
        return n

    def forward(
        self,
        source,
        padding_mask=None,
        mask: Optional[bool] = None,
        features_only: bool = False,
        layer: Optional[int] = None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
        corpus_key=None,
        **kwargs,
    ):
        if mask is None:
            mask = bool(self.cfg.apply_mask)

        # Safety: disable masking when sequence is too short to avoid
        # "the entire sequence is masked" ValueError.
        if mask and source is not None:
            min_wav_len = source.size(-1)
            est_frames = self._estimate_num_frames(min_wav_len)
            min_safe = int(self.cfg.mask_length) * 2 + 2
            if est_frames < min_safe:
                mask = False

        return self.w2v_model.forward(
            source=source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=features_only,
            layer=layer,
            mask_indices=mask_indices,
            mask_channel_indices=mask_channel_indices,
            padding_count=padding_count,
            corpus_key=corpus_key,
        )

    def extract_features(self, source, padding_mask, mask=False, layer=None, corpus_key=None):
        # Route through our forward() so the short-sequence safety check applies.
        return self.forward(
            source=source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
            corpus_key=corpus_key,
        )

    def get_logits(self, net_output):
        return self.w2v_model.get_logits(net_output)

    def get_targets(self, sample, net_output, expand_steps=True):
        return self.w2v_model.get_targets(sample, net_output, expand_steps=expand_steps)

    def get_extra_losses(self, net_output):
        if hasattr(self.w2v_model, "get_extra_losses"):
            return self.w2v_model.get_extra_losses(net_output)
        return []

    def quantize(self, x):
        return self.w2v_model.quantize(x)

