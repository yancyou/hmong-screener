"""
WavLM backbone for Fairseq training (CTC fine-tuning + feature extraction).

Why this file exists
--------------------
Your codebase currently trains with `model._name=wav2vec_ctc` which expects a
fairseq-style wav2vec2/XLS-R checkpoint (with `cfg/args` that can be converted
to OmegaConf and a `model` state dict that matches fairseq wav2vec2 modules).

Official WavLM checkpoints from Microsoft/UniLM have a different checkpoint
format and model implementation (see https://github.com/microsoft/unilm/tree/master/wavlm).
To run "true backbone training" with WavLM (not frozen feature extraction),
we provide a new fairseq model entrypoint that:
  - loads UniLM WavLM checkpoints (`checkpoint['cfg']`, `checkpoint['model']`)
  - exposes the SAME interface as `wav2vec_ctc`:
      * `model.w2v_encoder(...) -> {'encoder_out', 'padding_mask', ...}`
      * `model.get_logits(net_out)` for ASR CTC
  - keeps the projection head at `w2v_encoder.proj` so existing criteria that
    load/freeze ASR heads continue to work.

Dependency handling
-------------------
fairseq imports all model files at import time. We therefore do NOT hard-require
the UniLM `wavlm` python package at import time. Instead we import it lazily and
raise a clear error if it is missing.

To make `import wavlm` work in your repo, add `FreeVC/` to PYTHONPATH (it
contains the `wavlm/` package).
"""

import contextlib
import logging
import re
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.tasks import FairseqTask

# Reuse the same config surface as wav2vec_ctc so existing YAMLs/overrides work.
from fairseq.models.wav2vec.wav2vec2_asr import (  # noqa: E402
    Wav2Vec2CtcConfig,
    Linear,
    LayerNorm,
)

logger = logging.getLogger(__name__)


def _import_wavlm():
    """
    Lazily import UniLM WavLM implementation.

    Expected to be available as python package `wavlm` (e.g., from `FreeVC/wavlm`).
    """

    try:
        from wavlm import WavLM, WavLMConfig  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Failed to import `wavlm` package required for model._name=wavlm_ctc.\n"
            "Fix: add the directory that contains the `wavlm/` package to PYTHONPATH.\n"
            "In this repo, `FreeVC/wavlm` exists, so you likely want:\n"
            "  export PYTHONPATH=\"${DRIVE_ROOT}/FreeVC:${PYTHONPATH}\"\n"
            f"Original import error: {e}"
        ) from e

    return WavLM, WavLMConfig


def _load_unilm_wavlm_checkpoint(path: str) -> Tuple[dict, dict]:
    """
    Load a UniLM WavLM checkpoint.

    Returns:
      cfg_dict, model_state_dict
    """

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"WavLM checkpoint must be a dict, got: {type(state)}")

    if "cfg" not in state:
        raise ValueError(f"WavLM checkpoint missing 'cfg' key: {path}")

    model_sd = None
    # Official README uses checkpoint['model'].
    if "model" in state and isinstance(state["model"], dict):
        model_sd = state["model"]
    # Some repos wrap weights under 'state_dict'.
    if model_sd is None and "state_dict" in state and isinstance(state["state_dict"], dict):
        model_sd = state["state_dict"]

    if model_sd is None:
        raise ValueError(f"WavLM checkpoint missing 'model'/'state_dict' weights: {path}")

    cfg_dict = state["cfg"]
    if not isinstance(cfg_dict, dict):
        # Some checkpoints may store cfg as OmegaConf / Namespace-like; make it a plain dict.
        try:
            cfg_dict = dict(cfg_dict)
        except Exception as e:
            raise ValueError(f"Unsupported WavLM cfg type: {type(state['cfg'])}: {e}") from e

    return cfg_dict, model_sd


class WavLMEncoder(FairseqEncoder):
    """
    A FairseqEncoder wrapper around UniLM WavLM.

    Output contract matches `Wav2VecEncoder`:
      - encoder_out: [T, B, C] for features_only, or [T, B, V] after CTC head
      - padding_mask: [B, T] bool, at the post-conv frame rate
      - layer_results: optional (list)
    """

    def __init__(self, cfg: Wav2Vec2CtcConfig, output_size: Optional[int] = None):
        super().__init__(dictionary=None)
        self.cfg = cfg
        self.apply_mask = bool(getattr(cfg, "apply_mask", False))
        self.final_dropout = nn.Dropout(float(getattr(cfg, "final_dropout", 0.0)))
        self.freeze_finetune_updates = int(getattr(cfg, "freeze_finetune_updates", 0))
        self.num_updates = 0

        WavLM, WavLMConfig = _import_wavlm()

        cfg_dict, model_sd = _load_unilm_wavlm_checkpoint(cfg.w2v_path)
        wavlm_cfg = WavLMConfig(cfg_dict)
        # Apply fine-tuning overrides from the fairseq model config (mirrors wav2vec_ctc's arg_overrides).
        # IMPORTANT: only override hyperparameters that do NOT change parameter shapes.
        self._apply_finetune_overrides(wavlm_cfg, cfg)
        wavlm_model = WavLM(wavlm_cfg)
        missing, unexpected = wavlm_model.load_state_dict(model_sd, strict=False)
        if missing or unexpected:
            logger.warning(
                "[wavlm_ctc] WavLM load_state_dict non-strict. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        self.w2v_model = wavlm_model

        # ---- build CTC projection head ----
        d = int(getattr(wavlm_cfg, "encoder_embed_dim", 768))
        self.proj = None
        if output_size is not None:
            targ_d = int(output_size)
        else:
            targ_d = None

        if targ_d is not None:
            if bool(getattr(cfg, "use_mlp_head", False)):
                mlp_hidden = int(getattr(cfg, "mlp_head_hidden_dim", d))
                mlp_dropout = float(getattr(cfg, "mlp_head_dropout", 0.1))
                mlp_layers = int(getattr(cfg, "mlp_head_layers", 2))
                use_ln = bool(getattr(cfg, "mlp_head_use_layernorm", True))

                if mlp_layers == 2:
                    layers = [
                        Linear(d, mlp_hidden),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        Linear(mlp_hidden, targ_d),
                    ]
                    arch_desc = f"{d} -> {mlp_hidden} -> {targ_d}"
                elif mlp_layers == 3:
                    layers = [
                        Linear(d, mlp_hidden),
                        LayerNorm(mlp_hidden) if use_ln else nn.Identity(),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        Linear(mlp_hidden, mlp_hidden),
                        LayerNorm(mlp_hidden) if use_ln else nn.Identity(),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        Linear(mlp_hidden, targ_d),
                    ]
                    arch_desc = f"{d} -> {mlp_hidden} -> {mlp_hidden} -> {targ_d}"
                else:
                    raise ValueError(f"mlp_head_layers must be 2 or 3, got {mlp_layers}")

                self.proj = nn.Sequential(*layers)
                logger.info(f"[wavlm_ctc] Using MLP CTC head: {arch_desc} (dropout={mlp_dropout}, ln={use_ln})")
            else:
                self.proj = Linear(d, targ_d)

        # ---- optional freezing ----
        freeze_regex = getattr(cfg, "freeze_regex", None)
        if freeze_regex is not None:
            self.freeze_regex(str(freeze_regex))

        freeze_n = int(getattr(cfg, "freeze_encoder_layers", 0) or 0)
        if freeze_n > 0:
            self._freeze_first_n_transformer_layers(freeze_n)

    @staticmethod
    def _apply_finetune_overrides(wavlm_cfg: Any, cfg: Wav2Vec2CtcConfig) -> None:
        """
        Override WavLMConfig hyperparameters with fine-tuning-time values from fairseq config.

        This makes `wavlm_ctc` behave closer to `wav2vec_ctc`, which overrides dropout/mask
        settings when loading pretrained checkpoints.
        """

        def _set_if_present(attr: str, value: Any) -> None:
            if value is None:
                return
            if hasattr(wavlm_cfg, attr):
                setattr(wavlm_cfg, attr, value)

        # Dropouts / layerdrop
        _set_if_present("dropout", float(getattr(cfg, "dropout", 0.0)))
        _set_if_present("attention_dropout", float(getattr(cfg, "attention_dropout", 0.0)))
        _set_if_present("activation_dropout", float(getattr(cfg, "activation_dropout", 0.0)))
        _set_if_present("dropout_input", float(getattr(cfg, "dropout_input", 0.0)))
        _set_if_present("encoder_layerdrop", float(getattr(cfg, "layerdrop", 0.0)))

        # Feature extractor grad scale
        _set_if_present("feature_grad_mult", float(getattr(cfg, "feature_grad_mult", 1.0)))

        # Masking (only used if cfg.apply_mask is enabled)
        _set_if_present("mask_length", int(getattr(cfg, "mask_length", 10)))
        _set_if_present("mask_prob", float(getattr(cfg, "mask_prob", 0.0)))
        _set_if_present("mask_selection", str(getattr(cfg, "mask_selection", "static")))
        _set_if_present("mask_other", float(getattr(cfg, "mask_other", 0.0)))
        _set_if_present("no_mask_overlap", bool(getattr(cfg, "no_mask_overlap", False)))
        # Optional[int] in fairseq config
        mask_min_space = getattr(cfg, "mask_min_space", None)
        if mask_min_space is not None:
            _set_if_present("mask_min_space", int(mask_min_space))

        # Channel masking (only used if enabled)
        _set_if_present("mask_channel_length", int(getattr(cfg, "mask_channel_length", 10)))
        _set_if_present("mask_channel_prob", float(getattr(cfg, "mask_channel_prob", 0.0)))
        _set_if_present("mask_channel_selection", str(getattr(cfg, "mask_channel_selection", "static")))
        _set_if_present("mask_channel_other", float(getattr(cfg, "mask_channel_other", 0.0)))
        _set_if_present("no_mask_channel_overlap", bool(getattr(cfg, "no_mask_channel_overlap", False)))
        mcm = getattr(cfg, "mask_channel_min_space", None)
        if mcm is not None:
            _set_if_present("mask_channel_min_space", int(mcm))

        # Input normalization: align to task.normalize (fairseq does this in dataset postprocess)
        # WavLM README also applies layer_norm if cfg.normalize.
        _set_if_present("normalize", bool(getattr(cfg, "normalize", False)))

    def _freeze_first_n_transformer_layers(self, n_layers: int) -> None:
        if not (hasattr(self.w2v_model, "encoder") and hasattr(self.w2v_model.encoder, "layers")):
            return
        layers = self.w2v_model.encoder.layers
        total = len(layers)
        k = max(0, min(int(n_layers), total))
        if k == 0:
            return
        # Mirror wav2vec_ctc behavior: never freeze *all* layers (leave at least last trainable).
        if k >= total:
            k = max(0, total - 1)
        for i in range(k):
            for p in layers[i].parameters():
                p.requires_grad_(False)
        for i in range(k, total):
            for p in layers[i].parameters():
                p.requires_grad_(True)
        logger.info(f"[wavlm_ctc] ✓ Froze first {k}/{total} transformer layers (layers 0-{k-1})")
        logger.info(f"[wavlm_ctc] ✓ Trainable transformer layers: {k}-{total-1}")

    def freeze_regex(self, pattern: str) -> None:
        for name, param in self.named_parameters():
            if re.fullmatch(pattern, name) is not None:
                param.requires_grad_(False)

    def set_num_updates(self, num_updates: int) -> None:
        self.num_updates = int(num_updates)
        super().set_num_updates(num_updates)

    def _extract(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        output_layer: Optional[int],
        ret_layer_results: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """
        Call UniLM WavLM.extract_features and normalize output shapes.
        Returns: (x[B,T,C], padding_mask[B,T], layer_results?)
        """

        # UniLM WavLM uses `mask` to apply time masking; for fine-tuning we typically disable it.
        do_mask = self.apply_mask and self.training
        feat, pm = self.w2v_model.extract_features(
            source,
            padding_mask=padding_mask,
            mask=do_mask,
            output_layer=output_layer,
            ret_layer_results=ret_layer_results,
        )

        layer_results = None
        if ret_layer_results and isinstance(feat, tuple) and len(feat) == 2:
            x, layer_results = feat
        else:
            x = feat

        if not (hasattr(x, "dim") and x.dim() == 3):
            raise ValueError(f"[wavlm_ctc] Unexpected extract_features output shape: {type(x)} {getattr(x, 'shape', None)}")

        return x, pm, layer_results

    def forward(self, source, padding_mask, **kwargs) -> Dict[str, Any]:
        """
        Args:
          source: [B, T] float waveform
          padding_mask: [B, T] bool
          kwargs:
            - features_only: if True, skip CTC head
            - layer: optional int (0-based transformer layer index)
        """

        features_only = bool(kwargs.get("features_only", False))
        layer = kwargs.get("layer", None)

        output_layer = None
        ret_layers = False
        if layer is not None:
            try:
                layer_i = int(layer)
            except Exception as e:
                raise ValueError(f"[wavlm_ctc] layer must be int, got {layer}: {e}") from e
            if layer_i >= 0:
                max_layers = int(getattr(self.w2v_model.cfg, "encoder_layers", 0) or 0)
                if max_layers and layer_i >= max_layers:
                    raise ValueError(f"[wavlm_ctc] Requested layer={layer_i} but encoder_layers={max_layers}")
                output_layer = layer_i + 1
        else:
            # layer=None: caller wants all layers (e.g. exp1.py selects later).
            # Request the last layer as output_layer so WavLM populates layer_results.
            max_layers = int(getattr(self.w2v_model.cfg, "encoder_layers", 0) or 0)
            if max_layers > 0 and features_only:
                output_layer = max_layers
                ret_layers = True

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x_btc, pm, layer_results = self._extract(
                source=source,
                padding_mask=padding_mask,
                output_layer=output_layer,
                ret_layer_results=ret_layers,
            )

        # B x T x C -> T x B x C
        x = x_btc.transpose(0, 1)
        x = self.final_dropout(x)

        if self.proj is not None and not features_only:
            x = self.proj(x)

        return {
            "encoder_out": x,
            "padding_mask": pm,
            "layer_results": layer_results,
        }

    def reorder_encoder_out(self, encoder_out: Dict[str, Any], new_order):
        # Minimal implementation (used by seq2seq decoding; safe for CTC too).
        if encoder_out.get("encoder_out", None) is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(1, new_order)
        if encoder_out.get("padding_mask", None) is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(0, new_order)
        return encoder_out


@register_model("wavlm_ctc", dataclass=Wav2Vec2CtcConfig)
class WavLMCTC(BaseFairseqModel):
    """
    Fairseq model wrapper for UniLM WavLM + CTC head.

    Drop-in replacement for `wav2vec_ctc` in your training scripts:
      model._name=wavlm_ctc
      model.w2v_path=/path/to/WavLM-Base+.pt
    """

    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.blank_weight = float(getattr(cfg, "blank_weight", 0.0))
        self.blank_mode = str(getattr(cfg, "blank_mode", "add"))

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        # task.target_dictionary is required for ASR CTC head output dim
        output_size = len(task.target_dictionary) if hasattr(task, "target_dictionary") else None
        w2v_encoder = WavLMEncoder(cfg, output_size=output_size)
        return cls(cfg, w2v_encoder)

    def forward(self, **kwargs):
        return self.w2v_encoder(**kwargs)

    def get_logits(self, net_output, normalize: bool = False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise ValueError(f"invalid blank mode {self.blank_mode}")

        if net_output.get("padding_mask") is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(number_of_classes, device=logits.device) * float("-inf")
            masking_tensor[0] = 0
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        return logits

    def get_normalized_probs(self, net_output, log_probs):
        logits = self.get_logits(net_output, normalize=False)
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        return utils.softmax(logits.float(), dim=-1)

