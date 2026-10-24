import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics
from fairseq.logging.meters import safe_round


@dataclass
class UniSpeechStrictCriterionConfig(FairseqDataclass):
    """
    Strict UniSpeech baseline criterion (Wang et al., 2021):
    - wav2vec2-style contrastive loss (masked) + diversity loss
    - supervised CTC loss on mixed representations with replacement prob r

    Eq.(6): L = sum_{(x,y) in L} [ alpha * L_ctc(x,y) + (1-alpha) * L_self(x) ]
    Eq.(7): CTC uses c' where c'_t = q_t with prob r else c_t
    """

    alpha: float = field(default=0.5, metadata={"help": "CTC weight alpha (self weight is 1-alpha)"})
    replace_prob: float = field(default=0.5, metadata={"help": "replacement probability r for CTC"})

    diversity_weight: float = field(default=0.1, metadata={"help": "weight for diversity loss term"})
    features_pen_weight: float = field(default=0.0, metadata={"help": "weight for feature penalty term"})

    ctc_dropout: float = field(default=0.0, metadata={"help": "dropout on CTC inputs (after mixing)"})
    zero_infinity: bool = field(
        default=True, metadata={"help": "zero inf loss when input length <= target length"}
    )

    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "extra output keys from wav2vec2 forward() to log if present"},
    )


@register_criterion("unispeech_strict", dataclass=UniSpeechStrictCriterionConfig)
class UniSpeechStrictCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        alpha: float = 0.5,
        replace_prob: float = 0.5,
        diversity_weight: float = 0.1,
        features_pen_weight: float = 0.0,
        ctc_dropout: float = 0.0,
        zero_infinity: bool = True,
        log_keys: Optional[List[str]] = None,
    ):
        super().__init__(task)
        self.alpha = float(alpha)
        self.replace_prob = float(replace_prob)
        self.diversity_weight = float(diversity_weight)
        self.features_pen_weight = float(features_pen_weight)
        self.zero_infinity = bool(zero_infinity)
        self.log_keys = [] if log_keys is None else list(log_keys)

        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()

        self._ctc_head: Optional[nn.Linear] = None
        self._ctc_dropout = nn.Dropout(ctc_dropout) if ctc_dropout and ctc_dropout > 0 else None

    def _get_w2v_model(self, model):
        # Support both raw `Wav2Vec2Model` and wrapper models that store it under `.w2v_model`.
        return getattr(model, "w2v_model", model)

    def _maybe_build_ctc_head(self, in_dim: int, vocab_size: int, device: torch.device):
        if self._ctc_head is None:
            self._ctc_head = nn.Linear(in_dim, vocab_size).to(device)

    def _contrastive_loss(self, model, sample):
        """
        wav2vec2-style InfoNCE (cross-entropy) on masked positions, plus optional
        diversity/features penalties from `model.get_extra_losses()`.
        Returns: (self_loss, logs_dict)
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        targets = model.get_targets(sample, net_output)

        # InfoNCE CE: positive is index 0
        loss_ce = F.cross_entropy(logits, targets, reduction="mean")

        # Extra losses from wav2vec2 model
        diversity = None
        features_pen = None
        extra = None
        w2v = self._get_w2v_model(model)
        if hasattr(w2v, "get_extra_losses"):
            extra = w2v.get_extra_losses(net_output)
            if torch.is_tensor(extra):
                extra = [extra]
            if isinstance(extra, list):
                if len(extra) > 0:
                    diversity = extra[0]
                if len(extra) > 1:
                    features_pen = extra[1]

        loss_self = loss_ce
        if diversity is not None and self.diversity_weight != 0.0:
            loss_self = loss_self + self.diversity_weight * diversity.float()
        if features_pen is not None and self.features_pen_weight != 0.0:
            loss_self = loss_self + self.features_pen_weight * features_pen.float()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == 0).float().mean() if logits.numel() else logits.new_tensor(0.0)

        logs = {
            "loss_self_ce": loss_ce.detach(),
            "acc_self": acc.detach(),
        }
        if diversity is not None:
            logs["loss_self_div"] = diversity.detach()
        if features_pen is not None:
            logs["loss_self_fpen"] = features_pen.detach()
        for k in self.log_keys:
            if k in net_output and not torch.is_tensor(net_output[k]):
                # avoid non-tensors; keep tensors/nums only
                continue
            if k in net_output:
                v = net_output[k]
                logs[f"w2v_{k}"] = v.detach() if torch.is_tensor(v) else v

        return loss_self, logs

    def _ctc_with_replacement_loss(self, model, sample):
        """
        Compute CTC on mixed representations:
          c'_t = q_t w.p. r else c_t
        where c_t is contextual rep and q_t is quantized latent rep projected to same dim.
        """
        w2v = self._get_w2v_model(model)

        source = sample["net_input"]["source"]
        padding_mask = sample["net_input"].get("padding_mask", None)
        targets = sample["target"]
        target_lengths = sample["target_lengths"]

        # Extract contextual representations (full sequence) and unmasked features for quantizer
        feat_out = model.extract_features(source, padding_mask, mask=True, layer=None)
        c_enc = feat_out["x"]  # [B, T, Cenc]
        pad_frames = feat_out.get("padding_mask", None)  # [B, T] at conv frame rate
        unmasked = feat_out.get("features", None)  # [B, T, embed] (pre-encoder, unmasked)

        if unmasked is None:
            raise RuntimeError("model.extract_features() did not return 'features' needed for quantizer")
        if not hasattr(w2v, "quantizer") or w2v.quantizer is None:
            raise RuntimeError("Underlying wav2vec2 model has no quantizer; UniSpeech-strict requires quantize_targets=True")

        # Project contextual to final_dim (same space used by project_q)
        if not hasattr(w2v, "final_proj") or w2v.final_proj is None:
            raise RuntimeError("Underlying wav2vec2 model has no final_proj; cannot align c and q for replacement")
        c = w2v.final_proj(c_enc)  # [B, T, D]

        q_out = w2v.quantizer(unmasked, produce_targets=False)
        q = q_out["x"]
        if not hasattr(w2v, "project_q") or w2v.project_q is None:
            raise RuntimeError("Underlying wav2vec2 model has no project_q; cannot align c and q for replacement")
        q = w2v.project_q(q)  # [B, T, D]

        B, T, D = c.shape
        replace = (torch.rand(B, T, device=c.device) < self.replace_prob)
        if pad_frames is not None:
            replace = replace & (~pad_frames)
            denom = (~pad_frames).float().sum().clamp_min(1.0)
        else:
            denom = float(B * T)
        replaced_ratio = replace.float().sum() / denom

        c_mix = torch.where(replace.unsqueeze(-1), q, c)
        if self._ctc_dropout is not None:
            c_mix = self._ctc_dropout(c_mix)

        vocab_size = len(self.task.target_dictionary)
        self._maybe_build_ctc_head(in_dim=D, vocab_size=vocab_size, device=c_mix.device)
        logits = self._ctc_head(c_mix)  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1).contiguous()  # [T, B, V]

        if pad_frames is not None:
            input_lengths = (~pad_frames).long().sum(-1)
        else:
            input_lengths = targets.new_full((B,), T, dtype=torch.long)

        # Flatten targets (ignore pad/eos)
        pad_mask = (targets != self.pad_idx) & (targets != self.eos_idx)
        targets_flat = targets.masked_select(pad_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss_ctc = F.ctc_loss(
                log_probs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="mean",
                zero_infinity=self.zero_infinity,
            )

        logs = {
            "loss_ctc": loss_ctc.detach(),
            "ctc_replaced_ratio": replaced_ratio.detach(),
        }
        return loss_ctc, logs

    def forward(self, model, sample, reduce=True):
        if sample is None or len(sample) == 0:
            dummy = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
            return dummy, 0, {"loss": dummy.data, "sample_size": 0}

        bsz = int(sample["id"].numel()) if "id" in sample and sample["id"] is not None else 1
        sample_size = bsz

        loss_self, logs_self = self._contrastive_loss(model, sample)
        loss_ctc, logs_ctc = self._ctc_with_replacement_loss(model, sample) if self.alpha > 0 else (loss_self.new_tensor(0.0), {})

        # Eq.(6): alpha * CTC + (1-alpha) * self
        loss_mean = (1.0 - self.alpha) * loss_self + self.alpha * loss_ctc

        # Return sum-style loss for fairseq's gradient normalization (trainer divides by sample_size).
        loss = loss_mean * sample_size

        logging_output = {
            "loss": loss.detach(),
            "sample_size": sample_size,
            "nsentences": bsz,
            "loss_self": (loss_self.detach() * sample_size),
            "loss_ctc": (loss_ctc.detach() * sample_size),
        }
        logging_output.update(logs_self)
        logging_output.update(logs_ctc)
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nsentences", nsentences)

        # Component losses (already logged as sum over sample_size)
        self_sum = utils.item(sum(log.get("loss_self", 0) for log in logging_outputs))
        ctc_sum = utils.item(sum(log.get("loss_ctc", 0) for log in logging_outputs))
        if sample_size > 0:
            metrics.log_scalar("loss_self", self_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar("loss_ctc", ctc_sum / sample_size / math.log(2), sample_size, round=3)

        # Contrastive accuracy (mean over steps)
        acc_vals = [log.get("acc_self", None) for log in logging_outputs if "acc_self" in log]
        if acc_vals:
            metrics.log_scalar("acc_self", sum(acc_vals) / len(acc_vals), len(acc_vals), round=4)

        # Replacement ratio
        rep_vals = [log.get("ctc_replaced_ratio", None) for log in logging_outputs if "ctc_replaced_ratio" in log]
        if rep_vals:
            metrics.log_scalar("ctc_replaced_ratio", sum(rep_vals) / len(rep_vals), len(rep_vals), round=4)

        # Optional wav2vec2 keys (prob/code perplexity, temp, etc.)
        for key in set(k for log in logging_outputs for k in log.keys()):
            if key.startswith("w2v_"):
                vals = [log.get(key, None) for log in logging_outputs if key in log]
                if vals:
                    try:
                        metrics.log_scalar(key, sum(vals) / len(vals), len(vals), round=4)
                    except Exception:
                        pass

    def logging_outputs_can_be_summed(self) -> bool:
        return True

