# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
PronunciationPairCriterion
==========================

Implements a 3-class classifier over a (reference, speaker) embedding pair.

Given:
  u: reference audio pooled embedding (L2-normalized)
  v: speaker audio pooled embedding (L2-normalized)

Two head designs:
  (1) concat_mul:
        f(u,v) = Linear([u; v; u*v])
  (2) cosine_linear:
        f(u,v) = Linear(CosSim(u,v))    # CosSim is a scalar

Labels (kept as 1/2/3 in data, remapped to 0/1/2 internally):
  1: exactly correct           -> internal 0
  2: correct base word, wrong pronunciation  -> internal 1
  3: wrong base word, wrong pronunciation    -> internal 2

Binary mode (optional):
  If num_classes == 2, we collapse labels as:
    correct   = (raw_label == 1) -> internal 0
    incorrect = (raw_label in {2,3}) -> internal 1

Imbalance support:
  - optional class weights (manual or "auto" from dataset label_counts)
  - optional focal loss via focal_gamma > 0
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics


@dataclass
class PronunciationPairCriterionConfig(FairseqDataclass):
    lambda_pron: float = field(
        default=1.0,
        metadata={"help": "weight for pronunciation classification loss"},
    )
    pair_head: str = field(
        default="concat_mul",
        metadata={"help": "pair head type: concat_mul | cosine_linear"},
    )
    num_classes: int = field(
        default=3,
        metadata={"help": "number of classes (default=3 for correct/mispron/wrongword)"},
    )

    # embedding extraction
    pooling_mode: str = field(
        default="mean",
        metadata={"help": "pooling over time: mean, max, or last"},
    )
    embedding_extraction_layer: int = field(
        default=-1,
        metadata={"help": "w2v encoder layer index to extract (-1=last/all)"},
    )
    freeze_encoder_layers: int = field(
        default=0,
        metadata={"help": "freeze first N encoder transformer layers (0 disables)"},
    )
    freeze_entire_encoder: bool = field(
        default=False,
        metadata={"help": "if true, freeze ALL encoder components (CNN, pos_conv, all layers, layer_norm). Overrides freeze_encoder_layers."},
    )
    freeze_feature_extractor: bool = field(
        default=False,
        metadata={"help": "also freeze feature_extractor + post_extract_proj when freezing encoder layers"},
    )

    # imbalance
    class_weights: Optional[Any] = field(
        default=None,
        metadata={
            "help": "Optional class weights. Use a list of length num_classes, or 'auto' to infer from dataset."
        },
    )
    focal_gamma: float = field(
        default=0.0,
        metadata={"help": "focal gamma (0 disables focal loss)"},
    )

    # head dimensions
    head_embed_dim: int = field(
        default=1024,
        metadata={"help": "embedding dimension for the pair head (must match w2v encoder_embed_dim, default=1024 for XLS-R 300M)"},
    )

    # logging
    log_cosine_by_label: bool = field(
        default=True,
        metadata={"help": "log cosine similarity means by label (cheap, batch-level)"},
    )


@register_criterion("pronunciation_pair", dataclass=PronunciationPairCriterionConfig)
class PronunciationPairCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        lambda_pron: float = 1.0,
        pair_head: str = "concat_mul",
        num_classes: int = 3,
        pooling_mode: str = "mean",
        embedding_extraction_layer: int = -1,
        freeze_encoder_layers: int = 0,
        freeze_entire_encoder: bool = False,
        freeze_feature_extractor: bool = False,
        class_weights: Optional[Any] = None,
        focal_gamma: float = 0.0,
        head_embed_dim: int = 1024,
        log_cosine_by_label: bool = True,
    ):
        super().__init__(task)
        self.task = task

        self.lambda_pron = float(lambda_pron)
        self.pair_head = str(pair_head)
        self.num_classes = int(num_classes)
        self.pooling_mode = str(pooling_mode)
        self.embedding_extraction_layer = int(embedding_extraction_layer)
        self.freeze_encoder_layers = int(freeze_encoder_layers)
        self.freeze_entire_encoder = bool(freeze_entire_encoder)
        self.freeze_feature_extractor = bool(freeze_feature_extractor)

        self.class_weights_cfg = class_weights
        self.focal_gamma = float(focal_gamma)
        self.log_cosine_by_label = bool(log_cosine_by_label)

        self._layers_frozen = False
        self._weights_ready = False
        self._class_weights_tensor: Optional[torch.Tensor] = None

        # ---- Eager init of pair classifier ----
        # Must be created here so its parameters are visible to
        # criterion.parameters() when the optimizer is built.
        head = self.pair_head.lower().strip()
        embed_dim = int(head_embed_dim)
        if head == "concat_mul":
            in_dim = embed_dim * 3
            self.pron_classifier = nn.Linear(in_dim, self.num_classes)
        elif head == "cosine_linear":
            self.pron_classifier = nn.Linear(1, self.num_classes)
        else:
            raise ValueError(f"Unknown pair_head: {self.pair_head} (expected concat_mul|cosine_linear)")

        nn.init.xavier_uniform_(self.pron_classifier.weight)
        if self.pron_classifier.bias is not None:
            nn.init.zeros_(self.pron_classifier.bias)

        print("[PronunciationPairCriterion] Initialized with:")
        print(f"  lambda_pron={self.lambda_pron}")
        print(f"  pair_head={self.pair_head}")
        print(f"  num_classes={self.num_classes}")
        print(f"  pooling_mode={self.pooling_mode}")
        print(f"  embedding_extraction_layer={self.embedding_extraction_layer}")
        print(f"  freeze_encoder_layers={self.freeze_encoder_layers}")
        print(f"  freeze_entire_encoder={self.freeze_entire_encoder}")
        print(f"  freeze_feature_extractor={self.freeze_feature_extractor}")
        print(f"  head_embed_dim={embed_dim}")
        print(f"  class_weights={self.class_weights_cfg}")
        print(f"  focal_gamma={self.focal_gamma}")
        print(f"  pron_classifier params: {sum(p.numel() for p in self.pron_classifier.parameters())}")

    # ---------------------------
    # Embedding extraction helpers
    # ---------------------------
    def extract_embedding(self, model, audio_tensor, padding_mask=None):
        """
        Extract and pool embeddings from the wav2vec encoder.

        Returns: pooled embeddings [B, C]
        """
        if hasattr(model, "w2v_encoder"):
            encoder_out = model.w2v_encoder(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
                layer=self.embedding_extraction_layer if self.embedding_extraction_layer >= 0 else None,
            )
            # encoder_out['encoder_out'] is [T, B, C]
            x = encoder_out["encoder_out"].transpose(0, 1)  # [B, T, C]
            padding_mask = encoder_out.get("padding_mask", padding_mask)
        elif hasattr(model, "extract_features"):
            result = model.extract_features(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                layer=self.embedding_extraction_layer if self.embedding_extraction_layer >= 0 else -1,
            )
            x = result["x"]  # [B, T, C]
        else:
            raise ValueError("Model must have w2v_encoder or extract_features")

        # Pool over time
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            lengths = (~padding_mask).sum(dim=1).float().clamp(min=1.0)
        else:
            lengths = torch.full(
                (x.size(0),),
                x.size(1),
                dtype=torch.float,
                device=x.device,
            )

        if self.pooling_mode == "mean":
            pooled = x.sum(dim=1) / lengths.unsqueeze(-1)
        elif self.pooling_mode == "max":
            pooled = x.max(dim=1)[0]
        elif self.pooling_mode == "last":
            last_indices = (lengths - 1).long().clamp(min=0)
            pooled = x[torch.arange(x.size(0), device=x.device), last_indices]
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        return pooled

    def _maybe_freeze_encoder_layers(self, model):
        if self._layers_frozen:
            return
        if not self.freeze_entire_encoder and not self.freeze_feature_extractor and self.freeze_encoder_layers <= 0:
            return

        try:
            w2v_model = None
            if hasattr(model, "w2v_encoder") and hasattr(model.w2v_encoder, "w2v_model"):
                w2v_model = model.w2v_encoder.w2v_model

            # Handle nested Wav2VecCtc structure (common when loading via w2v_path)
            if w2v_model is not None and hasattr(w2v_model, "w2v_encoder") and hasattr(w2v_model.w2v_encoder, "w2v_model"):
                w2v_model = w2v_model.w2v_encoder.w2v_model

            if w2v_model is None:
                print("[PronunciationPairCriterion] Warning: could not locate w2v_model for freezing")
                return

            frozen_params = 0

            if self.freeze_entire_encoder:
                # ---- Freeze EVERYTHING in the encoder (CNN, pos_conv, layers, norms) ----
                # 1. feature_extractor (CNN)
                if hasattr(w2v_model, "feature_extractor"):
                    for p in w2v_model.feature_extractor.parameters():
                        p.requires_grad = False
                        frozen_params += p.numel()

                # 2. post_extract_proj
                if hasattr(w2v_model, "post_extract_proj"):
                    for p in w2v_model.post_extract_proj.parameters():
                        p.requires_grad = False
                        frozen_params += p.numel()

                # 3. layer_norm (input layer norm)
                if hasattr(w2v_model, "layer_norm"):
                    for p in w2v_model.layer_norm.parameters():
                        p.requires_grad = False
                        frozen_params += p.numel()

                # 4. encoder (pos_conv + all transformer layers + final layer_norm)
                if hasattr(w2v_model, "encoder"):
                    for p in w2v_model.encoder.parameters():
                        p.requires_grad = False
                        frozen_params += p.numel()

                print(f"[PronunciationPairCriterion] Frozen ENTIRE encoder ({frozen_params:,} params)")

            else:
                # ---- Freeze CNN / feature_extractor if requested ----
                if self.freeze_feature_extractor:
                    if hasattr(w2v_model, "feature_extractor"):
                        for p in w2v_model.feature_extractor.parameters():
                            p.requires_grad = False
                            frozen_params += p.numel()
                    if hasattr(w2v_model, "post_extract_proj"):
                        for p in w2v_model.post_extract_proj.parameters():
                            p.requires_grad = False
                            frozen_params += p.numel()
                    print(f"[PronunciationPairCriterion] Frozen feature_extractor + post_extract_proj ({frozen_params:,} params)")

                # ---- Freeze first N transformer layers ----
                if self.freeze_encoder_layers > 0 and hasattr(w2v_model, "encoder") and hasattr(w2v_model.encoder, "layers"):
                    encoder = w2v_model.encoder
                    total_layers = len(encoder.layers)
                    n_freeze = min(self.freeze_encoder_layers, total_layers)
                    for i in range(n_freeze):
                        for p in encoder.layers[i].parameters():
                            p.requires_grad = False
                            frozen_params += p.numel()
                    print(f"[PronunciationPairCriterion] Frozen first {n_freeze}/{total_layers} encoder layers ({frozen_params:,} params)")

        except Exception as e:
            print(f"[PronunciationPairCriterion] Warning: failed to freeze encoder layers: {e}")
        finally:
            self._layers_frozen = True

    # ---------------------------
    # Class weights & loss helpers
    # ---------------------------
    def _maybe_prepare_class_weights(self, device: torch.device, dtype: torch.dtype):
        if self._weights_ready:
            # ensure weights on correct device
            if self._class_weights_tensor is not None and self._class_weights_tensor.device != device:
                self._class_weights_tensor = self._class_weights_tensor.to(device=device)
            return

        w = None
        cfg = self.class_weights_cfg

        # Manual list/tuple
        if isinstance(cfg, (list, tuple)):
            if len(cfg) != self.num_classes:
                raise ValueError(f"class_weights must have length {self.num_classes}, got {len(cfg)}")
            w = torch.tensor([float(x) for x in cfg], device=device, dtype=torch.float32)

        # "auto" from dataset distribution (if available)
        elif isinstance(cfg, str) and cfg.strip().lower() == "auto":
            counts = None
            try:
                ds = getattr(self.task, "datasets", {}).get("train", None)
                if ds is not None and hasattr(ds, "label_counts"):
                    counts = ds.label_counts
            except Exception:
                counts = None

            if counts is None:
                print("[PronunciationPairCriterion] class_weights='auto' but no dataset.label_counts found; disabling weights.")
                w = None
            else:
                # inverse-frequency with mean normalization
                # counts dict uses original labels {1,2,3}.
                if self.num_classes == 2:
                    # collapse negatives: {2,3} -> class-1
                    c = torch.tensor(
                        [float(counts.get(1, 0)), float(counts.get(2, 0) + counts.get(3, 0))],
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    # internal indices {0,1,2} map to original {1,2,3}
                    c = torch.tensor(
                        [float(counts.get(i + 1, 0)) for i in range(self.num_classes)],
                        device=device,
                        dtype=torch.float32,
                    )
                c = torch.clamp(c, min=1.0)
                inv = 1.0 / c
                inv = inv * (inv.numel() / inv.sum())  # mean weight ~= 1
                w = inv
                if self.num_classes == 2:
                    print(
                        f"[PronunciationPairCriterion] Auto class weights (binary) from counts="
                        f"{{pos(label1):{counts.get(1,0)}, neg(label2+3):{counts.get(2,0)+counts.get(3,0)}}}: {w.tolist()}"
                    )
                else:
                    print(
                        f"[PronunciationPairCriterion] Auto class weights from counts="
                        f"{{1:{counts.get(1,0)}, 2:{counts.get(2,0)}, 3:{counts.get(3,0)}}}: {w.tolist()}"
                    )

        else:
            w = None

        self._class_weights_tensor = w
        self._weights_ready = True

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor], gamma: float):
        """
        Focal loss for multi-class classification.
        loss_i = - w_y * (1 - p_y)^gamma * log(p_y)
        """
        logp = F.log_softmax(logits, dim=-1)  # [B, K]
        # gather log(p_y)
        tgt = targets.long()
        logpt = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)  # [B]
        pt = logpt.exp()
        focal = (1.0 - pt).clamp(min=0.0) ** float(gamma)
        loss = -focal * logpt
        if weight is not None:
            loss = loss * weight.gather(0, tgt)
        return loss.mean()

    def forward(self, model, sample, reduce=True):
        net_input = sample["net_input"]
        speaker_audio = net_input["speaker_audio"]
        reference_audio = net_input["reference_audio"]
        speaker_mask = net_input.get("speaker_padding_mask", None)
        reference_mask = net_input.get("reference_padding_mask", None)
        raw_labels = net_input["pron_label"].long()
        # Remap data labels -> internal class indices for CE loss.
        #  - 3-class: {1,2,3} -> {0,1,2}
        #  - 2-class: {1} -> 0 (correct), {2,3} -> 1 (incorrect)
        if self.num_classes == 2:
            labels = (raw_labels != 1).long()
        else:
            labels = raw_labels - 1

        # Defensive guard: avoid CUDA device-side asserts if labels are out-of-range.
        # This can happen if an older checkpoint / codepath produced {0,1,2} labels but num_classes==2.
        try:
            min_lbl = int(labels.min().item())
            max_lbl = int(labels.max().item())
        except Exception:
            min_lbl, max_lbl = 0, self.num_classes - 1
        if min_lbl < 0 or max_lbl >= self.num_classes:
            if self.num_classes == 2:
                labels = (raw_labels != 1).long()
                min_lbl = int(labels.min().item())
                max_lbl = int(labels.max().item())
            if min_lbl < 0 or max_lbl >= self.num_classes:
                raise ValueError(
                    f"Label out of range for num_classes={self.num_classes}: "
                    f"min={min_lbl} max={max_lbl}. "
                    f"Raw labels unique={sorted(set(int(x) for x in raw_labels.detach().cpu().tolist()))[:20]}"
                )

        self._maybe_freeze_encoder_layers(model)

        # Extract embeddings
        v = self.extract_embedding(model, speaker_audio, speaker_mask)  # [B,C]
        u = self.extract_embedding(model, reference_audio, reference_mask)  # [B,C]

        # Normalize features (as per design)
        v = F.normalize(v, dim=-1)
        u = F.normalize(u, dim=-1)

        B, C = u.size(0), u.size(1)
        device = u.device
        dtype = u.dtype

        # Move classifier to correct device/dtype on first forward (e.g. CPU -> GPU)
        if next(self.pron_classifier.parameters()).device != device:
            self.pron_classifier = self.pron_classifier.to(device=device, dtype=dtype)

        self._maybe_prepare_class_weights(device=device, dtype=dtype)

        head = self.pair_head.lower().strip()
        if head == "concat_mul":
            feat = torch.cat([u, v, u * v], dim=-1)  # [B, 3C]
            logits = self.pron_classifier(feat)
        elif head == "cosine_linear":
            cos = torch.sum(u * v, dim=-1, keepdim=True)  # [B,1]
            logits = self.pron_classifier(cos)
        else:
            raise ValueError(f"Unknown pair_head: {self.pair_head}")

        # Loss
        if self.lambda_pron <= 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            if self.focal_gamma and self.focal_gamma > 0:
                loss = self._focal_loss(
                    logits=logits.float(),
                    targets=labels,
                    weight=self._class_weights_tensor,
                    gamma=self.focal_gamma,
                )
            else:
                loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    reduction="mean",
                    weight=self._class_weights_tensor,
                )
            loss = loss * float(self.lambda_pron)

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == labels).sum().float()
            total = torch.tensor(float(B), device=device)
            acc = correct / total.clamp(min=1.0)

            cos_all = torch.sum(u * v, dim=-1)  # [B]
            cos_mean = cos_all.mean()

        logging_output = {
            "loss": loss.data,
            "sample_size": B,
            "nsentences": B,
            "ntokens": B,
            "pron_correct": correct.data,
            "pron_total": total.data,
            "pron_acc": acc.data,
            "cos_mean": cos_mean.data,
        }

        # Per-label stats: log using *original* label numbers (1,2,3) for clarity
        if self.log_cosine_by_label:
            for internal_lbl in range(self.num_classes):
                orig_lbl = internal_lbl + 1  # map back: 0->1, 1->2, 2->3
                m = labels == internal_lbl
                if m.any():
                    logging_output[f"cos_l{orig_lbl}"] = cos_all[m].mean().data
                    logging_output[f"n_l{orig_lbl}"] = m.sum().data

        return loss, B, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        # accuracy
        correct_sum = sum(log.get("pron_correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("pron_total", 0) for log in logging_outputs)
        if total_sum > 0:
            metrics.log_scalar("pron_acc", correct_sum / total_sum, total_sum, round=4)

        # cosine stats
        cos_vals = [log.get("cos_mean", None) for log in logging_outputs if "cos_mean" in log]
        if cos_vals:
            metrics.log_scalar("cos_mean", sum(cos_vals) / len(cos_vals), len(cos_vals), round=4)

        # per-label cosine means if present (labels 1,2,3)
        for lbl in (1, 2, 3):
            k = f"cos_l{lbl}"
            vals = [log.get(k, None) for log in logging_outputs if k in log]
            if vals:
                metrics.log_scalar(k, sum(vals) / len(vals), len(vals), round=4)

        for lbl in (1, 2, 3):
            k = f"n_l{lbl}"
            vals = [log.get(k, None) for log in logging_outputs if k in log]
            if vals:
                # counts: treat as sums (use meters[k].sum downstream if needed)
                metrics.log_scalar(k, sum(vals), 1, round=0)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.pron_classifier is not None:
            state["pron_classifier"] = self.pron_classifier.state_dict()
            state["pair_head"] = self.pair_head
            state["num_classes"] = self.num_classes
        return state

    def load_state_dict(self, state_dict, strict=True):
        # If checkpoint contains pron_classifier weights, load them into the
        # already-created classifier (eager init).
        if "pron_classifier" in state_dict and state_dict["pron_classifier"] is not None:
            try:
                self.pron_classifier.load_state_dict(state_dict["pron_classifier"])
                print("[PronunciationPairCriterion] Loaded pron_classifier from checkpoint")
            except Exception as e:
                print(f"[PronunciationPairCriterion] Warning: failed to load pron_classifier: {e}")

        # remove extra keys and delegate
        filtered = {k: v for k, v in state_dict.items() if k not in {"pron_classifier", "pair_head", "num_classes"}}
        super().load_state_dict(filtered, strict=False)

