"""
LASR baseline: hard-triplet supervision using labels (gender + tone) plus optional ASR CTC.

This criterion is designed to be a rebuttal baseline inspired by:
  "Label Aware Speech Representation Learning (LASR)" (Vashishth et al., 2023)
  https://arxiv.org/pdf/2306.04374

It reuses the existing joint sample structure produced by `gender_tone_repulsive_task`:
  - Gender stream: female_audio, male_positive_audio (+ padding masks)
  - Tone stream:   anchor_audio, positive_audio, anchor_tone_label (+ masks)
  - ASR stream:    asr_audio, asr_target, asr_target_lengths (+ padding mask)

Loss:
  L = α * L_lasr_gender + β * L_lasr_tone + δ * L_ctc

Notes:
  - Gender LASR uses a batch-level hard-triplet objective on {female, male} utterances.
  - Tone LASR can optionally include `positive_audio` augmentations as extra instances
    so that positives exist even when batch size is small.
  - This is intended as a baseline; it does not aim to reproduce every LASR detail.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics


def _hard_triplet_loss_from_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Hard triplet loss (batch-hard) in cosine-similarity space.

    For each anchor i:
      pos = farthest positive   => min cosine among same-label (excluding self)
      neg = closest negative    => max cosine among different-label
      loss_i = relu(margin + neg_sim - pos_sim)

    Args:
      embeddings: [N, C]
      labels:     [N] int64
      margin:     similarity margin (e.g., 0.1~0.3)
    """
    device = embeddings.device
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be [N,C], got {tuple(embeddings.shape)}")
    if labels.dim() != 1 or labels.size(0) != embeddings.size(0):
        raise ValueError(
            f"labels must be [N] matching embeddings, got labels={tuple(labels.shape)} emb={tuple(embeddings.shape)}"
        )

    n = embeddings.size(0)
    if n <= 1:
        z = embeddings.new_tensor(0.0, requires_grad=True)
        return z, {
            "triplet_n": torch.tensor(0, device=device),
            "triplet_valid": torch.tensor(0, device=device),
            "triplet_active_frac": torch.tensor(0.0, device=device),
            "hard_pos_sim": torch.tensor(float("nan"), device=device),
            "hard_neg_sim": torch.tensor(float("nan"), device=device),
        }

    emb = F.normalize(embeddings, dim=-1)
    # [N,N] cosine similarity
    sim = emb @ emb.transpose(0, 1)

    y = labels.view(-1, 1)
    same = (y == y.transpose(0, 1))
    eye = torch.eye(n, dtype=torch.bool, device=device)
    same_no_self = same & (~eye)
    diff = ~same  # self is False automatically because same[i,i]=True

    has_pos = same_no_self.any(dim=1)
    has_neg = diff.any(dim=1)
    valid = has_pos & has_neg
    valid_n = int(valid.sum().item())

    if valid_n == 0:
        z = embeddings.new_tensor(0.0, requires_grad=True)
        return z, {
            "triplet_n": torch.tensor(n, device=device),
            "triplet_valid": torch.tensor(0, device=device),
            "triplet_active_frac": torch.tensor(0.0, device=device),
            "hard_pos_sim": torch.tensor(float("nan"), device=device),
            "hard_neg_sim": torch.tensor(float("nan"), device=device),
        }

    # Hardest positive: minimum similarity among same-label
    pos_sim = sim.masked_fill(~same_no_self, float("inf"))
    hard_pos_sim = pos_sim.min(dim=1).values

    # Hardest negative: maximum similarity among different-label
    neg_sim = sim.masked_fill(~diff, float("-inf"))
    hard_neg_sim = neg_sim.max(dim=1).values

    loss_vec = F.relu(float(margin) + hard_neg_sim - hard_pos_sim)
    loss = loss_vec[valid].mean()

    with torch.no_grad():
        active_frac = (loss_vec[valid] > 0).float().mean()
        stats = {
            "triplet_n": torch.tensor(n, device=device),
            "triplet_valid": torch.tensor(valid_n, device=device),
            "triplet_active_frac": active_frac,
            "hard_pos_sim": hard_pos_sim[valid].mean(),
            "hard_neg_sim": hard_neg_sim[valid].mean(),
        }
    return loss, stats


@dataclass
class GenderToneLasrCriterionConfig(FairseqDataclass):
    """Standalone config — duplicates parent YAML keys so Hydra struct mode is happy."""

    # ---- stream weights ----
    alpha: float = field(default=1.0, metadata={"help": "weight for gender LASR loss"})
    beta: float = field(default=1.0, metadata={"help": "weight for tone LASR loss"})
    delta: float = field(default=0.0, metadata={"help": "weight for ASR CTC loss (0 disables)"})

    # ---- gender (unused, kept for YAML compat) ----
    temperature_gender: float = field(default=0.07, metadata={"help": "(unused) kept for YAML compat"})

    # ---- tone keys from base YAML (all unused, kept for YAML compat) ----
    use_tone_hinge_loss: bool = field(default=False, metadata={"help": "(unused)"})
    tone_lambda_contrastive: float = field(default=1.0, metadata={"help": "(unused)"})
    tone_lambda_attract: float = field(default=1.0, metadata={"help": "(unused)"})
    tone_lambda_repulse_hard: float = field(default=1.5, metadata={"help": "(unused)"})
    tone_lambda_repulse_soft: float = field(default=0.3, metadata={"help": "(unused)"})
    tone_margin_hard: float = field(default=-0.3, metadata={"help": "(unused)"})
    tone_margin_soft: float = field(default=-0.2, metadata={"help": "(unused)"})
    tone_lambda_classification: float = field(default=0.0, metadata={"help": "(unused)"})
    tone_temperature: float = field(default=0.07, metadata={"help": "(unused)"})
    tone_num_classes: int = field(default=7, metadata={"help": "(unused)"})
    tone_freeze_classifier: bool = field(default=False, metadata={"help": "(unused)"})
    tone_classifier_ckpt: str = field(default="", metadata={"help": "(unused)"})

    # ---- ASR CTC head ----
    asr_ctc_head_ckpt: str = field(default="", metadata={"help": "path to finetuned ASR checkpoint to load CTC head from"})
    asr_ctc_head_freeze: bool = field(default=True, metadata={"help": "freeze loaded CTC head weights"})

    # ---- KD keys from base YAML (unused, kept for YAML compat) ----
    kd_teacher_ckpt: str = field(default="", metadata={"help": "(unused)"})
    kd_weight: float = field(default=0.0, metadata={"help": "(unused)"})
    kd_temperature: float = field(default=2.0, metadata={"help": "(unused)"})

    # ---- shared embedding / freezing ----
    pooling_mode: str = field(default="mean", metadata={"help": "pooling over time: mean, max, or last"})
    embedding_layer: int = field(default=-1, metadata={"help": "(unused) kept for YAML compat"})
    embedding_extraction_layer: int = field(default=-1, metadata={"help": "layer index for embedding extraction (-1=last)"})
    freeze_encoder_layers: int = field(default=0, metadata={"help": "freeze first N encoder layers"})
    freeze_entire_encoder: bool = field(default=False, metadata={"help": "freeze ALL encoder components"})
    freeze_feature_extractor: bool = field(default=False, metadata={"help": "also freeze feature_extractor + post_extract_proj"})

    # ---- speaker-adversarial keys from base YAML (unused, kept for YAML compat) ----
    speaker_adv_weight: float = field(default=0.0, metadata={"help": "(unused)"})
    speaker_grl_lambda: float = field(default=1.0, metadata={"help": "(unused)"})
    speaker_adv_hidden_dim: int = field(default=256, metadata={"help": "(unused)"})
    speaker_adv_num_speakers: int = field(default=0, metadata={"help": "(unused)"})
    speaker_adv_emb_dim: int = field(default=768, metadata={"help": "(unused)"})

    # ---- LASR-specific knobs ----
    lasr_margin_gender: float = field(default=0.2, metadata={"help": "hard-triplet margin (cosine-sim space) for gender LASR"})
    lasr_margin_tone: float = field(default=0.2, metadata={"help": "hard-triplet margin (cosine-sim space) for tone LASR"})
    tone_use_positives: bool = field(default=True, metadata={"help": "if true, include positive_audio as extra instances for tone LASR"})


@register_criterion("gender_tone_lasr", dataclass=GenderToneLasrCriterionConfig)
class GenderToneLasrCriterion(FairseqCriterion):
    """
    LASR baseline criterion: gender + tone hard-triplet loss + optional ASR CTC.

    Expects the same `sample['net_input']` keys as `GenderToneRepulsiveTask`.
    """

    def __init__(
        self,
        task,
        # ---- used ----
        alpha: float = 1.0,
        beta: float = 1.0,
        delta: float = 0.0,
        pooling_mode: str = "mean",
        embedding_layer: int = -1,
        embedding_extraction_layer: int = -1,
        freeze_encoder_layers: int = 0,
        freeze_entire_encoder: bool = False,
        freeze_feature_extractor: bool = False,
        asr_ctc_head_ckpt: str = "",
        asr_ctc_head_freeze: bool = True,
        lasr_margin_gender: float = 0.2,
        lasr_margin_tone: float = 0.2,
        tone_use_positives: bool = True,
        # ---- unused (absorbed from YAML so fairseq doesn't complain) ----
        temperature_gender: float = 0.07,
        use_tone_hinge_loss: bool = False,
        tone_lambda_contrastive: float = 1.0,
        tone_lambda_attract: float = 1.0,
        tone_lambda_repulse_hard: float = 1.5,
        tone_lambda_repulse_soft: float = 0.3,
        tone_margin_hard: float = -0.3,
        tone_margin_soft: float = -0.2,
        tone_lambda_classification: float = 0.0,
        tone_temperature: float = 0.07,
        tone_num_classes: int = 7,
        tone_freeze_classifier: bool = False,
        tone_classifier_ckpt: str = "",
        kd_teacher_ckpt: str = "",
        kd_weight: float = 0.0,
        kd_temperature: float = 2.0,
        speaker_adv_weight: float = 0.0,
        speaker_grl_lambda: float = 1.0,
        speaker_adv_hidden_dim: int = 256,
        speaker_adv_num_speakers: int = 0,
        speaker_adv_emb_dim: int = 768,
    ):
        super().__init__(task)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)

        self.pooling_mode = str(pooling_mode)
        self.embedding_extraction_layer = int(embedding_extraction_layer)

        self.freeze_encoder_layers = int(freeze_encoder_layers)
        self.freeze_entire_encoder = bool(freeze_entire_encoder)
        self.freeze_feature_extractor = bool(freeze_feature_extractor)
        self._layers_frozen = False

        # ASR head loading/freezing (optional)
        self.asr_ctc_head_ckpt = str(asr_ctc_head_ckpt or "")
        self.asr_ctc_head_freeze = bool(asr_ctc_head_freeze)
        self._head_installed = False
        self._ctc_head_frozen = False

        # LASR knobs
        self.lasr_margin_gender = float(lasr_margin_gender)
        self.lasr_margin_tone = float(lasr_margin_tone)
        self.tone_use_positives = bool(tone_use_positives)

        # Print a concise init summary (helpful for Colab logs)
        print("[GenderToneLasrCriterion] Initialized:")
        print(f"  α (gender LASR): {self.alpha}")
        print(f"  β (tone LASR):   {self.beta}")
        print(f"  δ (ASR CTC):     {self.delta}")
        print(f"  margins: gender={self.lasr_margin_gender}, tone={self.lasr_margin_tone}")
        print(f"  tone_use_positives={self.tone_use_positives}")
        if self.embedding_extraction_layer >= 0:
            print(f"  embeddings: layer={self.embedding_extraction_layer} pooling={self.pooling_mode}")
        else:
            print(f"  embeddings: full encoder output pooling={self.pooling_mode}")

    # -------------------------
    # Embedding extraction
    # -------------------------
    def extract_embedding(self, model, audio_tensor, padding_mask=None):
        """Extract and pool embeddings from encoder (optionally from a specific layer)."""
        if hasattr(model, "w2v_encoder"):
            enc = model.w2v_encoder(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
                layer=self.embedding_extraction_layer if self.embedding_extraction_layer >= 0 else None,
            )
            # encoder_out: [T, B, C] -> [B, T, C]
            feats = enc["encoder_out"].transpose(0, 1)
            pm = enc.get("padding_mask", padding_mask)
        elif hasattr(model, "extract_features"):
            # Fallback for other wav2vec wrappers
            res = model.extract_features(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                layer=self.embedding_extraction_layer if self.embedding_extraction_layer >= 0 else None,
            )
            feats = res["x"]  # [B, T, C]
            pm = res.get("padding_mask", padding_mask)
        else:
            raise ValueError("Model must have w2v_encoder or extract_features.")

        # Pool over time
        if pm is not None:
            feats = feats.masked_fill(pm.unsqueeze(-1), 0.0)
            lengths = (~pm).sum(dim=1).float().clamp(min=1.0)
        else:
            lengths = torch.full(
                (feats.size(0),),
                feats.size(1),
                dtype=torch.float32,
                device=feats.device,
            )

        if self.pooling_mode == "mean":
            pooled = feats.sum(dim=1) / lengths.unsqueeze(-1)
        elif self.pooling_mode == "max":
            pooled = feats.max(dim=1)[0]
        elif self.pooling_mode == "last":
            idx = (lengths - 1).long().clamp(min=0)
            pooled = feats[torch.arange(feats.size(0), device=feats.device), idx]
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        return pooled

    # -------------------------
    # Freezing utilities
    # -------------------------
    def _get_inner_w2v_model(self, model):
        if not (hasattr(model, "w2v_encoder") and hasattr(model.w2v_encoder, "w2v_model")):
            return None
        w2v_model = model.w2v_encoder.w2v_model
        # Handle nested Wav2VecCtc (common when loading via w2v_path)
        if hasattr(w2v_model, "w2v_encoder") and hasattr(w2v_model.w2v_encoder, "w2v_model"):
            w2v_model = w2v_model.w2v_encoder.w2v_model
        return w2v_model

    def _freeze_entire_encoder(self, model):
        w2v_model = self._get_inner_w2v_model(model)
        if w2v_model is None:
            return

        frozen = []
        # Freeze all params inside w2v_model
        for p in w2v_model.parameters():
            p.requires_grad = False
        frozen.append("w2v_model(*)")

        # Optionally be explicit about common components (for readability in logs)
        if hasattr(w2v_model, "feature_extractor"):
            frozen.append("feature_extractor")
        if hasattr(w2v_model, "post_extract_proj") and w2v_model.post_extract_proj is not None:
            frozen.append("post_extract_proj")
        if hasattr(w2v_model, "encoder"):
            if hasattr(w2v_model.encoder, "pos_conv") and w2v_model.encoder.pos_conv is not None:
                frozen.append("encoder.pos_conv")
            if hasattr(w2v_model.encoder, "layers"):
                frozen.append(f"encoder.layers[0:{len(w2v_model.encoder.layers)}]")
            if hasattr(w2v_model.encoder, "layer_norm") and w2v_model.encoder.layer_norm is not None:
                frozen.append("encoder.layer_norm")

        print(f"[GenderToneLasrCriterion] ✅ Frozen entire encoder: {', '.join(frozen)}")
        print("[GenderToneLasrCriterion] 🔓 w2v_encoder.proj (CTC head) remains controlled by asr_ctc_head_freeze")

    def _freeze_first_n_layers(self, model, n_layers: int):
        w2v_model = self._get_inner_w2v_model(model)
        if w2v_model is None:
            return
        if not (hasattr(w2v_model, "encoder") and hasattr(w2v_model.encoder, "layers")):
            return
        enc = w2v_model.encoder
        total = len(enc.layers)
        k = max(0, min(int(n_layers), total))
        if k == 0:
            return

        if self.freeze_feature_extractor:
            if hasattr(w2v_model, "feature_extractor"):
                for p in w2v_model.feature_extractor.parameters():
                    p.requires_grad = False
            if hasattr(w2v_model, "post_extract_proj") and w2v_model.post_extract_proj is not None:
                for p in w2v_model.post_extract_proj.parameters():
                    p.requires_grad = False
            if hasattr(w2v_model, "layer_norm") and w2v_model.layer_norm is not None:
                for p in w2v_model.layer_norm.parameters():
                    p.requires_grad = False
            if hasattr(enc, "pos_conv") and enc.pos_conv is not None:
                for p in enc.pos_conv.parameters():
                    p.requires_grad = False

        for i in range(k):
            for p in enc.layers[i].parameters():
                p.requires_grad = False

        print(f"[GenderToneLasrCriterion] 🔒 Frozen first {k}/{total} encoder layers (freeze_feature_extractor={self.freeze_feature_extractor})")

    def _maybe_freeze_encoder_layers(self, model):
        if self._layers_frozen:
            return
        if self.freeze_entire_encoder:
            self._freeze_entire_encoder(model)
            self._layers_frozen = True
            return
        if self.freeze_encoder_layers > 0:
            self._freeze_first_n_layers(model, self.freeze_encoder_layers)
            self._layers_frozen = True
            return
        # No freezing
        self._layers_frozen = True

    # -------------------------
    # ASR CTC head utilities (optional)
    # -------------------------
    def _maybe_install_ctc_head(self, model):
        """Load ASR CTC head weights from a checkpoint (optional)."""
        if self._head_installed or not self.asr_ctc_head_ckpt:
            return
        import os

        if not os.path.exists(self.asr_ctc_head_ckpt):
            print(f"[GenderToneLasrCriterion] Warning: ASR head ckpt not found: {self.asr_ctc_head_ckpt}")
            self._head_installed = True
            return

        try:
            print(f"[GenderToneLasrCriterion] Loading ASR CTC head from {self.asr_ctc_head_ckpt}")
            state = torch.load(self.asr_ctc_head_ckpt, map_location="cpu", weights_only=False)
            model_state = state["model"] if isinstance(state, dict) and "model" in state else state
            ctc_keys = [k for k in model_state.keys() if "w2v_encoder.proj" in k]
            if not ctc_keys:
                print("[GenderToneLasrCriterion] Warning: no w2v_encoder.proj keys found in ckpt")
                self._head_installed = True
                return
            ctc_state = {k: model_state[k] for k in ctc_keys}
            model.load_state_dict(ctc_state, strict=False)
            print(f"[GenderToneLasrCriterion] Loaded {len(ctc_keys)} CTC head parameters")
        except Exception as e:
            print(f"[GenderToneLasrCriterion] Error loading ASR head: {e}")
        finally:
            self._head_installed = True

    def _maybe_freeze_ctc_head(self, model):
        if self._ctc_head_frozen or (not self.asr_ctc_head_freeze):
            return
        if hasattr(model, "w2v_encoder") and hasattr(model.w2v_encoder, "proj") and model.w2v_encoder.proj is not None:
            for p in model.w2v_encoder.proj.parameters():
                p.requires_grad = False
            self._ctc_head_frozen = True
            print("[GenderToneLasrCriterion] ✓ Frozen ASR CTC head (w2v_encoder.proj)")

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, model, sample, reduce=True):
        """
        sample['net_input'] keys expected (subset is fine):
          - female_audio, male_positive_audio (+ padding masks)
          - anchor_audio, positive_audio, anchor_tone_label (+ padding masks)
          - asr_audio, asr_target, asr_target_lengths (+ asr_padding_mask)
        """
        self._maybe_freeze_encoder_layers(model)

        net_input = sample["net_input"]
        device = next(model.parameters()).device

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        sample_size = 0
        logs: Dict[str, torch.Tensor] = {}

        # ===== Gender LASR =====
        if self.alpha > 0 and ("female_audio" in net_input) and ("male_positive_audio" in net_input):
            female_emb = self.extract_embedding(
                model, net_input["female_audio"], net_input.get("female_padding_mask")
            )
            male_emb = self.extract_embedding(
                model, net_input["male_positive_audio"], net_input.get("male_positive_padding_mask")
            )
            b = female_emb.size(0)
            emb = torch.cat([female_emb, male_emb], dim=0)  # [2B, C]
            y = torch.cat(
                [
                    torch.zeros(b, dtype=torch.long, device=emb.device),
                    torch.ones(b, dtype=torch.long, device=emb.device),
                ],
                dim=0,
            )

            loss_g, st_g = _hard_triplet_loss_from_embeddings(
                emb, y, margin=self.lasr_margin_gender
            )
            total_loss = total_loss + self.alpha * loss_g
            sample_size += int(emb.size(0))

            logs["loss_gender_lasr"] = loss_g.detach()
            logs["gender_triplet_valid"] = st_g["triplet_valid"]
            logs["gender_triplet_active_frac"] = st_g["triplet_active_frac"]
            logs["gender_hard_pos_sim"] = st_g["hard_pos_sim"]
            logs["gender_hard_neg_sim"] = st_g["hard_neg_sim"]

        # ===== Tone LASR =====
        if self.beta > 0 and ("anchor_audio" in net_input) and ("anchor_tone_label" in net_input):
            anchor_emb = self.extract_embedding(
                model, net_input["anchor_audio"], net_input.get("anchor_padding_mask")
            )  # [B,C]
            tone_y = net_input["anchor_tone_label"].to(device=anchor_emb.device).long()  # [B]

            tone_emb = anchor_emb
            tone_labels = tone_y

            # Optionally add positive_audio augmentations as extra instances to guarantee positives.
            if self.tone_use_positives and ("positive_audio" in net_input):
                pos_audio = net_input["positive_audio"]  # [B,P,T]
                if pos_audio.dim() == 3 and pos_audio.size(0) == anchor_emb.size(0):
                    b, p, t = pos_audio.shape
                    pos_flat = pos_audio.view(b * p, t)
                    pos_pm = net_input.get("positive_padding_mask")
                    pos_pm_flat = pos_pm.view(b * p, t) if pos_pm is not None else None
                    pos_emb_flat = self.extract_embedding(model, pos_flat, pos_pm_flat)  # [B*P, C]
                    tone_emb = torch.cat([anchor_emb, pos_emb_flat], dim=0)
                    tone_labels = torch.cat(
                        [tone_y, tone_y.view(b, 1).expand(b, p).reshape(-1)], dim=0
                    )

            loss_t, st_t = _hard_triplet_loss_from_embeddings(
                tone_emb, tone_labels, margin=self.lasr_margin_tone
            )
            total_loss = total_loss + self.beta * loss_t
            sample_size += int(tone_emb.size(0))

            logs["loss_tone_lasr"] = loss_t.detach()
            logs["tone_triplet_valid"] = st_t["triplet_valid"]
            logs["tone_triplet_active_frac"] = st_t["triplet_active_frac"]
            logs["tone_hard_pos_sim"] = st_t["hard_pos_sim"]
            logs["tone_hard_neg_sim"] = st_t["hard_neg_sim"]

        # ===== ASR CTC =====
        if self.delta > 0 and ("asr_audio" in net_input):
            self._maybe_install_ctc_head(model)
            self._maybe_freeze_ctc_head(model)

            net_out = model(
                source=net_input["asr_audio"],
                padding_mask=net_input.get("asr_padding_mask"),
                mask=False,
            )
            logits = model.get_logits(net_out, normalize=False)  # [T,B,C]
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            if net_out.get("padding_mask") is not None:
                nonpad = (~net_out["padding_mask"]).long()
                input_lengths = nonpad.sum(-1)  # [B]
            else:
                input_lengths = log_probs.new_full(
                    (log_probs.size(1),), log_probs.size(0), dtype=torch.long
                )

            targets = net_input["asr_target"]
            target_lengths = net_input["asr_target_lengths"]

            ctc_loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
            total_loss = total_loss + self.delta * ctc_loss
            sample_size += int(targets.size(0))
            logs["loss_ctc"] = ctc_loss.detach()

        if sample_size == 0:
            dummy = torch.tensor(0.0, requires_grad=True, device=device)
            return dummy, 0, {"loss": dummy.detach(), "sample_size": 0}

        logging_output = {"loss": total_loss.detach(), "sample_size": sample_size}
        logging_output.update(logs)
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        for key in [
            "loss_gender_lasr",
            "loss_tone_lasr",
            "loss_ctc",
            "gender_triplet_valid",
            "gender_triplet_active_frac",
            "gender_hard_pos_sim",
            "gender_hard_neg_sim",
            "tone_triplet_valid",
            "tone_triplet_active_frac",
            "tone_hard_pos_sim",
            "tone_hard_neg_sim",
        ]:
            vals = [log.get(key, None) for log in logging_outputs if key in log]
            if vals:
                # Some are integer tensors; log_scalar handles it.
                metrics.log_scalar(key, sum(vals) / len(vals), len(vals), round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

