"""
UniSpeech-style baseline (adapted to this project):

UniSpeech (Wang et al., 2021) combines supervised CTC with a self-supervised
contrastive objective in a multitask manner:
  https://arxiv.org/pdf/2101.07597

In our project we already have two contrastive/self objectives (gender + tone)
and an optional ASR CTC stream. This baseline simply trains them jointly:

  L = α * L_gender_InfoNCE + β * L_tone_InfoNCE + δ * L_CTC

Important:
  - This is an adaptation for rebuttal baselines. It does NOT re-implement the
    full wav2vec2 pretraining objective with quantizer/ masking/replacement r.
  - It is designed to reuse the SAME joint sample structure produced by
    `gender_tone_repulsive_task` without changing any existing code.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics


@dataclass
class GenderToneUniSpeechCriterionConfig(FairseqDataclass):
    """Standalone config — duplicates parent YAML keys so Hydra struct mode is happy."""

    # ---- stream weights ----
    alpha: float = field(default=1.0, metadata={"help": "weight for gender InfoNCE loss"})
    beta: float = field(default=1.0, metadata={"help": "weight for tone InfoNCE loss"})
    delta: float = field(default=0.0, metadata={"help": "weight for ASR CTC loss (0 disables)"})

    # ---- gender ----
    temperature_gender: float = field(default=0.07, metadata={"help": "temperature for gender InfoNCE"})

    # ---- tone keys from base YAML (most unused, kept for YAML compat) ----
    use_tone_hinge_loss: bool = field(default=False, metadata={"help": "(unused)"})
    tone_lambda_contrastive: float = field(default=1.0, metadata={"help": "weight for tone contrastive loss"})
    tone_lambda_attract: float = field(default=1.0, metadata={"help": "(unused)"})
    tone_lambda_repulse_hard: float = field(default=1.5, metadata={"help": "(unused)"})
    tone_lambda_repulse_soft: float = field(default=0.3, metadata={"help": "(unused)"})
    tone_margin_hard: float = field(default=-0.3, metadata={"help": "(unused)"})
    tone_margin_soft: float = field(default=-0.2, metadata={"help": "(unused)"})
    tone_lambda_classification: float = field(default=0.0, metadata={"help": "(unused) set 0 for UniSpeech baseline"})
    tone_temperature: float = field(default=0.07, metadata={"help": "temperature for tone InfoNCE"})
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

    # ---- UniSpeech-specific ----
    tone_use_soft_negatives: bool = field(
        default=True,
        metadata={"help": "if true, include soft negatives in tone InfoNCE (in addition to hard negatives)"},
    )


@register_criterion("gender_tone_unispeech", dataclass=GenderToneUniSpeechCriterionConfig)
class GenderToneUniSpeechCriterion(FairseqCriterion):
    """
    UniSpeech-style multitask baseline:
      - Gender InfoNCE (female vs male positives + male negatives)
      - Tone InfoNCE (anchor vs same-tone positives + (hard [+soft]) negatives)
      - Optional ASR CTC loss
    """

    def __init__(
        self,
        task,
        # ---- used ----
        alpha: float = 1.0,
        beta: float = 1.0,
        delta: float = 0.0,
        temperature_gender: float = 0.07,
        tone_temperature: float = 0.07,
        tone_lambda_contrastive: float = 1.0,
        tone_use_soft_negatives: bool = True,
        pooling_mode: str = "mean",
        embedding_layer: int = -1,
        embedding_extraction_layer: int = -1,
        freeze_encoder_layers: int = 0,
        freeze_entire_encoder: bool = False,
        freeze_feature_extractor: bool = False,
        asr_ctc_head_ckpt: str = "",
        asr_ctc_head_freeze: bool = True,
        # ---- unused (absorbed from YAML so fairseq doesn't complain) ----
        use_tone_hinge_loss: bool = False,
        tone_lambda_attract: float = 1.0,
        tone_lambda_repulse_hard: float = 1.5,
        tone_lambda_repulse_soft: float = 0.3,
        tone_margin_hard: float = -0.3,
        tone_margin_soft: float = -0.2,
        tone_lambda_classification: float = 0.0,
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

        self.temperature_gender = float(temperature_gender)
        self.tone_temperature = float(tone_temperature)
        self.tone_lambda_contrastive = float(tone_lambda_contrastive)
        self.tone_use_soft_negatives = bool(tone_use_soft_negatives)

        self.pooling_mode = str(pooling_mode)
        self.embedding_extraction_layer = int(embedding_extraction_layer)

        self.freeze_encoder_layers = int(freeze_encoder_layers)
        self.freeze_entire_encoder = bool(freeze_entire_encoder)
        self.freeze_feature_extractor = bool(freeze_feature_extractor)
        self._layers_frozen = False

        self.asr_ctc_head_ckpt = str(asr_ctc_head_ckpt or "")
        self.asr_ctc_head_freeze = bool(asr_ctc_head_freeze)
        self._head_installed = False
        self._ctc_head_frozen = False

        print("[GenderToneUniSpeechCriterion] Initialized:")
        print(f"  α (gender InfoNCE): {self.alpha}")
        print(f"  β (tone InfoNCE):   {self.beta}  (λ_contrastive={self.tone_lambda_contrastive})")
        print(f"  δ (ASR CTC):        {self.delta}")
        print(f"  temps: gender={self.temperature_gender}, tone={self.tone_temperature}")
        print(f"  tone_use_soft_negatives={self.tone_use_soft_negatives}")
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
            feats = enc["encoder_out"].transpose(0, 1)  # [B,T,C]
            pm = enc.get("padding_mask", padding_mask)
        elif hasattr(model, "extract_features"):
            res = model.extract_features(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                layer=self.embedding_extraction_layer if self.embedding_extraction_layer >= 0 else None,
            )
            feats = res["x"]
            pm = res.get("padding_mask", padding_mask)
        else:
            raise ValueError("Model must have w2v_encoder or extract_features.")

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
        if hasattr(w2v_model, "w2v_encoder") and hasattr(w2v_model.w2v_encoder, "w2v_model"):
            w2v_model = w2v_model.w2v_encoder.w2v_model
        return w2v_model

    def _freeze_entire_encoder(self, model):
        w2v_model = self._get_inner_w2v_model(model)
        if w2v_model is None:
            return
        for p in w2v_model.parameters():
            p.requires_grad = False
        print("[GenderToneUniSpeechCriterion] ✅ Frozen entire encoder (w2v_model.*)")

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
            if hasattr(enc, "pos_conv") and enc.pos_conv is not None:
                for p in enc.pos_conv.parameters():
                    p.requires_grad = False

        for i in range(k):
            for p in enc.layers[i].parameters():
                p.requires_grad = False

        print(
            f"[GenderToneUniSpeechCriterion] 🔒 Frozen first {k}/{total} encoder layers "
            f"(freeze_feature_extractor={self.freeze_feature_extractor})"
        )

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
        self._layers_frozen = True

    # -------------------------
    # ASR head utilities (optional)
    # -------------------------
    def _maybe_install_ctc_head(self, model):
        if self._head_installed or not self.asr_ctc_head_ckpt:
            return
        import os

        if not os.path.exists(self.asr_ctc_head_ckpt):
            print(f"[GenderToneUniSpeechCriterion] Warning: ASR head ckpt not found: {self.asr_ctc_head_ckpt}")
            self._head_installed = True
            return

        try:
            print(f"[GenderToneUniSpeechCriterion] Loading ASR CTC head from {self.asr_ctc_head_ckpt}")
            state = torch.load(self.asr_ctc_head_ckpt, map_location="cpu", weights_only=False)
            model_state = state["model"] if isinstance(state, dict) and "model" in state else state
            keys = [k for k in model_state.keys() if "w2v_encoder.proj" in k]
            if keys:
                model.load_state_dict({k: model_state[k] for k in keys}, strict=False)
                print(f"[GenderToneUniSpeechCriterion] Loaded {len(keys)} CTC head parameters")
            else:
                print("[GenderToneUniSpeechCriterion] Warning: no w2v_encoder.proj keys found in ckpt")
        except Exception as e:
            print(f"[GenderToneUniSpeechCriterion] Error loading ASR head: {e}")
        finally:
            self._head_installed = True

    def _maybe_freeze_ctc_head(self, model):
        if self._ctc_head_frozen or (not self.asr_ctc_head_freeze):
            return
        if hasattr(model, "w2v_encoder") and hasattr(model.w2v_encoder, "proj") and model.w2v_encoder.proj is not None:
            for p in model.w2v_encoder.proj.parameters():
                p.requires_grad = False
            self._ctc_head_frozen = True
            print("[GenderToneUniSpeechCriterion] ✓ Frozen ASR CTC head (w2v_encoder.proj)")

    # -------------------------
    # Losses
    # -------------------------
    def _gender_infonce(self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gender InfoNCE (same as your existing gender stream):
          - anchor:   [B,C] (female)
          - positive: [B,C] (male same word)
          - negatives:[B,N,C] (male different words)
        """
        b = anchor.size(0)
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = (anchor * positive).sum(dim=-1)  # [B]
        pos_logits = pos_sim / self.temperature_gender

        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)  # [B,N]
        neg_logits = neg_sim / self.temperature_gender

        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # [B,1+N]
        y = torch.zeros(b, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, y, reduction="mean")
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == y).float().mean()
        return loss, acc

    def _tone_infonce(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        hard_negs: torch.Tensor,
        soft_negs: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tone InfoNCE (multi-positive):
          - anchor:    [B,C]
          - positives: [B,P,C]
          - hard_negs: [B,H,C]
          - soft_negs: [B,S,C] or None
        """
        b = anchor.size(0)
        anchor = F.normalize(anchor, dim=-1)
        positives = F.normalize(positives, dim=-1)
        hard_negs = F.normalize(hard_negs, dim=-1)

        negs = hard_negs
        if self.tone_use_soft_negatives and soft_negs is not None:
            soft_negs = F.normalize(soft_negs, dim=-1)
            negs = torch.cat([hard_negs, soft_negs], dim=1)  # [B,H+S,C]

        # mean positive similarity
        pos_sim = (anchor.unsqueeze(1) * positives).sum(dim=-1)  # [B,P]
        mean_pos = pos_sim.mean(dim=1)  # [B]
        pos_logit = mean_pos / self.tone_temperature

        neg_sim = torch.bmm(anchor.unsqueeze(1), negs.transpose(1, 2)).squeeze(1)  # [B,N]
        neg_logits = neg_sim / self.tone_temperature

        logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
        y = torch.zeros(b, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, y, reduction="mean")

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == y).float().mean()
            avg_pos = mean_pos.mean()
            avg_neg = neg_sim.mean()
        return loss, acc, avg_pos, avg_neg

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, model, sample, reduce=True):
        self._maybe_freeze_encoder_layers(model)
        net_input = sample["net_input"]
        device = next(model.parameters()).device

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        sample_size = 0
        logs: Dict[str, torch.Tensor] = {}

        # ===== Gender InfoNCE =====
        if self.alpha > 0 and ("female_audio" in net_input) and ("male_positive_audio" in net_input):
            female_emb = self.extract_embedding(model, net_input["female_audio"], net_input.get("female_padding_mask"))
            male_pos_emb = self.extract_embedding(
                model, net_input["male_positive_audio"], net_input.get("male_positive_padding_mask")
            )
            neg_audio = net_input.get("male_negative_audio", None)
            if neg_audio is not None and neg_audio.dim() == 3:
                b, n, t = neg_audio.shape
                neg_flat = neg_audio.view(b * n, t)
                neg_pm = net_input.get("male_negative_padding_mask")
                neg_pm_flat = neg_pm.view(b * n, t) if neg_pm is not None else None
                neg_emb_flat = self.extract_embedding(model, neg_flat, neg_pm_flat)  # [B*N,C]
                neg_emb = neg_emb_flat.view(b, n, -1)

                loss_g, acc_g = self._gender_infonce(female_emb, male_pos_emb, neg_emb)
                total_loss = total_loss + self.alpha * loss_g
                sample_size += int(b)
                logs["loss_gender"] = loss_g.detach()
                logs["acc_gender"] = acc_g.detach()

        # ===== Tone InfoNCE =====
        if self.beta > 0 and ("anchor_audio" in net_input) and ("positive_audio" in net_input):
            anchor_emb = self.extract_embedding(model, net_input["anchor_audio"], net_input.get("anchor_padding_mask"))

            # positives
            pos_audio = net_input["positive_audio"]  # [B,P,T]
            b, p, t = pos_audio.shape
            pos_flat = pos_audio.view(b * p, t)
            pos_pm = net_input.get("positive_padding_mask")
            pos_pm_flat = pos_pm.view(b * p, t) if pos_pm is not None else None
            pos_emb_flat = self.extract_embedding(model, pos_flat, pos_pm_flat)
            pos_emb = pos_emb_flat.view(b, p, -1)

            # hard negatives
            hard_audio = net_input.get("hard_negative_audio", None)
            if hard_audio is None:
                # no negatives => skip
                pass
            else:
                bh, h, th = hard_audio.shape
                hard_flat = hard_audio.view(bh * h, th)
                hard_pm = net_input.get("hard_negative_padding_mask")
                hard_pm_flat = hard_pm.view(bh * h, th) if hard_pm is not None else None
                hard_emb_flat = self.extract_embedding(model, hard_flat, hard_pm_flat)
                hard_emb = hard_emb_flat.view(bh, h, -1)

                # soft negatives optional
                soft_emb = None
                soft_audio = net_input.get("soft_negative_audio", None)
                if soft_audio is not None and self.tone_use_soft_negatives:
                    bs, s, ts = soft_audio.shape
                    soft_flat = soft_audio.view(bs * s, ts)
                    soft_pm = net_input.get("soft_negative_padding_mask")
                    soft_pm_flat = soft_pm.view(bs * s, ts) if soft_pm is not None else None
                    soft_emb_flat = self.extract_embedding(model, soft_flat, soft_pm_flat)
                    soft_emb = soft_emb_flat.view(bs, s, -1)

                loss_t, acc_t, avg_pos, avg_neg = self._tone_infonce(
                    anchor_emb, pos_emb, hard_emb, soft_emb
                )
                # apply tone_lambda_contrastive inside tone stream
                loss_t_total = self.tone_lambda_contrastive * loss_t
                total_loss = total_loss + self.beta * loss_t_total
                sample_size += int(b)
                logs["loss_tone"] = loss_t.detach()
                logs["loss_tone_scaled"] = loss_t_total.detach()
                logs["acc_tone"] = acc_t.detach()
                logs["tone_avg_pos_sim"] = avg_pos.detach()
                logs["tone_avg_neg_sim"] = avg_neg.detach()

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
                input_lengths = nonpad.sum(-1)
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
            "loss_gender",
            "acc_gender",
            "loss_tone",
            "loss_tone_scaled",
            "acc_tone",
            "tone_avg_pos_sim",
            "tone_avg_neg_sim",
            "loss_ctc",
        ]:
            vals = [log.get(key, None) for log in logging_outputs if key in log]
            if vals:
                metrics.log_scalar(key, sum(vals) / len(vals), len(vals), round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

