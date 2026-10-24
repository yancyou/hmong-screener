# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics


@dataclass
class GenderToneContrastiveCriterionConfig(FairseqDataclass):
    # Loss weights
    alpha: float = field(
        default=1.0,
        metadata={"help": "weight for gender contrastive loss"},
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "weight for tone contrastive loss"},
    )
    gamma: float = field(
        default=0.0,
        metadata={"help": "weight for joint invariance loss (if positives provided)"},
    )

    # Temperatures per head
    temperature_gender: float = field(
        default=0.1,
        metadata={"help": "temperature for gender InfoNCE"},
    )
    temperature_tone: float = field(
        default=0.1,
        metadata={"help": "temperature for tone multi-positive InfoNCE"},
    )
    temperature_joint: float = field(
        default=0.1,
        metadata={"help": "temperature for joint multi-positive InfoNCE"},
    )

    # ASR auxiliary loss (CTC)
    delta: float = field(
        default=0.0,
        metadata={"help": "weight for auxiliary ASR CTC loss (0 disables)"},
    )

    # Optionally reuse a frozen CTC head from a finetuned ASR checkpoint
    asr_ctc_head_ckpt: str = field(
        default="",
        metadata={"help": "path to finetuned ASR checkpoint to load CTC head from"},
    )
    asr_ctc_head_freeze: bool = field(
        default=True,
        metadata={"help": "if true, keep loaded CTC head weights frozen (no grads)"},
    )

    # Embedding extraction
    pooling_mode: str = field(
        default="mean",
        metadata={"help": "pooling over time: mean, max, or last"},
    )
    embedding_layer: int = field(
        default=-1,
        metadata={"help": "layer index for extract_features (-1=last)"},
    )
    freeze_encoder_layers: int = field(
        default=0,
        metadata={"help": "freeze first N encoder layers (0 = no freezing)"},
    )


@register_criterion("gender_tone_contrastive", dataclass=GenderToneContrastiveCriterionConfig)
class GenderToneContrastiveCriterion(FairseqCriterion):
    """
    Combined multi-objective contrastive criterion for gender and tone invariance.

    Expects the task to provide two groups of inputs under `net_input`:
    - Gender group (single-positive InfoNCE):
      female_audio [B, T], male_positive_audio [B, T], male_negative_audio [B, N, T]
      with padding masks: female_padding_mask, male_positive_padding_mask, male_negative_padding_mask
    - Tone group (multi-positive InfoNCE):
      anchor_audio [B, T], positive_audio [B, P, T], negative_audio [B, N, T]
      with padding masks: anchor_padding_mask, positive_padding_mask, negative_padding_mask
      and positive_valid_mask [B, P] (False=valid, True=padding)

    Optionally, joint positives can be supplied:
      joint_positive_audio [B, PJ, T], joint_positive_padding_mask [B, PJ, T], joint_positive_valid_mask [B, PJ]
    """

    def __init__(
        self,
        task,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        temperature_gender: float = 0.1,
        temperature_tone: float = 0.1,
        temperature_joint: float = 0.1,
        delta: float = 0.0,
        asr_ctc_head_ckpt: str = "",
        asr_ctc_head_freeze: bool = True,
        pooling_mode: str = "mean",
        embedding_layer: int = -1,
        freeze_encoder_layers: int = 0,
    ):
        super().__init__(task)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.asr_ctc_head_ckpt = asr_ctc_head_ckpt
        self.asr_ctc_head_freeze = asr_ctc_head_freeze
        self.temperature_gender = temperature_gender
        self.temperature_tone = temperature_tone
        self.temperature_joint = temperature_joint
        self.pooling_mode = pooling_mode
        self.embedding_layer = embedding_layer
        self.freeze_encoder_layers = freeze_encoder_layers
        self._layers_frozen = False
        self._head_installed = False

        # Preload CTC head weights if provided
        self._ctc_head_weight = None
        self._ctc_head_bias = None
        if isinstance(self.asr_ctc_head_ckpt, str) and len(self.asr_ctc_head_ckpt) > 0:
            # Strict mode: if a checkpoint is provided, we must find a CTC head
            # and raise a helpful error otherwise
            # Allow older pickles
            try:
                from torch.serialization import add_safe_globals  # type: ignore
                from fairseq.data import Dictionary  # type: ignore
                add_safe_globals([Dictionary])  # type: ignore
            except Exception:
                pass
            try:
                import torch
                ckpt = torch.load(self.asr_ctc_head_ckpt, map_location='cpu', weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load ASR checkpoint for CTC head: {self.asr_ctc_head_ckpt}: {e}")

            sd = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(sd, dict):
                raise RuntimeError(f"Unexpected checkpoint format (no model state_dict) in {self.asr_ctc_head_ckpt}")

            # Prefer explicit key
            w_key = None
            for k in sd.keys():
                if k.endswith('w2v_encoder.proj.weight') or k == 'w2v_encoder.proj.weight':
                    w_key = k
                    break
            if w_key is None:
                # Fallback: any ".proj.weight" under w2v_encoder
                for k in sd.keys():
                    if k.startswith('w2v_encoder.') and k.endswith('.proj.weight'):
                        w_key = k
                        break
            if w_key is None:
                sample_keys = list(sd.keys())[:60]
                raise RuntimeError(
                    "ASR CTC head not found in provided checkpoint. Expected 'w2v_encoder.proj.{weight,bias}'.\n"
                    f"Checkpoint: {self.asr_ctc_head_ckpt}\nKeys sample: {sample_keys}"
                )

            b_key = w_key[:-6] + 'bias'  # replace .weight -> .bias
            self._ctc_head_weight = sd[w_key].clone().detach()
            self._ctc_head_bias = sd.get(b_key, None)
            if self._ctc_head_bias is not None:
                self._ctc_head_bias = self._ctc_head_bias.clone().detach()

    # ---- Embedding extraction (mirrors existing criterions) ----
    def extract_embedding(self, model, audio_tensor, padding_mask=None):
        if hasattr(model, 'w2v_encoder'):
            encoder_out = model.w2v_encoder(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
            )
            embeddings = encoder_out['encoder_out'].transpose(0, 1)
            padding_mask = encoder_out['padding_mask']
        elif hasattr(model, 'extract_features'):
            result = model.extract_features(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                layer=self.embedding_layer,
            )
            embeddings = result['x']
        else:
            raise ValueError("Model must have either w2v_encoder or extract_features method")

        if padding_mask is not None:
            embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            lengths = (~padding_mask).sum(dim=1).float()
        else:
            lengths = torch.full((embeddings.size(0),), embeddings.size(1), dtype=torch.float, device=embeddings.device)

        if self.pooling_mode == "mean":
            pooled = embeddings.sum(dim=1) / lengths.unsqueeze(-1)
        elif self.pooling_mode == "max":
            pooled = embeddings.max(dim=1)[0]
        elif self.pooling_mode == "last":
            last_indices = (lengths - 1).long()
            pooled = embeddings[torch.arange(embeddings.size(0)), last_indices]
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        return pooled

    # ---- Losses ----
    def info_nce_single(self, anchor, positive, negatives, temperature):
        batch_size = anchor.size(0)

        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = torch.sum(anchor * positive, dim=-1) / temperature  # [B]
        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / temperature  # [B, N]

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, targets, reduction='mean')
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == 0).float().mean()

        return loss, acc, batch_size

    def info_nce_multi_positive(self, anchor, positives, negatives, valid_mask, temperature):
        # anchors: [B, C], positives: [B, P, C], negatives: [B, N, C], valid_mask: [B, P] (False=valid)
        batch_size, max_p, _ = positives.shape

        anchor = F.normalize(anchor, dim=-1)
        positives = F.normalize(positives, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        anchor_exp = anchor.unsqueeze(1)
        pos_sim = torch.sum(anchor_exp * positives, dim=-1)  # [B, P]
        pos_sim_masked = pos_sim.masked_fill(valid_mask, 0.0)
        valid_count = (~valid_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        mean_pos = pos_sim_masked.sum(dim=1, keepdim=True) / valid_count  # [B, 1]

        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)  # [B, N]

        logits = torch.cat([mean_pos, neg_sim], dim=1) / temperature
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, targets, reduction='mean')
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == 0).float().mean()

        return loss, acc, batch_size

    def _maybe_install_ctc_head(self, model):
        """If a CTC head was loaded from a checkpoint, install it into the model so it trains/saves properly."""
        if self._ctc_head_weight is None or self._head_installed:
            return
        # Try to find existing proj
        proj = getattr(getattr(model, 'w2v_encoder', object()), 'proj', None)

        
        out_dim = self._ctc_head_weight.size(0)
        in_dim = self._ctc_head_weight.size(1)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if proj is not None and isinstance(proj, nn.Linear) and proj.in_features == in_dim and proj.out_features == out_dim:
            # Overwrite weights
            proj.weight.data = self._ctc_head_weight.to(device=device, dtype=dtype).clone()
            if self._ctc_head_bias is not None and proj.bias is not None:
                proj.bias.data = self._ctc_head_bias.to(device=device, dtype=dtype).clone()
            elif proj.bias is not None:
                # No bias in ckpt, zero existing bias
                proj.bias.data.zero_()
        else:
            # Create new projection layer and assign
            new_proj = nn.Linear(in_dim, out_dim, bias=self._ctc_head_bias is not None)
            new_proj.to(device=device, dtype=dtype)
            new_proj.weight.data = self._ctc_head_weight.to(device=device, dtype=dtype).clone()
            if self._ctc_head_bias is not None:
                new_proj.bias.data = self._ctc_head_bias.to(device=device, dtype=dtype).clone()
            setattr(model.w2v_encoder, 'proj', new_proj)

        # Freeze or unfreeze according to flag (enforce immediately upon install)
        proj = model.w2v_encoder.proj
        proj.weight.requires_grad_(not self.asr_ctc_head_freeze)
        if proj.bias is not None:
            proj.bias.requires_grad_(not self.asr_ctc_head_freeze)

        self._head_installed = True

    def forward(self, model, sample, reduce=True):
        # Optional layer freezing
        if self.freeze_encoder_layers > 0 and not getattr(self, "_layers_frozen", False):
            try:
                # Lazy import to avoid mandatory dependency
                from freeze_layers_hook import freeze_encoder_layers, count_trainable_parameters  # type: ignore

                freeze_encoder_layers(model, self.freeze_encoder_layers)
                count_trainable_parameters(model)
                self._layers_frozen = True
            except Exception:
                # Silently continue if not available
                self._layers_frozen = True

        net_input = sample['net_input']

        total_loss = 0.0
        total_size = 0
        logs = {}

        # ---- Gender loss (if provided) ----
        if all(k in net_input for k in [
            'female_audio', 'male_positive_audio', 'male_negative_audio',
            'female_padding_mask', 'male_positive_padding_mask', 'male_negative_padding_mask'
        ]):
            female_embed = self.extract_embedding(
                model,
                net_input['female_audio'],
                net_input.get('female_padding_mask'),
            )
            male_pos_embed = self.extract_embedding(
                model,
                net_input['male_positive_audio'],
                net_input.get('male_positive_padding_mask'),
            )
            B, N, T = net_input['male_negative_audio'].shape
            neg_audio_flat = net_input['male_negative_audio'].view(-1, T)
            neg_mask = net_input.get('male_negative_padding_mask')
            neg_mask_flat = neg_mask.view(-1, T) if neg_mask is not None else None
            neg_embed_flat = self.extract_embedding(model, neg_audio_flat, neg_mask_flat)
            embed_dim = neg_embed_flat.size(-1)
            male_neg_embed = neg_embed_flat.view(B, N, embed_dim)

            loss_g, acc_g, size_g = self.info_nce_single(
                female_embed, male_pos_embed, male_neg_embed, self.temperature_gender
            )
            total_loss = total_loss + self.alpha * loss_g
            total_size += size_g
            logs.update({
                "loss_gender": loss_g.data,
                "accuracy_gender": acc_g.data if isinstance(acc_g, torch.Tensor) else acc_g,
            })

        # ---- Tone loss (if provided) ----
        if all(k in net_input for k in [
            'anchor_audio', 'positive_audio', 'negative_audio',
            'anchor_padding_mask', 'positive_padding_mask', 'negative_padding_mask', 'positive_valid_mask'
        ]):
            anchor_embed = self.extract_embedding(
                model,
                net_input['anchor_audio'],
                net_input.get('anchor_padding_mask'),
            )
            B, P, T = net_input['positive_audio'].shape
            pos_audio_flat = net_input['positive_audio'].view(-1, T)
            pos_mask = net_input.get('positive_padding_mask')
            pos_mask_flat = pos_mask.view(-1, T) if pos_mask is not None else None
            pos_embed_flat = self.extract_embedding(model, pos_audio_flat, pos_mask_flat)
            embed_dim = pos_embed_flat.size(-1)
            pos_embed = pos_embed_flat.view(B, P, embed_dim)

            Bn, N, Tn = net_input['negative_audio'].shape
            neg_audio_flat = net_input['negative_audio'].view(-1, Tn)
            neg_mask = net_input.get('negative_padding_mask')
            neg_mask_flat = neg_mask.view(-1, Tn) if neg_mask is not None else None
            neg_embed_flat = self.extract_embedding(model, neg_audio_flat, neg_mask_flat)
            embed_dim_n = neg_embed_flat.size(-1)
            neg_embed = neg_embed_flat.view(Bn, N, embed_dim_n)

            valid_mask = net_input.get('positive_valid_mask')
            loss_t, acc_t, size_t = self.info_nce_multi_positive(
                anchor_embed, pos_embed, neg_embed, valid_mask, self.temperature_tone
            )
            total_loss = total_loss + self.beta * loss_t
            total_size += size_t
            logs.update({
                "loss_tone": loss_t.data,
                "accuracy_tone": acc_t.data if isinstance(acc_t, torch.Tensor) else acc_t,
            })

        # ---- Joint loss (optional, only if joint positives provided) ----
        if self.gamma > 0.0 and all(k in net_input for k in [
            'joint_positive_audio', 'joint_positive_padding_mask', 'joint_positive_valid_mask',
            'anchor_audio', 'negative_audio', 'negative_padding_mask', 'anchor_padding_mask'
        ]):
            # Reuse tone anchor/negatives, different positive set
            anchor_embed = self.extract_embedding(
                model,
                net_input['anchor_audio'],
                net_input.get('anchor_padding_mask'),
            )
            B, PJ, Tj = net_input['joint_positive_audio'].shape
            jpos_audio_flat = net_input['joint_positive_audio'].view(-1, Tj)
            jpos_mask = net_input.get('joint_positive_padding_mask')
            jpos_mask_flat = jpos_mask.view(-1, Tj) if jpos_mask is not None else None
            jpos_embed_flat = self.extract_embedding(model, jpos_audio_flat, jpos_mask_flat)
            ed = jpos_embed_flat.size(-1)
            jpos_embed = jpos_embed_flat.view(B, PJ, ed)

            Bn, N, Tn = net_input['negative_audio'].shape
            neg_audio_flat = net_input['negative_audio'].view(-1, Tn)
            neg_mask = net_input.get('negative_padding_mask')
            neg_mask_flat = neg_mask.view(-1, Tn) if neg_mask is not None else None
            neg_embed_flat = self.extract_embedding(model, neg_audio_flat, neg_mask_flat)
            edn = neg_embed_flat.size(-1)
            neg_embed = neg_embed_flat.view(Bn, N, edn)

            jvalid = net_input.get('joint_positive_valid_mask')
            loss_j, acc_j, size_j = self.info_nce_multi_positive(
                anchor_embed, jpos_embed, neg_embed, jvalid, self.temperature_joint
            )
            total_loss = total_loss + self.gamma * loss_j
            total_size += size_j
            logs.update({
                "loss_joint": loss_j.data,
                "accuracy_joint": acc_j.data if isinstance(acc_j, torch.Tensor) else acc_j,
            })

        # ---- ASR auxiliary CTC loss (optional) ----
        if self.delta > 0.0 and all(k in net_input for k in [
            'asr_audio',  # [B, T]
            # padding mask may be None when not padded
            'asr_target',  # [B, L]
            'asr_target_lengths',  # [B]
        ]):
            # If a head was provided, install into model (so it trains/saves)
            self._maybe_install_ctc_head(model)
            # Enforce freezing of CTC projection parameters if requested (or by default)
            try:
                proj = getattr(getattr(model, 'w2v_encoder', object()), 'proj', None)
                if proj is not None:
                    # Always enforce freeze when flag is true; default is True
                    if getattr(self, 'asr_ctc_head_freeze', True):
                        if hasattr(proj, 'weight') and proj.weight is not None:
                            proj.weight.requires_grad_(False)
                        if hasattr(proj, 'bias') and proj.bias is not None:
                            proj.bias.requires_grad_(False)
            except Exception:
                pass
            
            # Print CTC layer status before training
            proj = getattr(getattr(model, 'w2v_encoder', object()), 'proj', None)
            if proj is not None and not getattr(self, '_ctc_status_printed', False):
                print("\n" + "="*80)
                print("CTC HEAD STATUS BEFORE TRAINING")
                print("="*80)
                print(f"CTC head checkpoint: {self.asr_ctc_head_ckpt}")
                print(f"asr_ctc_head_freeze flag: {self.asr_ctc_head_freeze}")
                print(f"CTC layer type: {type(proj)}")
                print(f"CTC layer shape: in_features={proj.in_features}, out_features={proj.out_features}")
                print(f"proj.weight.requires_grad: {proj.weight.requires_grad}")
                if proj.bias is not None:
                    print(f"proj.bias.requires_grad: {proj.bias.requires_grad}")
                else:
                    print(f"proj.bias: None")
                print(f"Expected requires_grad (based on freeze flag): {not self.asr_ctc_head_freeze}")
                
                # Check if status matches expectation
                expected_grad = not self.asr_ctc_head_freeze
                if proj.weight.requires_grad == expected_grad:
                    print("✓ CTC head weight freeze status is CORRECT")
                else:
                    print(f"✗ WARNING: CTC head weight freeze status is WRONG! Expected {expected_grad}, got {proj.weight.requires_grad}")
                
                if proj.bias is not None:
                    if proj.bias.requires_grad == expected_grad:
                        print("✓ CTC head bias freeze status is CORRECT")
                    else:
                        print(f"✗ WARNING: CTC head bias freeze status is WRONG! Expected {expected_grad}, got {proj.bias.requires_grad}")
                
                print("="*80 + "\n")
                self._ctc_status_printed = True
            
            # Use model path to get logits
            net_out_asr = model(
                source=net_input['asr_audio'],
                padding_mask=net_input.get('asr_padding_mask'),
                mask=False,
            )
            log_probs = model.get_logits(net_out_asr, normalize=True)  # [T, B, C]

            if net_out_asr.get("padding_mask") is not None:
                nonpad = (~net_out_asr["padding_mask"]).long()
                input_lengths = nonpad.sum(-1)  # [B]
            else:
                input_lengths = log_probs.new_full(
                    (log_probs.size(1),), log_probs.size(0), dtype=torch.long
                )

            targets = net_input['asr_target']  # [B, L]
            target_lengths = net_input['asr_target_lengths']  # [B]

            # torch.nn.CTCLoss expects (T, B, C)
            ctc_loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=True,
            )

            total_loss = total_loss + self.delta * ctc_loss
            total_size += targets.size(0)
            logs.update({
                "loss_ctc": ctc_loss.data,
            })

        if total_size == 0:
            # Nothing provided; return zero loss
            dummy = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
            logging_output = {"loss": dummy.data, "sample_size": 0}
            return dummy, 0, logging_output

        logging_output = {
            "loss": (total_loss).data,
            "sample_size": total_size,
        }
        logging_output.update(logs)
        return total_loss, total_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        # Optional per-head metrics if present
        for key in ["loss_gender", "loss_tone", "loss_joint"]:
            vals = [log.get(key, None) for log in logging_outputs]
            vals = [v for v in vals if v is not None]
            if vals and sample_size > 0:
                metrics.log_scalar(key, sum(vals) / len(vals) / math.log(2), len(vals), round=3)

        # Optional auxiliary metrics that are not in bits
        for key in ["accuracy_gender", "accuracy_tone", "accuracy_joint", "loss_ctc"]:
            vals = [log.get(key, None) for log in logging_outputs]
            vals = [v for v in vals if v is not None]
            if vals:
                # Log averages directly
                metrics.log_scalar(key, sum(vals) / len(vals), len(vals), round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True


