# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics


@dataclass
class ToneRepulsiveHingeCriterionConfig(FairseqDataclass):
    temperature: float = field(
        default=0.07,
        metadata={"help": "temperature for attractive InfoNCE loss"}
    )
    margin_hard: float = field(
        default=-0.3,
        metadata={"help": "target margin for same-word different-tone (want cos < margin)"}
    )
    margin_soft: float = field(
        default=-0.2,
        metadata={"help": "target margin for different-word (want cos < margin)"}
    )
    lambda_attract: float = field(
        default=1.0,
        metadata={"help": "weight for attractive loss (pull same word+tone together)"}
    )
    lambda_repulse_hard: float = field(
        default=1.5,
        metadata={"help": "weight for repulsive hard loss (push different tones apart)"}
    )
    lambda_repulse_soft: float = field(
        default=0.3,
        metadata={"help": "weight for repulsive soft loss (push different words apart)"}
    )
    lambda_classification: float = field(
        default=0.0,
        metadata={"help": "weight for tone classification loss (0 = disabled, 1.0 = pure classification)"}
    )
    num_tone_classes: int = field(
        default=7,
        metadata={"help": "number of tone classes (default 7 for Hmong)"}
    )
    freeze_tone_classifier: bool = field(
        default=False,
        metadata={"help": "freeze tone classifier weights (requires_grad=False). Use after pre-training classifier."}
    )
    tone_classifier_ckpt: str = field(
        default="",
        metadata={"help": "path to checkpoint with pre-trained tone classifier (optional)"}
    )
    pooling_mode: str = field(
        default="mean",
        metadata={"help": "pooling mode: mean, max, or last"}
    )
    freeze_encoder_layers: int = field(
        default=0,
        metadata={"help": "freeze first N encoder layers (0 = no freezing)"}
    )


@register_criterion("tone_repulsive_hinge", dataclass=ToneRepulsiveHingeCriterionConfig)
class ToneRepulsiveHingeCriterion(FairseqCriterion):
    """
    Tone learning criterion with Hinge Loss (explicit hard/soft negative separation):
    
    1. Attractive loss: pull same word+tone together (InfoNCE style)
    2. Repulsive hard loss (Hinge): push same word+different tone apart (CRITICAL for tone discrimination)
    3. Repulsive soft loss (Hinge): push different words apart (less important, word-level separation)
    4. Optional classification: supervised tone class prediction
    
    Key difference from InfoNCE version:
    - Hard negatives get SEPARATE, STRONGER treatment (higher weight, stricter margin)
    - Hinge loss provides explicit margin-based separation
    - Better monitoring: separately track hard vs soft negative similarities
    
    Total loss = λ_attract * L_attract + λ_hard * L_hard + λ_soft * L_soft + λ_cls * L_cls
    """

    def __init__(self, task, temperature=0.07, margin_hard=-0.3, margin_soft=-0.2,
                 lambda_attract=1.0, lambda_repulse_hard=1.5, lambda_repulse_soft=0.3,
                 lambda_classification=0.0, num_tone_classes=7,
                 freeze_tone_classifier=False, tone_classifier_ckpt="",
                 pooling_mode="mean", freeze_encoder_layers=0):
        super().__init__(task)
        self.temperature = temperature
        self.margin_hard = margin_hard
        self.margin_soft = margin_soft
        self.lambda_attract = lambda_attract
        self.lambda_repulse_hard = lambda_repulse_hard
        self.lambda_repulse_soft = lambda_repulse_soft
        self.lambda_classification = lambda_classification
        self.num_tone_classes = num_tone_classes
        self.freeze_tone_classifier = freeze_tone_classifier
        self.tone_classifier_ckpt = tone_classifier_ckpt
        self.pooling_mode = pooling_mode
        self.freeze_encoder_layers = freeze_encoder_layers
        self._layers_frozen = False
        self._classifier_frozen = False
        
        # Tone classification head (lazily initialized on first forward pass)
        self.tone_classifier = None
        
        # Load pre-trained classifier if specified
        if tone_classifier_ckpt:
            self._load_pretrained_classifier()
        
        print(f"[ToneRepulsiveHingeCriterion] Initialized with:")
        print(f"  Attractive weight:     {lambda_attract}")
        print(f"  Hard repulsive weight: {lambda_repulse_hard}, margin: {margin_hard}")
        print(f"  Soft repulsive weight: {lambda_repulse_soft}, margin: {margin_soft}")
        print(f"  Classification weight: {lambda_classification}")

    def extract_embedding(self, model, audio_tensor, padding_mask=None):
        """Extract and pool embeddings from encoder"""
        if hasattr(model, 'w2v_encoder'):
            encoder_out = model.w2v_encoder(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                features_only=True
            )
            embeddings = encoder_out['encoder_out'].transpose(0, 1)  # [B, T, C]
            padding_mask = encoder_out['padding_mask']
        elif hasattr(model, 'extract_features'):
            result = model.extract_features(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                layer=-1
            )
            embeddings = result['x']  # [B, T, C]
        else:
            raise ValueError("Model must have w2v_encoder or extract_features")

        # Pool over time
        if padding_mask is not None:
            embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            lengths = (~padding_mask).sum(dim=1).float()
        else:
            lengths = torch.full((embeddings.size(0),), embeddings.size(1),
                               dtype=torch.float, device=embeddings.device)

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

    def attractive_loss(self, anchor, positives, positive_valid_mask):
        """
        Pull same word+tone together (multi-positive InfoNCE style).
        
        Args:
            anchor: [B, C] - anchor embeddings
            positives: [B, P, C] - positive embeddings (same word+tone, augmented)
            positive_valid_mask: [B, P] - True where positives are invalid/padding
        
        Returns:
            loss: scalar loss
            avg_pos_sim: average positive similarity
        """
        batch_size, max_pos, _ = positives.shape
        
        # Normalize
        anchor = F.normalize(anchor, dim=-1)  # [B, C]
        positives = F.normalize(positives, dim=-1)  # [B, P, C]
        
        # Positive similarities
        anchor_exp = anchor.unsqueeze(1)  # [B, 1, C]
        pos_sim = torch.sum(anchor_exp * positives, dim=-1)  # [B, P]
        
        # Mask and average valid positives
        pos_sim_masked = pos_sim.masked_fill(positive_valid_mask, 0.0)
        valid_count = (~positive_valid_mask).sum(dim=1).float().clamp(min=1.0)  # [B]
        mean_pos_sim = pos_sim_masked.sum(dim=1) / valid_count  # [B]
        
        # Loss: maximize similarity (minimize negative similarity)
        loss = -mean_pos_sim.mean()
        
        with torch.no_grad():
            avg_pos_sim = mean_pos_sim.mean()
        
        return loss, avg_pos_sim

    def repulsive_loss(self, anchor, negatives, margin, negative_valid_mask=None):
        """
        Push negatives away using Hinge Loss.
        
        We want: cos_sim < margin (typically margin < 0, e.g., -0.3)
        Hinge loss: max(0, sim - margin)
        - If sim < margin: loss = 0 (good, already separated)
        - If sim > margin: loss = sim - margin (penalize)
        
        Args:
            anchor: [B, C] - anchor embeddings
            negatives: [B, N, C] - negative embeddings
            margin: float - target margin (want similarity below this)
            negative_valid_mask: [B, N] - True where negatives are invalid/padding
        
        Returns:
            loss: scalar loss
            avg_neg_sim: average negative similarity
            violation_rate: percentage of negatives violating margin
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)  # [B, C]
        negatives = F.normalize(negatives, dim=-1)  # [B, N, C]
        
        # Compute cosine similarities
        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)  # [B, N]
        
        # Mask invalid negatives if provided
        if negative_valid_mask is not None:
            neg_sim_masked = neg_sim.masked_fill(negative_valid_mask, margin - 1.0)  # Set to below margin
        else:
            neg_sim_masked = neg_sim
        
        # Hinge loss: penalize if similarity > margin
        # We want cos_sim < margin (e.g., < -0.3 for hard, < -0.2 for soft)
        violations = neg_sim_masked - margin  # [B, N]
        hinge_loss = F.relu(violations)  # [B, N], zero out satisfied constraints
        
        # Average over valid negatives
        if negative_valid_mask is not None:
            valid_mask = ~negative_valid_mask
            loss = (hinge_loss * valid_mask.float()).sum() / valid_mask.sum().float().clamp(min=1.0)
        else:
            loss = hinge_loss.mean()
        
        # Compute statistics
        with torch.no_grad():
            if negative_valid_mask is not None:
                avg_neg_sim = (neg_sim * (~negative_valid_mask).float()).sum() / (~negative_valid_mask).sum().float().clamp(min=1.0)
                violation_rate = (violations > 0).float().sum() / (~negative_valid_mask).sum().float().clamp(min=1.0)
            else:
                avg_neg_sim = neg_sim.mean()
                violation_rate = (violations > 0).float().mean()
        
        return loss, avg_neg_sim, violation_rate

    def _load_pretrained_classifier(self, model=None):
        """Load pre-trained tone classifier from checkpoint"""
        import os
        if not self.tone_classifier_ckpt or not os.path.exists(self.tone_classifier_ckpt):
            print(f"[ToneRepulsiveHingeCriterion] Warning: Classifier checkpoint not found: {self.tone_classifier_ckpt}")
            return
        
        try:
            print(f"[ToneRepulsiveHingeCriterion] Loading pre-trained classifier from {self.tone_classifier_ckpt}")
            checkpoint = torch.load(self.tone_classifier_ckpt, map_location='cpu')
            
            # Extract classifier state from criterion state
            if 'criterion' in checkpoint and checkpoint['criterion'] is not None:
                criterion_state = checkpoint['criterion']
                if 'tone_classifier' in criterion_state:
                    # Infer embed_dim from saved weights
                    weight_shape = criterion_state['tone_classifier']['weight'].shape
                    embed_dim = weight_shape[1]  # [num_classes, embed_dim]
                    
                    # Create classifier (will be moved to correct device/dtype later)
                    self.tone_classifier = nn.Linear(embed_dim, self.num_tone_classes)
                    self.tone_classifier.load_state_dict(criterion_state['tone_classifier'])
                    print(f"[ToneRepulsiveHingeCriterion] Successfully loaded tone classifier (embed_dim={embed_dim})")
                    
                    # If model is provided, convert dtype immediately
                    if model is not None:
                        device = next(model.parameters()).device
                        dtype = next(model.parameters()).dtype
                        self.tone_classifier = self.tone_classifier.to(device=device, dtype=dtype)
                        print(f"[ToneRepulsiveHingeCriterion] Converted classifier to device={device}, dtype={dtype}")
                else:
                    print(f"[ToneRepulsiveHingeCriterion] Warning: No tone_classifier found in checkpoint")
            else:
                print(f"[ToneRepulsiveHingeCriterion] Warning: No criterion state in checkpoint")
        except Exception as e:
            print(f"[ToneRepulsiveHingeCriterion] Error loading classifier: {e}")

    def _maybe_init_tone_classifier(self, model):
        """Lazily initialize tone classification head"""
        if self.tone_classifier is not None:
            return
        
        # Get embedding dimension from model
        if hasattr(model, 'w2v_encoder'):
            # Try to infer from encoder
            if hasattr(model.w2v_encoder, 'w2v_model'):
                if hasattr(model.w2v_encoder.w2v_model, 'encoder'):
                    if hasattr(model.w2v_encoder.w2v_model.encoder, 'embedding_dim'):
                        embed_dim = model.w2v_encoder.w2v_model.encoder.embedding_dim
                    else:
                        # Fallback: get from first layer
                        embed_dim = model.w2v_encoder.w2v_model.encoder.layers[0].embed_dim
                else:
                    embed_dim = 1024  # Default for XLSR
            else:
                embed_dim = 1024
        else:
            embed_dim = 1024  # Default
        
        # Create classification head
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype  # Match model's dtype (fp16/fp32)
        self.tone_classifier = nn.Linear(embed_dim, self.num_tone_classes).to(device=device, dtype=dtype)
        
        # Initialize weights (only if not loaded from checkpoint)
        if not self.tone_classifier_ckpt:
            nn.init.xavier_uniform_(self.tone_classifier.weight)
            nn.init.zeros_(self.tone_classifier.bias)
        
        print(f"[ToneRepulsiveHingeCriterion] Initialized tone classifier: {embed_dim} -> {self.num_tone_classes} (dtype={dtype})")
    
    def _maybe_freeze_tone_classifier(self):
        """Freeze tone classifier if requested"""
        if self._classifier_frozen or not self.freeze_tone_classifier or self.tone_classifier is None:
            return
        
        # Freeze classifier parameters
        frozen_count = 0
        for param in self.tone_classifier.parameters():
            param.requires_grad = False
            frozen_count += 1
        
        self._classifier_frozen = True
        print(f"[ToneRepulsiveHingeCriterion] ✓ Frozen tone classifier: {frozen_count} parameters set to requires_grad=False")
        
        # Verify freezing
        for name, param in self.tone_classifier.named_parameters():
            if param.requires_grad:
                print(f"[ToneRepulsiveHingeCriterion] ⚠️  WARNING: {name} still has requires_grad=True!")
            else:
                print(f"[ToneRepulsiveHingeCriterion]   ✓ {name}: requires_grad=False")
    
    def _maybe_freeze_encoder_layers(self, model):
        """Freeze encoder layers if requested"""
        if self._layers_frozen or self.freeze_encoder_layers == 0:
            return
        
        if hasattr(model, 'w2v_encoder') and hasattr(model.w2v_encoder, 'w2v_model'):
            if hasattr(model.w2v_encoder.w2v_model, 'encoder'):
                encoder = model.w2v_encoder.w2v_model.encoder
                if hasattr(encoder, 'layers'):
                    total_layers = len(encoder.layers)
                    for i in range(min(self.freeze_encoder_layers, total_layers)):
                        for param in encoder.layers[i].parameters():
                            param.requires_grad = False
                    self._layers_frozen = True
                    print(f"[ToneRepulsiveHingeCriterion] Frozen first {self.freeze_encoder_layers} encoder layers")
    
    def classification_loss(self, embeddings, tone_labels):
        """
        Compute tone classification loss.
        
        Args:
            embeddings: [B, C] pooled embeddings
            tone_labels: [B] tone class indices (0-6 for 7 tones)
        
        Returns:
            loss: scalar loss
            accuracy: scalar accuracy
        """
        # Get logits
        logits = self.tone_classifier(embeddings)  # [B, num_classes]
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, tone_labels, reduction='mean')
        
        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == tone_labels).sum().float()
            total = tone_labels.size(0)
            accuracy = correct / total if total > 0 else torch.tensor(0.0)
        
        return loss, accuracy

    def forward(self, model, sample, reduce=True):
        """
        Expected sample structure:
        {
            'net_input': {
                'anchor_audio': [B, T],
                'positive_audio': [B, P, T],  # Same word+tone (augmentations)
                'hard_negative_audio': [B, H, T],  # Same word, different tone (CRITICAL!)
                'soft_negative_audio': [B, S, T],  # Different word
                'anchor_padding_mask': [B, T],
                'positive_padding_mask': [B, P, T],
                'hard_negative_padding_mask': [B, H, T],
                'soft_negative_padding_mask': [B, S, T],
                'positive_valid_mask': [B, P],
                'hard_negative_valid_mask': [B, H],
                'soft_negative_valid_mask': [B, S],
                'anchor_tone_label': [B],  # Optional: tone labels for classification
            }
        }
        """
        # Initialize classification head and freeze if needed
        if self.lambda_classification > 0 or self.tone_classifier_ckpt:
            self._maybe_init_tone_classifier(model)
            self._maybe_freeze_tone_classifier()
        self._maybe_freeze_encoder_layers(model)
        
        net_input = sample['net_input']

        # Extract anchor embeddings
        anchor_emb = self.extract_embedding(
            model,
            net_input['anchor_audio'],
            net_input.get('anchor_padding_mask')
        )

        batch_size = anchor_emb.size(0)
        total_loss = torch.tensor(0.0, device=anchor_emb.device, requires_grad=True)
        logs = {}

        # Part A: Attractive loss (pull positives together)
        if self.lambda_attract > 0 and 'positive_audio' in net_input:
            B, P, T = net_input['positive_audio'].shape
            pos_audio_flat = net_input['positive_audio'].view(-1, T)
            pos_mask = net_input.get('positive_padding_mask')
            pos_mask_flat = pos_mask.view(-1, T) if pos_mask is not None else None

            pos_emb_flat = self.extract_embedding(model, pos_audio_flat, pos_mask_flat)
            pos_emb = pos_emb_flat.view(B, P, -1)
            pos_valid_mask = net_input.get('positive_valid_mask', torch.zeros(B, P, dtype=torch.bool, device=anchor_emb.device))

            loss_attr, avg_pos_sim = self.attractive_loss(anchor_emb, pos_emb, pos_valid_mask)
            total_loss = total_loss + self.lambda_attract * loss_attr
            logs['loss_attract'] = loss_attr.data
            logs['avg_pos_sim'] = avg_pos_sim.data

        # Part B: Repulsive HARD loss (push same-word-different-tone apart)
        # THIS IS CRITICAL FOR TONE DISCRIMINATION!
        if self.lambda_repulse_hard > 0 and 'hard_negative_audio' in net_input:
            B, H, T = net_input['hard_negative_audio'].shape
            hard_neg_flat = net_input['hard_negative_audio'].view(-1, T)
            hard_neg_mask = net_input.get('hard_negative_padding_mask')
            hard_neg_mask_flat = hard_neg_mask.view(-1, T) if hard_neg_mask is not None else None

            hard_neg_emb_flat = self.extract_embedding(model, hard_neg_flat, hard_neg_mask_flat)
            hard_neg_emb = hard_neg_emb_flat.view(B, H, -1)
            hard_neg_valid_mask = net_input.get('hard_negative_valid_mask')

            loss_rep_hard, avg_hard_sim, hard_violation_rate = self.repulsive_loss(
                anchor_emb, hard_neg_emb, self.margin_hard, hard_neg_valid_mask
            )
            total_loss = total_loss + self.lambda_repulse_hard * loss_rep_hard
            logs['loss_repulse_hard'] = loss_rep_hard.data
            logs['avg_hard_neg_sim'] = avg_hard_sim.data
            logs['hard_violation_rate'] = hard_violation_rate.data

        # Part C: Repulsive SOFT loss (push different words apart)
        # Less important, mainly for word-level separation
        if self.lambda_repulse_soft > 0 and 'soft_negative_audio' in net_input:
            B, S, T = net_input['soft_negative_audio'].shape
            soft_neg_flat = net_input['soft_negative_audio'].view(-1, T)
            soft_neg_mask = net_input.get('soft_negative_padding_mask')
            soft_neg_mask_flat = soft_neg_mask.view(-1, T) if soft_neg_mask is not None else None

            soft_neg_emb_flat = self.extract_embedding(model, soft_neg_flat, soft_neg_mask_flat)
            soft_neg_emb = soft_neg_emb_flat.view(B, S, -1)
            soft_neg_valid_mask = net_input.get('soft_negative_valid_mask')

            loss_rep_soft, avg_soft_sim, soft_violation_rate = self.repulsive_loss(
                anchor_emb, soft_neg_emb, self.margin_soft, soft_neg_valid_mask
            )
            total_loss = total_loss + self.lambda_repulse_soft * loss_rep_soft
            logs['loss_repulse_soft'] = loss_rep_soft.data
            logs['avg_soft_neg_sim'] = avg_soft_sim.data
            logs['soft_violation_rate'] = soft_violation_rate.data

        # Part D: Classification loss (supervised tone prediction)
        if self.lambda_classification > 0 and 'anchor_tone_label' in net_input:
            tone_labels = net_input['anchor_tone_label']  # [B]
            loss_cls, accuracy_cls = self.classification_loss(anchor_emb, tone_labels)
            total_loss = total_loss + self.lambda_classification * loss_cls
            logs['loss_classification'] = loss_cls.data
            logs['tone_accuracy'] = accuracy_cls.data

        logging_output = {
            "loss": total_loss.data,
            "ntokens": sample.get("ntokens", batch_size),
            "nsentences": batch_size,
            "sample_size": batch_size,
        }
        logging_output.update(logs)

        return total_loss, batch_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs"""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        # Log component losses
        for key in ["loss_attract", "loss_repulse_hard", "loss_repulse_soft", "loss_classification",
                    "avg_pos_sim", "avg_hard_neg_sim", "avg_soft_neg_sim", 
                    "hard_violation_rate", "soft_violation_rate", "tone_accuracy"]:
            vals = [log.get(key, None) for log in logging_outputs if key in log]
            if vals:
                metrics.log_scalar(key, sum(vals) / len(vals), len(vals), round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Save criterion state including tone classifier"""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.tone_classifier is not None:
            state['tone_classifier'] = self.tone_classifier.state_dict()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load criterion state including tone classifier"""
        # Load tone classifier if present
        if 'tone_classifier' in state_dict and self.lambda_classification > 0:
            # Initialize classifier if not already done
            if self.tone_classifier is None:
                # Need to infer embed_dim from state_dict
                weight_shape = state_dict['tone_classifier']['weight'].shape
                embed_dim = weight_shape[1]  # [num_classes, embed_dim]
                self.tone_classifier = nn.Linear(embed_dim, self.num_tone_classes)
            
            self.tone_classifier.load_state_dict(state_dict['tone_classifier'])
            print(f"[ToneRepulsiveHingeCriterion] Loaded tone classifier from checkpoint")
        
        # Remove tone_classifier from state_dict before calling super
        # (since parent class doesn't expect it)
        state_dict_copy = {k: v for k, v in state_dict.items() if k != 'tone_classifier'}
        super().load_state_dict(state_dict_copy, strict=False)

