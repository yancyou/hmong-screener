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
class ToneRepulsiveCriterionConfig(FairseqDataclass):
    temperature: float = field(
        default=0.07,
        metadata={"help": "temperature for InfoNCE contrastive loss"}
    )
    lambda_contrastive: float = field(
        default=1.0,
        metadata={"help": "weight for contrastive loss"}
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


@register_criterion("tone_repulsive", dataclass=ToneRepulsiveCriterionConfig)
class ToneRepulsiveCriterion(FairseqCriterion):
    """
    Simplified tone learning criterion with standard InfoNCE contrastive loss:
    - Positives: same word + same tone (augmented versions)
    - Negatives: all other samples (different tones/words)
    - Optional: supervised classification loss
    
    Total loss = λ_contrastive * L_InfoNCE + λ_classification * L_cls
    """

    def __init__(self, task, temperature=0.07, lambda_contrastive=1.0,
                 lambda_classification=0.0, num_tone_classes=7,
                 freeze_tone_classifier=False, tone_classifier_ckpt="",
                 pooling_mode="mean", freeze_encoder_layers=0):
        super().__init__(task)
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
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

    def contrastive_loss(self, anchor, positives, negatives, positive_valid_mask, negative_valid_mask):
        """
        Standard InfoNCE contrastive loss.
        
        Args:
            anchor: [B, C] - anchor embeddings
            positives: [B, P, C] - positive embeddings (same word+tone, augmented)
            negatives: [B, N, C] - negative embeddings (different tones/words)
            positive_valid_mask: [B, P] - True where positives are invalid/padding
            negative_valid_mask: [B, N] - True where negatives are invalid/padding
        
        Returns:
            loss: scalar InfoNCE loss
            accuracy: scalar accuracy (correct positive rank)
            avg_pos_sim: average positive similarity
            avg_neg_sim: average negative similarity
        """
        batch_size = anchor.size(0)
        
        # Normalize all embeddings
        anchor = F.normalize(anchor, dim=-1)  # [B, C]
        positives = F.normalize(positives, dim=-1)  # [B, P, C]
        negatives = F.normalize(negatives, dim=-1)  # [B, N, C]
        
        # Compute positive similarities
        anchor_exp = anchor.unsqueeze(1)  # [B, 1, C]
        pos_sim = torch.sum(anchor_exp * positives, dim=-1)  # [B, P]
        
        # Mask invalid positives and average across valid ones
        pos_sim_masked = pos_sim.masked_fill(positive_valid_mask, float('-inf'))
        pos_logits = pos_sim_masked / self.temperature  # [B, P]
        
        # For InfoNCE, we'll use the mean of valid positive logits as the "positive score"
        valid_pos_count = (~positive_valid_mask).sum(dim=1).float().clamp(min=1.0)  # [B]
        pos_logits_valid = pos_logits.masked_fill(positive_valid_mask, 0.0)
        mean_pos_logit = pos_logits_valid.sum(dim=1) / valid_pos_count  # [B]
        
        # Compute negative similarities
        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)  # [B, N]
        neg_sim_masked = neg_sim.masked_fill(negative_valid_mask, float('-inf'))
        neg_logits = neg_sim_masked / self.temperature  # [B, N]
        
        # Combine: logits = [mean_pos_logit, neg_logits]
        # InfoNCE: -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
        all_logits = torch.cat([mean_pos_logit.unsqueeze(1), neg_logits], dim=1)  # [B, 1+N]
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)  # Positive is index 0
        
        loss = F.cross_entropy(all_logits, labels, reduction='mean')
        
        # Compute accuracy
        with torch.no_grad():
            predictions = all_logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean()
            avg_pos_sim = pos_sim.masked_fill(positive_valid_mask, 0.0).sum() / (~positive_valid_mask).sum().float().clamp(min=1.0)
            avg_neg_sim = neg_sim.masked_fill(negative_valid_mask, 0.0).sum() / (~negative_valid_mask).sum().float().clamp(min=1.0)
        
        return loss, accuracy, avg_pos_sim, avg_neg_sim

    def _load_pretrained_classifier(self, model=None):
        """Load pre-trained tone classifier from checkpoint"""
        import os
        if not self.tone_classifier_ckpt or not os.path.exists(self.tone_classifier_ckpt):
            print(f"[ToneRepulsiveCriterion] Warning: Classifier checkpoint not found: {self.tone_classifier_ckpt}")
            return
        
        try:
            print(f"[ToneRepulsiveCriterion] Loading pre-trained classifier from {self.tone_classifier_ckpt}")
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
                    print(f"[ToneRepulsiveCriterion] Successfully loaded tone classifier (embed_dim={embed_dim})")
                    
                    # If model is provided, convert dtype immediately
                    if model is not None:
                        device = next(model.parameters()).device
                        dtype = next(model.parameters()).dtype
                        self.tone_classifier = self.tone_classifier.to(device=device, dtype=dtype)
                        print(f"[ToneRepulsiveCriterion] Converted classifier to device={device}, dtype={dtype}")
                else:
                    print(f"[ToneRepulsiveCriterion] Warning: No tone_classifier found in checkpoint")
            else:
                print(f"[ToneRepulsiveCriterion] Warning: No criterion state in checkpoint")
        except Exception as e:
            print(f"[ToneRepulsiveCriterion] Error loading classifier: {e}")

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
        
        print(f"[ToneRepulsiveCriterion] Initialized tone classifier: {embed_dim} -> {self.num_tone_classes} (dtype={dtype})")
    
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
        print(f"[ToneRepulsiveCriterion] ✓ Frozen tone classifier: {frozen_count} parameters set to requires_grad=False")
        
        # Verify freezing
        for name, param in self.tone_classifier.named_parameters():
            if param.requires_grad:
                print(f"[ToneRepulsiveCriterion] ⚠️  WARNING: {name} still has requires_grad=True!")
            else:
                print(f"[ToneRepulsiveCriterion]   ✓ {name}: requires_grad=False")
    
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
                    print(f"[ToneRepulsiveCriterion] Frozen first {self.freeze_encoder_layers} encoder layers")
    
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
                'hard_negative_audio': [B, H, T],  # Same word, different tone
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
        # Note: We initialize even if lambda_classification=0 because we might have a frozen pre-trained classifier
        if self.lambda_classification > 0 or self.tone_classifier_ckpt:
            self._maybe_init_tone_classifier(model)
            self._maybe_freeze_tone_classifier()  # Freeze classifier if requested
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

        # Contrastive loss: combine all negatives (hard + soft) into one set
        if self.lambda_contrastive > 0 and 'positive_audio' in net_input:
            # Extract positive embeddings
            B, P, T = net_input['positive_audio'].shape
            pos_audio_flat = net_input['positive_audio'].view(-1, T)
            pos_mask = net_input.get('positive_padding_mask')
            pos_mask_flat = pos_mask.view(-1, T) if pos_mask is not None else None

            pos_emb_flat = self.extract_embedding(model, pos_audio_flat, pos_mask_flat)
            pos_emb = pos_emb_flat.view(B, P, -1)
            pos_valid_mask = net_input.get('positive_valid_mask', torch.zeros(B, P, dtype=torch.bool, device=anchor_emb.device))

            # Combine hard and soft negatives
            all_negatives = []
            all_neg_valid_masks = []
            
            if 'hard_negative_audio' in net_input:
                B, H, T = net_input['hard_negative_audio'].shape
                hard_neg_flat = net_input['hard_negative_audio'].view(-1, T)
                hard_neg_mask = net_input.get('hard_negative_padding_mask')
                hard_neg_mask_flat = hard_neg_mask.view(-1, T) if hard_neg_mask is not None else None
                
                hard_neg_emb_flat = self.extract_embedding(model, hard_neg_flat, hard_neg_mask_flat)
                hard_neg_emb = hard_neg_emb_flat.view(B, H, -1)
                all_negatives.append(hard_neg_emb)
                
                hard_neg_valid_mask = net_input.get('hard_negative_valid_mask', torch.zeros(B, H, dtype=torch.bool, device=anchor_emb.device))
                all_neg_valid_masks.append(hard_neg_valid_mask)
            
            if 'soft_negative_audio' in net_input:
                B, S, T = net_input['soft_negative_audio'].shape
                soft_neg_flat = net_input['soft_negative_audio'].view(-1, T)
                soft_neg_mask = net_input.get('soft_negative_padding_mask')
                soft_neg_mask_flat = soft_neg_mask.view(-1, T) if soft_neg_mask is not None else None
                
                soft_neg_emb_flat = self.extract_embedding(model, soft_neg_flat, soft_neg_mask_flat)
                soft_neg_emb = soft_neg_emb_flat.view(B, S, -1)
                all_negatives.append(soft_neg_emb)
                
                soft_neg_valid_mask = net_input.get('soft_negative_valid_mask', torch.zeros(B, S, dtype=torch.bool, device=anchor_emb.device))
                all_neg_valid_masks.append(soft_neg_valid_mask)
            
            # Concatenate all negatives
            if all_negatives:
                combined_negatives = torch.cat(all_negatives, dim=1)  # [B, H+S, C]
                combined_neg_valid_mask = torch.cat(all_neg_valid_masks, dim=1)  # [B, H+S]
                
                # Compute InfoNCE loss
                loss_contrastive, acc_contrastive, avg_pos_sim, avg_neg_sim = self.contrastive_loss(
                    anchor_emb, pos_emb, combined_negatives, pos_valid_mask, combined_neg_valid_mask
                )
                
                total_loss = total_loss + self.lambda_contrastive * loss_contrastive
                logs['loss_contrastive'] = loss_contrastive.data
                logs['acc_contrastive'] = acc_contrastive.data
                logs['avg_pos_sim'] = avg_pos_sim.data
                logs['avg_neg_sim'] = avg_neg_sim.data

        # Classification loss (supervised tone prediction)
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
        for key in ["loss_contrastive", "loss_classification", "acc_contrastive",
                    "avg_pos_sim", "avg_neg_sim", "tone_accuracy"]:
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
            print(f"[ToneRepulsiveCriterion] Loaded tone classifier from checkpoint")
        
        # Remove tone_classifier from state_dict before calling super
        # (since parent class doesn't expect it)
        state_dict_copy = {k: v for k, v in state_dict.items() if k != 'tone_classifier'}
        super().load_state_dict(state_dict_copy, strict=False)

