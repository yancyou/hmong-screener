# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round

# Import our layer freezing utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
try:
    from freeze_layers_hook import freeze_encoder_layers, count_trainable_parameters
except ImportError:
    print("Warning: Could not import freeze_layers_hook. Layer freezing will be disabled.")


@dataclass
class ToneContrastiveCriterionConfig(FairseqDataclass):
    temperature: float = field(
        default=0.1,
        metadata={"help": "temperature parameter for contrastive loss"}
    )
    num_negatives: int = field(
        default=20,
        metadata={"help": "number of negative samples for contrastive learning"}
    )
    pooling_mode: str = field(
        default="mean",
        metadata={"help": "pooling mode for sequence embeddings: mean, max, or last"}
    )
    embedding_layer: int = field(
        default=-1,
        metadata={"help": "which layer to extract embeddings from (-1 for last layer)"}
    )
    positive_weight: float = field(
        default=1.0,
        metadata={"help": "weight for positive samples in multi-positive contrastive loss"}
    )
    freeze_encoder_layers: int = field(
        default=0,
        metadata={"help": "number of encoder layers to freeze from the beginning (0 = no freezing)"}
    )


@register_criterion("tone_contrastive", dataclass=ToneContrastiveCriterionConfig)
class ToneContrastiveCriterion(FairseqCriterion):
    def __init__(self, task, temperature=0.1, num_negatives=20, pooling_mode="mean", 
                 embedding_layer=-1, positive_weight=1.0, freeze_encoder_layers=0):
        super().__init__(task)
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.pooling_mode = pooling_mode
        self.freeze_encoder_layers = freeze_encoder_layers
        self._layers_frozen = False
        self.embedding_layer = embedding_layer
        self.positive_weight = positive_weight

    def extract_embedding(self, model, audio_tensor, padding_mask=None):
        """Extract embeddings from wav2vec2 model"""
        if hasattr(model, 'w2v_encoder'):
            # For fine-tuned models (Wav2VecCtc)
            encoder_out = model.w2v_encoder(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,  # No masking during inference
                features_only=True
            )
            embeddings = encoder_out['encoder_out'].transpose(0, 1) # [T, B, C] -> [B, T, C]
            padding_mask = encoder_out['padding_mask'] # Use the new down-sampled padding mask
        elif hasattr(model, 'extract_features'):
            # For base wav2vec2 models
            result = model.extract_features(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                layer=self.embedding_layer
            )
            embeddings = result['x']  # [B, T, C]
        else:
            raise ValueError("Model must have either w2v_encoder or extract_features method")
        
        # Pool sequence embeddings
        if padding_mask is not None:
            # Mask out padding positions
            embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            lengths = (~padding_mask).sum(dim=1).float()  # [B]
        else:
            lengths = torch.full((embeddings.size(0),), embeddings.size(1), 
                               dtype=torch.float, device=embeddings.device)
        
        if self.pooling_mode == "mean":
            # Mean pooling excluding padding
            pooled = embeddings.sum(dim=1) / lengths.unsqueeze(-1)  # [B, C]
        elif self.pooling_mode == "max":
            # Max pooling
            pooled = embeddings.max(dim=1)[0]  # [B, C]
        elif self.pooling_mode == "last":
            # Last non-padding position
            last_indices = (lengths - 1).long()  # [B]
            pooled = embeddings[torch.arange(embeddings.size(0)), last_indices]  # [B, C]
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
        
        return pooled

    def compute_multi_positive_contrastive_loss(self, anchor_embeddings, positive_embeddings, 
                                              negative_embeddings, positive_valid_mask):
        """
        Compute simplified multi-positive InfoNCE contrastive loss for tone learning
        
        The key insight: All positive tone variants should have higher similarity to the anchor
        than any negative sample. We use a simplified approach that averages positive similarities
        and contrasts them against negative similarities.
        
        Args:
            anchor_embeddings: [B, C] - anchor tone embeddings
            positive_embeddings: [B, P, C] - positive tone variant embeddings (same base word)
            negative_embeddings: [B, N, C] - negative embeddings (different base words)
            positive_valid_mask: [B, P] - mask for valid positives (False=valid, True=padding)
        """
        batch_size, max_positives, embed_dim = positive_embeddings.shape
        num_negatives = negative_embeddings.shape[1]
        
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)  # [B, C]
        positive_embeddings = F.normalize(positive_embeddings, dim=-1)  # [B, P, C]
        negative_embeddings = F.normalize(negative_embeddings, dim=-1)  # [B, N, C]
        
        # Compute positive similarities
        anchor_expanded = anchor_embeddings.unsqueeze(1)  # [B, 1, C]
        pos_sim = torch.sum(anchor_expanded * positive_embeddings, dim=-1)  # [B, P]
        
        # Mask out invalid positives and compute mean positive similarity per sample
        pos_sim_masked = pos_sim.masked_fill(positive_valid_mask, 0.0)  # Zero out invalid
        valid_pos_count = (~positive_valid_mask).sum(dim=1, keepdim=True).float()  # [B, 1]
        valid_pos_count = torch.clamp(valid_pos_count, min=1.0)  # Avoid division by zero
        mean_pos_sim = pos_sim_masked.sum(dim=1, keepdim=True) / valid_pos_count  # [B, 1]
        
        # Compute negative similarities
        neg_sim = torch.bmm(
            anchor_embeddings.unsqueeze(1),  # [B, 1, C]
            negative_embeddings.transpose(1, 2)  # [B, C, N]
        ).squeeze(1)  # [B, N]
        
        # Combine mean positive similarity with all negative similarities
        # Each sample: [mean_pos_sim, neg_sim_1, neg_sim_2, ..., neg_sim_N]
        all_sim = torch.cat([mean_pos_sim, neg_sim], dim=1)  # [B, 1 + N]
        all_sim = all_sim / self.temperature  # Apply temperature
        
        # Target is always position 0 (the mean positive similarity)
        targets = torch.zeros(batch_size, dtype=torch.long, device=all_sim.device)
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(all_sim, targets, reduction='mean')
        
        # Compute accuracy: is mean positive similarity highest?
        with torch.no_grad():
            predictions = all_sim.argmax(dim=1)
            accuracy = (predictions == targets).float().mean()
            
        return loss, accuracy, batch_size

    def forward(self, model, sample, reduce=True):
        """
        Compute the tone contrastive loss
        
        Expected sample format:
        {
            'net_input': {
                'anchor_audio': [B, T],  # Anchor audio tensors
                'positive_audio': [B, P, T],  # Positive tone variants (same base word)
                'negative_audio': [B, N, T],  # Negative audio (different base words)
                'anchor_padding_mask': [B, T],  # Optional padding masks
                'positive_padding_mask': [B, P, T],
                'negative_padding_mask': [B, N, T],
                'positive_valid_mask': [B, P],  # Which positives are valid
            },
            'base_word_labels': [B],  # Base word labels for logging
            'ntokens': int,  # Number of tokens
            'nsentences': int,  # Batch size
        }
        """
        # Apply layer freezing if specified (only once)
        if self.freeze_encoder_layers > 0 and not self._layers_frozen:
            try:
                freeze_encoder_layers(model, self.freeze_encoder_layers)
                count_trainable_parameters(model)
                self._layers_frozen = True
                print(f"✅ Applied layer freezing: first {self.freeze_encoder_layers} layers frozen")
            except Exception as e:
                print(f"⚠️  Warning: Could not apply layer freezing: {e}")
        
        net_input = sample['net_input']
        
        # Extract embeddings for anchor audio
        anchor_embeddings = self.extract_embedding(
            model, 
            net_input['anchor_audio'],
            net_input.get('anchor_padding_mask')
        )
        
        # Extract embeddings for positive audio (tone variants)
        batch_size, max_positives, seq_len = net_input['positive_audio'].shape
        positive_audio_flat = net_input['positive_audio'].view(-1, seq_len)
        
        positive_padding_mask = net_input.get('positive_padding_mask')
        if positive_padding_mask is not None:
            positive_padding_mask_flat = positive_padding_mask.view(-1, seq_len)
        else:
            positive_padding_mask_flat = None
            
        positive_embeddings_flat = self.extract_embedding(
            model,
            positive_audio_flat,
            positive_padding_mask_flat
        )
        
        # Reshape back to [B, P, C]
        embed_dim = positive_embeddings_flat.size(-1)
        positive_embeddings = positive_embeddings_flat.view(batch_size, max_positives, embed_dim)
        
        # Extract embeddings for negative audio (different base words)
        batch_size, num_negatives, seq_len = net_input['negative_audio'].shape
        negative_audio_flat = net_input['negative_audio'].view(-1, seq_len)
        
        negative_padding_mask = net_input.get('negative_padding_mask')
        if negative_padding_mask is not None:
            negative_padding_mask_flat = negative_padding_mask.view(-1, seq_len)
        else:
            negative_padding_mask_flat = None
            
        negative_embeddings_flat = self.extract_embedding(
            model,
            negative_audio_flat,
            negative_padding_mask_flat
        )
        
        # Reshape back to [B, N, C]
        negative_embeddings = negative_embeddings_flat.view(batch_size, num_negatives, embed_dim)
        
        # Get valid positive mask
        positive_valid_mask = net_input.get('positive_valid_mask')
        if positive_valid_mask is None:
            # If not provided, assume all positives are valid
            positive_valid_mask = torch.zeros(batch_size, max_positives, 
                                            dtype=torch.bool, device=anchor_embeddings.device)
        
        # Compute multi-positive contrastive loss
        loss, accuracy, sample_size = self.compute_multi_positive_contrastive_loss(
            anchor_embeddings,
            positive_embeddings, 
            negative_embeddings,
            positive_valid_mask
        )
        
        if not reduce:
            loss = loss / sample_size if sample_size > 0 else loss
        
        logging_output = {
            "loss": loss.data,
            "ntokens": sample.get("ntokens", sample_size),
            "nsentences": sample.get("nsentences", sample_size),
            "sample_size": sample_size,
            "accuracy": accuracy.data if isinstance(accuracy, torch.Tensor) else accuracy,
        }
        
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        accuracy_sum = sum(log.get("accuracy", 0) for log in logging_outputs)
        
        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar("accuracy", accuracy_sum / sample_size, sample_size, round=3)
        
        if sample_size != sum(log.get("ntokens", 0) for log in logging_outputs):
            metrics.log_scalar("nll_loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
