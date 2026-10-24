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
class GenderContrastiveCriterionConfig(FairseqDataclass):
    temperature: float = field(
        default=0.1,
        metadata={"help": "temperature parameter for contrastive loss"}
    )
    num_negatives: int = field(
        default=100,
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
    freeze_encoder_layers: int = field(
        default=0,
        metadata={"help": "number of encoder layers to freeze from the beginning (0 = no freezing)"}
    )


@register_criterion("gender_contrastive", dataclass=GenderContrastiveCriterionConfig)
class GenderContrastiveCriterion(FairseqCriterion):
    def __init__(self, task, temperature=0.1, num_negatives=100, pooling_mode="mean", embedding_layer=-1, freeze_encoder_layers=0):
        super().__init__(task)
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.pooling_mode = pooling_mode
        self.embedding_layer = embedding_layer
        self.freeze_encoder_layers = freeze_encoder_layers
        self._layers_frozen = False

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

    def compute_contrastive_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        """
        Compute InfoNCE contrastive loss
        
        Args:
            anchor_embeddings: [B, C] - female word embeddings
            positive_embeddings: [B, C] - corresponding male word embeddings (same word)
            negative_embeddings: [B, N, C] - different male word embeddings (different words)
        """
        batch_size = anchor_embeddings.size(0)
        
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, dim=-1)
        negative_embeddings = F.normalize(negative_embeddings, dim=-1)
        
        # Compute positive similarities
        pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=-1)  # [B]
        pos_sim = pos_sim / self.temperature
        
        # Compute negative similarities
        neg_sim = torch.bmm(
            anchor_embeddings.unsqueeze(1),  # [B, 1, C]
            negative_embeddings.transpose(1, 2)  # [B, C, N]
        ).squeeze(1)  # [B, N]
        neg_sim = neg_sim / self.temperature
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
        
        # Targets are always 0 (first position is positive)
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, targets, reduction='sum')
        
        # Compute accuracy (how often positive is ranked first)
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == 0).float().sum()
        
        return loss, accuracy, batch_size

    def forward(self, model, sample, reduce=True):
        """
        Compute the gender contrastive loss
        
        Expected sample format:
        {
            'net_input': {
                'female_audio': [B, T],  # Female audio tensors
                'male_positive_audio': [B, T],  # Corresponding male audio (same word)
                'male_negative_audio': [B, N, T],  # Different male audio (different words)
                'female_padding_mask': [B, T],  # Optional padding masks
                'male_positive_padding_mask': [B, T],
                'male_negative_padding_mask': [B, N, T],
            },
            'word_labels': [B],  # Word labels for logging
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
        
        # Extract embeddings for female audio (anchors)
        female_embeddings = self.extract_embedding(
            model, 
            net_input['female_audio'],
            net_input.get('female_padding_mask')
        )
        
        # Extract embeddings for male positive audio (same words)
        male_positive_embeddings = self.extract_embedding(
            model,
            net_input['male_positive_audio'],
            net_input.get('male_positive_padding_mask')
        )
        
        # Extract embeddings for male negative audio (different words)
        batch_size, num_negatives, seq_len = net_input['male_negative_audio'].shape
        male_negative_audio_flat = net_input['male_negative_audio'].view(-1, seq_len)
        
        male_negative_padding_mask = net_input.get('male_negative_padding_mask')
        if male_negative_padding_mask is not None:
            male_negative_padding_mask_flat = male_negative_padding_mask.view(-1, seq_len)
        else:
            male_negative_padding_mask_flat = None
            
        male_negative_embeddings_flat = self.extract_embedding(
            model,
            male_negative_audio_flat,
            male_negative_padding_mask_flat
        )
        
        # Reshape back to [B, N, C]
        embed_dim = male_negative_embeddings_flat.size(-1)
        male_negative_embeddings = male_negative_embeddings_flat.view(batch_size, num_negatives, embed_dim)
        
        # Compute contrastive loss
        loss, accuracy, sample_size = self.compute_contrastive_loss(
            female_embeddings,
            male_positive_embeddings, 
            male_negative_embeddings
        )
        
        if not reduce:
            loss = loss / sample_size
        
        logging_output = {
            "loss": loss.data,
            "ntokens": sample.get("ntokens", sample_size),
            "nsentences": sample.get("nsentences", sample_size),
            "sample_size": sample_size,
            "accuracy": accuracy.data,
        }
        
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        accuracy_sum = sum(log.get("accuracy", 0) for log in logging_outputs)
        
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