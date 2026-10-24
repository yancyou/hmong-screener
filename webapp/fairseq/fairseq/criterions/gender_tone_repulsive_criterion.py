# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.tone_repulsive_criterion import ToneRepulsiveCriterion
from fairseq.criterions.tone_repulsive_criterion_hinge import ToneRepulsiveHingeCriterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics
from fairseq.logging.meters import safe_round
from fairseq.data.data_utils import post_process


class _GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for speaker-adversarial baselines.
    Forward: identity
    Backward: multiply gradient by -lambda_
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return _GradReverse.apply(x, lambda_)


@dataclass
class GenderToneRepulsiveCriterionConfig(FairseqDataclass):
    # ===== Loss Weights =====
    alpha: float = field(
        default=1.0,
        metadata={"help": "weight for gender contrastive loss"},
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "weight for tone repulsive loss (includes all tone sub-losses)"},
    )
    delta: float = field(
        default=0.0,
        metadata={"help": "weight for auxiliary ASR CTC loss (0 disables)"},
    )

    # ===== Gender Contrastive Settings =====
    temperature_gender: float = field(
        default=0.07,
        metadata={"help": "temperature for gender InfoNCE"},
    )

    # ===== Tone Repulsive Settings =====
    # These are passed to ToneRepulsiveCriterion or ToneRepulsiveHingeCriterion
    use_tone_hinge_loss: bool = field(
        default=False,
        metadata={"help": "use Hinge Loss version for tone (separate hard/soft negatives with explicit margins)"},
    )
    tone_lambda_contrastive: float = field(
        default=1.0,
        metadata={"help": "[InfoNCE only] weight for tone contrastive loss"},
    )
    tone_lambda_attract: float = field(
        default=1.0,
        metadata={"help": "[Hinge only] weight for attractive loss (pull same tone together)"},
    )
    tone_lambda_repulse_hard: float = field(
        default=1.5,
        metadata={"help": "[Hinge only] weight for hard repulsive loss (same word, different tone)"},
    )
    tone_lambda_repulse_soft: float = field(
        default=0.3,
        metadata={"help": "[Hinge only] weight for soft repulsive loss (different words)"},
    )
    tone_margin_hard: float = field(
        default=-0.3,
        metadata={"help": "[Hinge only] target margin for hard negatives (want cos_sim < this)"},
    )
    tone_margin_soft: float = field(
        default=-0.2,
        metadata={"help": "[Hinge only] target margin for soft negatives (want cos_sim < this)"},
    )
    tone_lambda_classification: float = field(
        default=0.0,
        metadata={"help": "weight for tone classification loss within tone stream"},
    )
    tone_temperature: float = field(
        default=0.07,
        metadata={"help": "temperature for tone InfoNCE / attractive loss"},
    )
    tone_num_classes: int = field(
        default=7,
        metadata={"help": "number of tone classes"},
    )
    tone_freeze_classifier: bool = field(
        default=False,
        metadata={"help": "freeze tone classifier (use pre-trained)"},
    )
    tone_classifier_ckpt: str = field(
        default="",
        metadata={"help": "path to pre-trained tone classifier checkpoint"},
    )

    # ===== ASR CTC Settings =====
    asr_ctc_head_ckpt: str = field(
        default="",
        metadata={"help": "path to finetuned ASR checkpoint to load CTC head from"},
    )
    asr_ctc_head_freeze: bool = field(
        default=True,
        metadata={"help": "if true, keep loaded CTC head weights frozen (no grads)"},
    )

    # ===== Knowledge Distillation (optional) =====
    kd_teacher_ckpt: str = field(
        default="",
        metadata={"help": "path to teacher ASR checkpoint for KD (encoder+CTC head)"},
    )
    kd_weight: float = field(
        default=0.0,
        metadata={"help": "weight for KD loss (0 disables)"},
    )
    kd_temperature: float = field(
        default=2.0,
        metadata={"help": "temperature T for KD (softmax/log-softmax)"},
    )

    # ===== Shared Settings =====
    pooling_mode: str = field(
        default="mean",
        metadata={"help": "pooling over time: mean, max, or last"},
    )
    embedding_layer: int = field(
        default=-1,
        metadata={"help": "layer index for extract_features (-1=last)"},
    )
    embedding_extraction_layer: int = field(
        default=-1,
        metadata={"help": "layer index for gender/tone embedding extraction (-1=use all layers, 0-23=specific layer). ASR always uses all layers."},
    )
    freeze_encoder_layers: int = field(
        default=0,
        metadata={"help": "freeze first N encoder layers (0 = no freezing)"},
    )
    freeze_entire_encoder: bool = field(
        default=False,
        metadata={"help": "if true, freeze ALL encoder components (CNN, projections, all layers, layer_norm). Overrides freeze_encoder_layers."},
    )
    freeze_feature_extractor: bool = field(
        default=False,
        metadata={"help": "when freezing encoder layers, also freeze feature_extractor + post_extract_proj"},
    )

    # ===== Speaker-adversarial baseline (GRL) =====
    speaker_adv_weight: float = field(
        default=0.0,
        metadata={"help": "weight for speaker-adversarial loss via GRL (0 disables)"},
    )
    speaker_grl_lambda: float = field(
        default=1.0,
        metadata={"help": "GRL lambda (gradient multiplier, typically 1.0)"},
    )
    speaker_adv_hidden_dim: int = field(
        default=256,
        metadata={"help": "hidden dim for speaker adversarial MLP head"},
    )
    speaker_adv_num_speakers: int = field(
        default=0,
        metadata={"help": "number of speakers for adversarial head (0 => infer from GenderContrastiveDataset)"},
    )
    speaker_adv_emb_dim: int = field(
        default=768,
        metadata={"help": "embedding dim for speaker adversarial head (768 for base, 1024 for large)"},
    )


@register_criterion("gender_tone_repulsive", dataclass=GenderToneRepulsiveCriterionConfig)
class GenderToneRepulsiveCriterion(FairseqCriterion):
    """
    Multi-stream joint training criterion combining:
    1. Gender contrastive learning (InfoNCE)
    2. Tone contrastive learning (InfoNCE - simplified, no repulsive margins)
    3. ASR CTC loss (optional, frozen head)

    Total loss = α * L_gender + β * L_tone + δ * L_ctc

    Where L_tone internally combines:
      - Contrastive loss (standard InfoNCE with positives and negatives)
      - Classification loss (optional supervised tone prediction)
    """

    def __init__(
        self,
        task,
        alpha: float = 1.0,
        beta: float = 1.0,
        delta: float = 0.0,
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
        asr_ctc_head_ckpt: str = "",
        asr_ctc_head_freeze: bool = True,
        pooling_mode: str = "mean",
        embedding_layer: int = -1,
        embedding_extraction_layer: int = -1,
        freeze_encoder_layers: int = 0,
        freeze_entire_encoder: bool = False,
        freeze_feature_extractor: bool = False,
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
        
        # Stream weights
        self.alpha = alpha  # Gender
        self.beta = beta    # Tone
        self.delta = delta  # ASR

        # Gender settings
        self.temperature_gender = temperature_gender
        
        # Shared settings
        self.pooling_mode = pooling_mode
        self.embedding_layer = embedding_layer
        self.embedding_extraction_layer = embedding_extraction_layer
        self.freeze_encoder_layers = freeze_encoder_layers
        self.freeze_entire_encoder = freeze_entire_encoder
        self.freeze_feature_extractor = freeze_feature_extractor
        self._layers_frozen = False

        # ASR CTC settings
        self.asr_ctc_head_ckpt = asr_ctc_head_ckpt
        self.asr_ctc_head_freeze = asr_ctc_head_freeze
        self._head_installed = False
        self._ctc_head_frozen = False

        # KD settings
        self.kd_teacher_ckpt = kd_teacher_ckpt
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self._teacher_model = None  # lazily created on first forward

        # ===== Speaker-adversarial (GRL) baseline =====
        self.speaker_adv_weight = speaker_adv_weight
        self.speaker_grl_lambda = speaker_grl_lambda
        self.speaker_adv_hidden_dim = int(speaker_adv_hidden_dim)
        self.speaker_adv_num_speakers = int(speaker_adv_num_speakers)
        self.speaker_adv_emb_dim = int(speaker_adv_emb_dim)
        # Build speaker classifier NOW so optimizer includes it
        self.speaker_classifier = None
        if self.speaker_adv_weight > 0 and self.speaker_adv_num_speakers > 0:
            self.speaker_classifier = nn.Sequential(
                nn.Linear(self.speaker_adv_emb_dim, self.speaker_adv_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.speaker_adv_hidden_dim, self.speaker_adv_num_speakers),
            )
            print(f"  ✅ Speaker classifier created: {self.speaker_adv_emb_dim} -> {self.speaker_adv_hidden_dim} -> {self.speaker_adv_num_speakers}")

        # ===== Create Tone Repulsive Sub-Criterion =====
        # Choose between InfoNCE and Hinge Loss versions
        if use_tone_hinge_loss:
            print("[GenderToneRepulsiveCriterion] Using Hinge Loss for tone (separate hard/soft negatives)")
            self.tone_criterion = ToneRepulsiveHingeCriterion(
                task=task,
                temperature=tone_temperature,
                margin_hard=tone_margin_hard,
                margin_soft=tone_margin_soft,
                lambda_attract=tone_lambda_attract,
                lambda_repulse_hard=tone_lambda_repulse_hard,
                lambda_repulse_soft=tone_lambda_repulse_soft,
                lambda_classification=tone_lambda_classification,
                num_tone_classes=tone_num_classes,
                freeze_tone_classifier=tone_freeze_classifier,
                tone_classifier_ckpt=tone_classifier_ckpt,
                pooling_mode=pooling_mode,
                freeze_encoder_layers=0,  # We handle freezing at this level
            )
        else:
            print("[GenderToneRepulsiveCriterion] Using InfoNCE for tone (combined negatives)")
            self.tone_criterion = ToneRepulsiveCriterion(
                task=task,
                temperature=tone_temperature,
                lambda_contrastive=tone_lambda_contrastive,
                lambda_classification=tone_lambda_classification,
                num_tone_classes=tone_num_classes,
                freeze_tone_classifier=tone_freeze_classifier,
                tone_classifier_ckpt=tone_classifier_ckpt,
                pooling_mode=pooling_mode,
                freeze_encoder_layers=0,  # We handle freezing at this level
            )
        
        # Override tone_criterion's extract_embedding to use our layer-specific version
        self.tone_criterion.extract_embedding = lambda model, audio, mask=None: self.extract_embedding(model, audio, mask, use_embedding_layer=True)

        print(f"[GenderToneRepulsiveCriterion] Initialized with:")
        print(f"  α (gender weight): {self.alpha}")
        print(f"  β (tone weight): {self.beta}")
        print(f"  δ (ASR weight): {self.delta}")
        if use_tone_hinge_loss:
            print(f"  Tone sub-losses (Hinge): attract={tone_lambda_attract}, "
                  f"repulse_hard={tone_lambda_repulse_hard}, repulse_soft={tone_lambda_repulse_soft}, "
                  f"classification={tone_lambda_classification}")
            print(f"  Tone margins: hard={tone_margin_hard}, soft={tone_margin_soft}")
        else:
            print(f"  Tone sub-losses (InfoNCE): contrastive={tone_lambda_contrastive}, "
                  f"classification={tone_lambda_classification}")
        if self.kd_weight > 0 and self.kd_teacher_ckpt:
            print(f"  KD enabled: weight={self.kd_weight}, T={self.kd_temperature}, teacher={self.kd_teacher_ckpt}")
        
        # Layer-specific embedding extraction
        if self.embedding_extraction_layer >= 0:
            print(f"  ✅ Gender/Tone embeddings: from layer {self.embedding_extraction_layer} (early layers)")
            print(f"  ✅ ASR: uses ALL 24 layers (full model)")
        else:
            print(f"  ℹ️  All tasks use full model output (no layer-specific extraction)")

        if self.speaker_adv_weight > 0:
            print(f"  ✅ Speaker-adversarial (GRL) enabled: weight={self.speaker_adv_weight}, grl_lambda={self.speaker_grl_lambda}")

    def _infer_num_speakers(self):
        if self.speaker_adv_num_speakers and self.speaker_adv_num_speakers > 0:
            return int(self.speaker_adv_num_speakers)
        # Try to infer from the loaded gender dataset
        ds = getattr(self.task, "_gender_ds", None)
        if ds is not None and hasattr(ds, "speaker_to_id"):
            try:
                return int(len(ds.speaker_to_id))
            except Exception:
                return 0
        return 0

    def _maybe_init_speaker_classifier(self, emb_dim, device, labels_max=None):
        if self._speaker_classifier is not None:
            return
        num_speakers = self._infer_num_speakers()
        if (not num_speakers) and labels_max is not None:
            try:
                num_speakers = int(labels_max) + 1
            except Exception:
                num_speakers = 0
        if not num_speakers or num_speakers <= 1:
            # Not enough info to build a meaningful classifier
            return
        self._speaker_classifier = nn.Sequential(
            nn.Linear(int(emb_dim), int(self.speaker_adv_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.speaker_adv_hidden_dim), int(num_speakers)),
        ).to(device)

    def extract_embedding(self, model, audio_tensor, padding_mask=None, use_embedding_layer=True):
        """Extract and pool embeddings from encoder
        
        Args:
            model: Wav2Vec2 model
            audio_tensor: Input audio [B, T]
            padding_mask: Padding mask [B, T]
            use_embedding_layer: If True, use self.embedding_extraction_layer for extraction
                                If False, use all layers (for ASR)
        
        Returns:
            pooled: Pooled embeddings [B, C]
        """
        if hasattr(model, 'w2v_encoder'):
            # Use w2v_encoder's forward method which handles layer extraction correctly
            encoder_out = model.w2v_encoder(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
                layer=self.embedding_extraction_layer if (use_embedding_layer and self.embedding_extraction_layer >= 0) else None
            )
            
            # encoder_out is a dict with keys: 'encoder_out', 'padding_mask', 'layer_results'
            # encoder_out['encoder_out'] is [T, B, C]
            embeddings = encoder_out['encoder_out'].transpose(0, 1)  # [B, T, C]
            padding_mask = encoder_out.get('padding_mask', padding_mask)
        elif hasattr(model, 'extract_features'):
            target_layer = self.embedding_layer if use_embedding_layer else None
            result = model.extract_features(
                source=audio_tensor,
                padding_mask=padding_mask,
                mask=False,
                layer=target_layer
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

    def gender_contrastive_loss(self, anchor_emb, positive_emb, negative_emb):
        """
        Compute gender contrastive loss (InfoNCE).
        
        Args:
            anchor_emb: [B, C] - female embeddings
            positive_emb: [B, C] - male same-word embeddings
            negative_emb: [B, N, C] - male different-word embeddings
        
        Returns:
            loss, accuracy
        """
        batch_size = anchor_emb.size(0)
        num_negatives = negative_emb.size(1)

        # Normalize
        anchor_emb = F.normalize(anchor_emb, dim=-1)  # [B, C]
        positive_emb = F.normalize(positive_emb, dim=-1)  # [B, C]
        negative_emb = F.normalize(negative_emb, dim=-1)  # [B, N, C]

        # Positive similarities
        pos_sim = torch.sum(anchor_emb * positive_emb, dim=-1)  # [B]
        pos_logits = pos_sim / self.temperature_gender  # [B]

        # Negative similarities
        neg_sim = torch.bmm(anchor_emb.unsqueeze(1), negative_emb.transpose(1, 2)).squeeze(1)  # [B, N]
        neg_logits = neg_sim / self.temperature_gender  # [B, N]

        # InfoNCE loss
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # [B, 1+N]
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # Positive is index 0

        loss = F.cross_entropy(logits, labels, reduction='mean')

        # Accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean()

        return loss, accuracy

    def _freeze_entire_encoder(self, model):
        """Freeze ALL encoder components (CNN, projections, all transformer layers, layer_norm)
        
        NOTE: Does NOT freeze w2v_encoder.proj (CTC head) - that should be trained separately
        """
        if not (hasattr(model, 'w2v_encoder') and hasattr(model.w2v_encoder, 'w2v_model')):
            return
        
        w2v_model = model.w2v_encoder.w2v_model
        
        # Handle nested Wav2VecCtc structure (model loading via w2v_path creates nested structure)
        if hasattr(w2v_model, 'w2v_encoder') and hasattr(w2v_model.w2v_encoder, 'w2v_model'):
            print(f"[GenderToneRepulsiveCriterion] Detected nested Wav2VecCtc in _freeze_entire_encoder, going one level deeper")
            w2v_model = w2v_model.w2v_encoder.w2v_model
        
        frozen_components = []
        
        # Freeze feature extractor (CNN)
        if hasattr(w2v_model, 'feature_extractor'):
            for param in w2v_model.feature_extractor.parameters():
                param.requires_grad = False
            frozen_components.append("feature_extractor")
        
        # Freeze post_extract_proj
        if hasattr(w2v_model, 'post_extract_proj') and w2v_model.post_extract_proj is not None:
            for param in w2v_model.post_extract_proj.parameters():
                param.requires_grad = False
            frozen_components.append("post_extract_proj")
        
        # Freeze ALL transformer encoder layers
        if hasattr(w2v_model, 'encoder') and hasattr(w2v_model.encoder, 'layers'):
            encoder = w2v_model.encoder
            total_layers = len(encoder.layers)
            
            # Freeze positional convolution (CRITICAL: was missing!)
            if hasattr(encoder, 'pos_conv') and encoder.pos_conv is not None:
                for param in encoder.pos_conv.parameters():
                    param.requires_grad = False
                frozen_components.append("encoder.pos_conv")
            
            for i in range(total_layers):
                for param in encoder.layers[i].parameters():
                    param.requires_grad = False
            
            frozen_components.append(f"encoder.layers[0:{total_layers}] (ALL)")
            
            # Freeze encoder's layer_norm
            if hasattr(encoder, 'layer_norm') and encoder.layer_norm is not None:
                for param in encoder.layer_norm.parameters():
                    param.requires_grad = False
                frozen_components.append("encoder.layer_norm")
        
        # NOTE: w2v_encoder.proj (CTC head) is NOT frozen here
        # It should be controlled separately by asr_ctc_head_freeze parameter
        
        print(f"[GenderToneRepulsiveCriterion] ✅ FULLY FROZEN encoder (embeddings): {', '.join(frozen_components)}")
        print(f"[GenderToneRepulsiveCriterion] 🔓 w2v_encoder.proj (CTC head) remains TRAINABLE")

    def _freeze_encoder_layers_only(self, model, num_layers):
        """Freeze only the first N transformer encoder layers (partial freezing)"""
        print(
            f"[GenderToneRepulsiveCriterion] _freeze_encoder_layers_only called with "
            f"num_layers={num_layers}, freeze_feature_extractor={self.freeze_feature_extractor}"
        )
        
        if not (hasattr(model, 'w2v_encoder') and hasattr(model.w2v_encoder, 'w2v_model')):
            print(f"[GenderToneRepulsiveCriterion] ERROR: Model structure incorrect - missing w2v_encoder or w2v_model")
            return
        
        w2v_model = model.w2v_encoder.w2v_model
        print(f"[GenderToneRepulsiveCriterion] Found w2v_model: {type(w2v_model).__name__}")
        
        # Handle nested Wav2VecCtc structure (model loading via w2v_path creates nested structure)
        if hasattr(w2v_model, 'w2v_encoder') and hasattr(w2v_model.w2v_encoder, 'w2v_model'):
            print(f"[GenderToneRepulsiveCriterion] Detected nested Wav2VecCtc, going one level deeper")
            w2v_model = w2v_model.w2v_encoder.w2v_model
            print(f"[GenderToneRepulsiveCriterion] Now using inner w2v_model: {type(w2v_model).__name__}")
        
        # Optionally freeze feature extractor + projection
        if self.freeze_feature_extractor:
            if hasattr(w2v_model, 'feature_extractor'):
                for param in w2v_model.feature_extractor.parameters():
                    param.requires_grad = False
                print("[GenderToneRepulsiveCriterion] 🔒 Frozen: feature_extractor (CNN)")
            if hasattr(w2v_model, 'post_extract_proj') and w2v_model.post_extract_proj is not None:
                for param in w2v_model.post_extract_proj.parameters():
                    param.requires_grad = False
                print("[GenderToneRepulsiveCriterion] 🔒 Frozen: post_extract_proj")
            # 🔧 FIX: Also freeze w2v_model.layer_norm (input layer norm)
            if hasattr(w2v_model, 'layer_norm') and w2v_model.layer_norm is not None:
                for param in w2v_model.layer_norm.parameters():
                    param.requires_grad = False
                print("[GenderToneRepulsiveCriterion] 🔒 Frozen: w2v_model.layer_norm (input)")
        
        # Freeze only transformer encoder layers
        if hasattr(w2v_model, 'encoder') and hasattr(w2v_model.encoder, 'layers'):
            encoder = w2v_model.encoder
            total_layers = len(encoder.layers)
            layers_to_freeze = min(num_layers, total_layers)
            
            print(f"[GenderToneRepulsiveCriterion] Found {total_layers} encoder layers, will freeze first {layers_to_freeze}")
            
            # 🔧 FIX: Freeze encoder.pos_conv (position encoding affects ALL layers!)
            if self.freeze_feature_extractor and hasattr(encoder, 'pos_conv') and encoder.pos_conv is not None:
                for param in encoder.pos_conv.parameters():
                    param.requires_grad = False
                print("[GenderToneRepulsiveCriterion] 🔒 Frozen: encoder.pos_conv")
            
            # Freeze first N layers
            for i in range(layers_to_freeze):
                for param in encoder.layers[i].parameters():
                    param.requires_grad = False
            
            # Ensure remaining layers are trainable (important!)
            for i in range(layers_to_freeze, total_layers):
                for param in encoder.layers[i].parameters():
                    param.requires_grad = True
            
            # 🔧 FIX: Freeze encoder.layer_norm if freezing most layers
            # This layer_norm is applied after all transformer layers
            if self.freeze_feature_extractor and hasattr(encoder, 'layer_norm') and encoder.layer_norm is not None:
                for param in encoder.layer_norm.parameters():
                    param.requires_grad = False
                print("[GenderToneRepulsiveCriterion] 🔒 Frozen: encoder.layer_norm")
            
            print(f"[GenderToneRepulsiveCriterion] 🔒 Frozen layers: 0-{layers_to_freeze-1} ({layers_to_freeze}/{total_layers} layers)")
            print(f"[GenderToneRepulsiveCriterion] 🔓 Trainable layers: {layers_to_freeze}-{total_layers-1} ({total_layers-layers_to_freeze} layers)")
            if not self.freeze_feature_extractor:
                print(f"[GenderToneRepulsiveCriterion] 🔓 Also trainable: feature_extractor, post_extract_proj, layer_norm, pos_conv")
        else:
            print(f"[GenderToneRepulsiveCriterion] ERROR: Could not find encoder.layers in w2v_model")
    
    def _maybe_freeze_encoder_layers(self, model):
        """Freeze encoder layers or entire encoder based on config"""
        if self._layers_frozen:
            return
        
        # Option 1: Freeze entire encoder (CNN + all layers + projections)
        if self.freeze_entire_encoder:
            print(f"[GenderToneRepulsiveCriterion] Freezing entire encoder (freeze_entire_encoder=True)")
            self._freeze_entire_encoder(model)
            self._layers_frozen = True
            return
        
        # Option 2: Freeze only first N transformer layers
        if self.freeze_encoder_layers > 0:
            print(f"[GenderToneRepulsiveCriterion] Freezing first {self.freeze_encoder_layers} encoder layers")
            self._freeze_encoder_layers_only(model, self.freeze_encoder_layers)
            self._layers_frozen = True
            return
        
        # No freezing - only print once
        print(f"[GenderToneRepulsiveCriterion] No layer freezing applied (all layers trainable)")
        self._layers_frozen = True  # Set to True to avoid repeated printing

    def _maybe_install_ctc_head(self, model):
        """Load and optionally freeze ASR CTC head from checkpoint"""
        if self._head_installed or not self.asr_ctc_head_ckpt:
            return

        import os
        if not os.path.exists(self.asr_ctc_head_ckpt):
            print(f"[GenderToneRepulsiveCriterion] Warning: ASR checkpoint not found: {self.asr_ctc_head_ckpt}")
            return

        try:
            print(f"[GenderToneRepulsiveCriterion] Loading ASR CTC head from {self.asr_ctc_head_ckpt}")
            checkpoint = torch.load(self.asr_ctc_head_ckpt, map_location='cpu', weights_only=False)

            if 'model' in checkpoint:
                model_state = checkpoint['model']
                # Extract CTC head weights
                ctc_keys = [k for k in model_state.keys() if 'w2v_encoder.proj' in k]
                if ctc_keys:
                    ctc_state = {k: model_state[k] for k in ctc_keys}
                    model.load_state_dict(ctc_state, strict=False)
                    print(f"[GenderToneRepulsiveCriterion] Loaded {len(ctc_keys)} CTC head parameters")

                    # Freeze if requested
                    if self.asr_ctc_head_freeze:
                        if hasattr(model, 'w2v_encoder') and hasattr(model.w2v_encoder, 'proj'):
                            frozen_count = 0
                            for param in model.w2v_encoder.proj.parameters():
                                param.requires_grad = False
                                frozen_count += 1
                            print(f"[GenderToneRepulsiveCriterion] ✓ Frozen ASR CTC head: {frozen_count} parameters set to requires_grad=False")
                            
                            # Verify freezing
                            for name, param in model.w2v_encoder.proj.named_parameters():
                                if param.requires_grad:
                                    print(f"[GenderToneRepulsiveCriterion] ⚠️  WARNING: {name} still has requires_grad=True!")
                                else:
                                    print(f"[GenderToneRepulsiveCriterion]   ✓ {name}: requires_grad=False")
                else:
                    print(f"[GenderToneRepulsiveCriterion] Warning: No CTC head found in checkpoint")
            else:
                print(f"[GenderToneRepulsiveCriterion] Warning: No model state in checkpoint")

        except Exception as e:
            print(f"[GenderToneRepulsiveCriterion] Error loading ASR head: {e}")

        self._head_installed = True

    def _maybe_freeze_ctc_head(self, model):
        """Freeze student's CTC head even if we do not load from external ckpt."""
        if not self.asr_ctc_head_freeze or self._ctc_head_frozen:
            return
        if hasattr(model, 'w2v_encoder') and hasattr(model.w2v_encoder, 'proj') and model.w2v_encoder.proj is not None:
            for param in model.w2v_encoder.proj.parameters():
                param.requires_grad = False
            self._ctc_head_frozen = True
            print("[GenderToneRepulsiveCriterion] ✓ Frozen ASR CTC head (student proj) by config")

    def _maybe_build_teacher(self, student_model):
        """Lazily build and load the teacher model for KD.

        We deep-copy the student structure to avoid rebuilding from task/model cfg,
        then load teacher weights. Teacher is placed on the same device, eval, and frozen.
        If shapes mismatch (e.g., different dictionary size), KD will be disabled.
        """
        if self._teacher_model is not None or not self.kd_teacher_ckpt or self.kd_weight <= 0:
            return

        import copy
        import os
        import torch

        if not os.path.exists(self.kd_teacher_ckpt):
            print(f"[GenderToneRepulsiveCriterion] KD disabled: teacher ckpt not found: {self.kd_teacher_ckpt}")
            self.kd_weight = 0.0
            return

        try:
            # Load teacher state first
            state = torch.load(self.kd_teacher_ckpt, map_location='cpu', weights_only=False)
            state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
            
            # Build teacher from checkpoint_utils instead of deepcopy (避免weight_norm问题)
            from fairseq import checkpoint_utils
            try:
                models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                    [self.kd_teacher_ckpt],
                    arg_overrides={"data": getattr(self.task, 'asr_data_dir', '.')},
                    strict=False,
                )
                self._teacher_model = models[0].eval()
                for p in self._teacher_model.parameters():
                    p.requires_grad = False
            except Exception as load_err:
                print(f"[GenderToneRepulsiveCriterion] Failed to load teacher via checkpoint_utils: {load_err}")
                print(f"[GenderToneRepulsiveCriterion] Falling back to manual state_dict loading...")
                # Fallback: manually reconstruct (假设student和teacher架构相同)
                import torch.nn as nn
                self._teacher_model = type(student_model)(student_model.w2v_encoder.cfg, len(self.task.target_dictionary))
                self._teacher_model.load_state_dict(state_dict, strict=False)
                self._teacher_model.eval()
                for p in self._teacher_model.parameters():
                    p.requires_grad = False

            # Move to student device and finalize
            device = next(student_model.parameters()).device
            self._teacher_model.to(device)
            self._teacher_model.eval()
            print(f"[GenderToneRepulsiveCriterion] Teacher model loaded for KD from {self.kd_teacher_ckpt}")
        except Exception as e:
            print(f"[GenderToneRepulsiveCriterion] KD disabled: failed to load teacher -> {e}")
            self._teacher_model = None
            self.kd_weight = 0.0

    def forward(self, model, sample, reduce=True):
        """
        Expected sample structure:
        {
            'net_input': {
                # Gender stream
                'female_audio': [B, T],
                'male_positive_audio': [B, T],
                'male_negative_audio': [B, N, T],
                'female_padding_mask': [B, T],
                'male_positive_padding_mask': [B, T],
                'male_negative_padding_mask': [B, N, T],
                
                # Tone stream (repulsive format)
                'anchor_audio': [B, T],
                'positive_audio': [B, P, T],
                'hard_negative_audio': [B, H, T],
                'soft_negative_audio': [B, S, T],
                'anchor_padding_mask': [B, T],
                'positive_padding_mask': [B, P, T],
                'hard_negative_padding_mask': [B, H, T],
                'soft_negative_padding_mask': [B, S, T],
                'positive_valid_mask': [B, P],
                'hard_negative_valid_mask': [B, H],
                'soft_negative_valid_mask': [B, S],
                'anchor_tone_label': [B],  # For classification
                
                # ASR stream (optional)
                'asr_audio': [B, T],
                'asr_padding_mask': [B, T],
                'asr_target': [B, L],
                'asr_target_lengths': [B],
            }
        }
        """
        # Apply layer freezing and CTC head loading (once)
        self._maybe_freeze_encoder_layers(model)
        if self.delta > 0:
            self._maybe_install_ctc_head(model)

        net_input = sample['net_input']
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        total_size = 0
        logs = {}

        # ===== Stream 1: Gender Contrastive (and/or Speaker-adversarial) =====
        # Run if: (1) gender loss enabled (alpha>0), OR (2) speaker-adversarial enabled
        need_gender_stream = (self.alpha > 0) or (self.speaker_adv_weight > 0 and self.speaker_classifier is not None)
        if need_gender_stream and 'female_audio' in net_input:
            # Extract embeddings (always needed for gender or speaker-adversarial)
            female_emb = self.extract_embedding(
                model, net_input['female_audio'], net_input.get('female_padding_mask')
            )
            male_pos_emb = self.extract_embedding(
                model, net_input['male_positive_audio'], net_input.get('male_positive_padding_mask')
            )
            B = female_emb.size(0)
            total_size += B

            # Compute gender loss (only if alpha > 0)
            if self.alpha > 0:
                # Handle negative embeddings (only needed for gender contrastive)
                _, N, T = net_input['male_negative_audio'].shape
            male_neg_flat = net_input['male_negative_audio'].view(-1, T)
            male_neg_mask = net_input.get('male_negative_padding_mask')
            male_neg_mask_flat = male_neg_mask.view(-1, T) if male_neg_mask is not None else None
            
            male_neg_emb_flat = self.extract_embedding(model, male_neg_flat, male_neg_mask_flat)
            male_neg_emb = male_neg_emb_flat.view(B, N, -1)

            loss_gender, acc_gender = self.gender_contrastive_loss(female_emb, male_pos_emb, male_neg_emb)
            total_loss = total_loss + self.alpha * loss_gender
            logs.update({
                'loss_gender': loss_gender.data,
                'acc_gender': acc_gender.data,
            })

            # Optional: Speaker-adversarial baseline (GRL)
            if self.speaker_adv_weight and self.speaker_adv_weight > 0 and self.speaker_classifier is not None:
                female_sid = net_input.get('female_speaker_id', None)
                male_sid = net_input.get('male_positive_speaker_id', None)
                if female_sid is not None and male_sid is not None:
                    # Concatenate embeddings/labels from both sides
                    spk_emb = torch.cat([female_emb, male_pos_emb], dim=0)
                    spk_y = torch.cat([female_sid, male_sid], dim=0).to(spk_emb.device)
                    valid = (spk_y >= 0)
                    logs['speaker_adv_n'] = valid.sum().detach()
                    if valid.any():
                        # Move classifier to correct device if needed
                        if next(self.speaker_classifier.parameters()).device != spk_emb.device:
                            self.speaker_classifier = self.speaker_classifier.to(spk_emb.device)
                        logits = self.speaker_classifier(grad_reverse(spk_emb, self.speaker_grl_lambda))
                        loss_spk = F.cross_entropy(logits[valid], spk_y[valid].long(), reduction='mean')
                        total_loss = total_loss + float(self.speaker_adv_weight) * loss_spk
                        logs['loss_speaker_adv'] = loss_spk.data
                        with torch.no_grad():
                            pred = logits[valid].argmax(dim=-1)
                            logs['acc_speaker_adv'] = (pred == spk_y[valid].long()).float().mean()

        # ===== Stream 2: Tone Repulsive =====
        if self.beta > 0 and 'anchor_audio' in net_input:
            # Create a sub-sample for tone criterion
            tone_sample = {'net_input': {k: v for k, v in net_input.items() 
                                         if k.startswith('anchor_') or k.startswith('positive_') or 
                                            k.startswith('hard_negative_') or k.startswith('soft_negative_')}}
            
            # Call tone repulsive criterion
            loss_tone, batch_size_tone, tone_logs = self.tone_criterion.forward(model, tone_sample, reduce=reduce)
            
            total_loss = total_loss + self.beta * loss_tone
            total_size += batch_size_tone
            
            # Prefix tone logs
            for key, val in tone_logs.items():
                if key not in ['loss', 'ntokens', 'nsentences', 'sample_size']:
                    logs[f'tone_{key}'] = val

        # ===== Stream 3: ASR CTC + KD =====
        # Run ASR stream if: (1) delta>0 (CTC enabled), OR (2) kd_weight>0 (KD enabled)
        if (self.delta > 0 or self.kd_weight > 0) and 'asr_audio' in net_input:
            # Forward pass for ASR
            # Ensure CTC head installed/frozen per config
            self._maybe_install_ctc_head(model)
            self._maybe_freeze_ctc_head(model)
            net_out_asr = model(
                source=net_input['asr_audio'],
                padding_mask=net_input.get('asr_padding_mask'),
                mask=False,
            )
            # student logits (unnormalized) for KD; log_probs for CTC
            student_logits = model.get_logits(net_out_asr, normalize=False)  # [T, B, C]
            log_probs = torch.log_softmax(student_logits.float(), dim=-1)

            if net_out_asr.get("padding_mask") is not None:
                nonpad = (~net_out_asr["padding_mask"]).long()
                input_lengths = nonpad.sum(-1)  # [B]
            else:
                input_lengths = log_probs.new_full(
                    (log_probs.size(1),), log_probs.size(0), dtype=torch.long
                )

            targets = net_input['asr_target']  # [B, L]
            target_lengths = net_input['asr_target_lengths']  # [B]

            # CTC loss (only if delta > 0)
            if self.delta > 0:
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
            else:
                # delta=0: 不计算CTC loss，但total_size仍需统计
                total_size += targets.size(0)

            # ===== KD (optional) =====
            if self.kd_weight > 0:
                self._maybe_build_teacher(model)
                if self._teacher_model is not None:
                    with torch.no_grad():
                        net_out_teacher = self._teacher_model(
                            source=net_input['asr_audio'],
                            padding_mask=net_input.get('asr_padding_mask'),
                            mask=False,
                        )
                        teacher_logits = self._teacher_model.get_logits(net_out_teacher, normalize=False)

                    T = float(self.kd_temperature)
                    # padding mask: [B, T]; convert to [T, B]
                    pad = net_out_asr.get("padding_mask")
                    if pad is not None:
                        valid = (~pad).T  # [T, B]
                        stu_flat = student_logits[valid]
                        tea_flat = teacher_logits[valid]
                    else:
                        stu_flat = student_logits.view(-1, student_logits.size(-1))
                        tea_flat = teacher_logits.view(-1, teacher_logits.size(-1))

                    # KL(student || teacher)
                    s_logp = F.log_softmax(stu_flat / T, dim=-1)
                    t_prob = F.softmax(tea_flat / T, dim=-1)
                    kd_loss = F.kl_div(s_logp, t_prob, reduction='batchmean') * (T * T)
                    total_loss = total_loss + self.kd_weight * kd_loss
                    logs.update({"loss_kd": kd_loss.data, "kd_on": torch.tensor(1.0, device=kd_loss.device)})
                else:
                    logs.update({"kd_on": torch.tensor(0.0, device=student_logits.device)})
            else:
                # KD disabled by config
                logs.update({"kd_on": torch.tensor(0.0, device=student_logits.device)})

            # ===== WER/CER logging during validation (no grad) =====
            if not model.training:
                import editdistance
                with torch.no_grad():
                    # log_probs: [T,B,C] (log-softmax)
                    lprobs_t = log_probs.transpose(0, 1).float().contiguous().cpu()  # [B,T,C]
                    w_errs = 0
                    w_len = 0
                    c_err = 0
                    c_len = 0
                    B = lprobs_t.size(0)
                    for i in range(B):
                        lp = lprobs_t[i]
                        inp_l = int(input_lengths[i].item()) if i < input_lengths.size(0) else lp.size(0)
                        lp = lp[:inp_l]

                        # Targets
                        targ = targets[i, : int(target_lengths[i].item())]
                        # Char-level
                        toks = lp.argmax(dim=-1).unique_consecutive()
                        pred_units_arr = [t for t in toks.tolist() if t != 0]  # blank=0
                        targ_units_arr = targ.tolist()
                        c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                        c_len += len(targ_units_arr)

                        # Word-level via post_process("letter")
                        targ_units = self.task.target_dictionary.string(targ)
                        pred_units = self.task.target_dictionary.string(torch.tensor(pred_units_arr))
                        targ_words = post_process(targ_units, "letter").split()
                        pred_words = post_process(pred_units, "letter").split()
                        w_errs += editdistance.eval(pred_words, targ_words)
                        w_len += len(targ_words)

                    logs.update({
                        "_w_errors": w_errs,
                        "_w_total": w_len,
                        "_c_errors": c_err,
                        "_c_total": c_len,
                    })

        if total_size == 0:
            # Nothing provided; return zero loss
            dummy = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
            logging_output = {"loss": dummy.data, "sample_size": 0}
            return dummy, 0, logging_output

        logging_output = {
            "loss": total_loss.data,
            "sample_size": total_size,
        }
        logging_output.update(logs)
        return total_loss, total_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs"""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        # Log KD-specific info
        kd_on_sum = sum(log.get("kd_on", 0) for log in logging_outputs)
        kd_on_count = len([log for log in logging_outputs if "kd_on" in log])
        if kd_on_count > 0:
            metrics.log_scalar("kd_on", kd_on_sum / kd_on_count, kd_on_count, round=2)

        # Log all component metrics
        # Note: Hinge loss metrics are prefixed with "tone_" in forward()
        for key in ["loss_gender", "acc_gender", "loss_ctc", "loss_kd",
                    # InfoNCE tone metrics
                    "tone_loss_contrastive", "tone_loss_classification", 
                    "tone_acc_contrastive", "tone_avg_pos_sim", "tone_avg_neg_sim", 
                    "tone_tone_accuracy",
                    # Hinge loss tone metrics (with tone_ prefix)
                    "tone_loss_attract", "tone_loss_repulse_hard", "tone_loss_repulse_soft",
                    "tone_avg_hard_neg_sim", "tone_avg_soft_neg_sim",
                    "tone_hard_violation_rate", "tone_soft_violation_rate"]:
            vals = [log.get(key, None) for log in logging_outputs if key in log]
            if vals:
                metrics.log_scalar(key, sum(vals) / len(vals), len(vals), round=4)

        # KD on ratio (0..1)
        kd_vals = [log.get("kd_on", None) for log in logging_outputs if "kd_on" in log]
        if kd_vals:
            metrics.log_scalar("kd_on", sum(kd_vals) / len(kd_vals), len(kd_vals), round=4)

        # Aggregate WER/CER if present
        w_errors = sum(log.get("_w_errors", 0) for log in logging_outputs)
        w_total = sum(log.get("_w_total", 0) for log in logging_outputs)
        c_errors = sum(log.get("_c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("_c_total", 0) for log in logging_outputs)
        if w_total > 0:
            metrics.log_scalar("_w_errors", w_errors)
            metrics.log_scalar("_w_total", w_total)
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
        if c_total > 0:
            metrics.log_scalar("_c_errors", c_errors)
            metrics.log_scalar("_c_total", c_total)
            metrics.log_derived(
                "cer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )

        # Composite validation metric (小值更好): 0.6*WER + 0.2*(1-acc_gender) + 0.2*(1-tone_acc)
        def _composite(meters):
            try:
                wer = (meters["_w_errors"].sum * 1.0 / meters["_w_total"].sum) if meters.get("_w_total") else float("nan")
                acc_g = meters["acc_gender"].avg if meters.get("acc_gender") else float("nan")
                acc_t = meters["tone_tone_accuracy"].avg if meters.get("tone_tone_accuracy") else float("nan")
                # 归一化到 [0,1]
                asr_term = wer  # already ratio
                g_term = 1.0 - acc_g if isinstance(acc_g, float) else 1.0
                t_term = 1.0 - acc_t if isinstance(acc_t, float) else 1.0
                val = 0.6 * asr_term + 0.2 * g_term + 0.2 * t_term
                return safe_round(val * 100.0, 3)  # 转为百分比便于观察
            except Exception:
                return float("nan")

        metrics.log_derived("val_composite", _composite)

        # Also expose a pure total validation loss (same units as 'loss' meter)
        metrics.log_derived(
            "val_total",
            lambda meters: meters["loss"].avg if meters.get("loss") else float("nan"),
        )

        # Unified best metric: if KD effectively on and ASR WER available, use composite; else use total loss
        def _best_val(meters):
            try:
                kd_on = meters["kd_on"].avg if meters.get("kd_on") else 0.0
                # Prefer composite only when KD is active and WER present
                has_wer = meters.get("_w_total") and meters["_w_total"].sum > 0
                if kd_on and has_wer:
                    # reuse composite computation
                    wer = (meters["_w_errors"].sum * 1.0 / meters["_w_total"].sum) if meters.get("_w_total") else float("nan")
                    acc_g = meters["acc_gender"].avg if meters.get("acc_gender") else float("nan")
                    acc_t = meters["tone_tone_accuracy"].avg if meters.get("tone_tone_accuracy") else float("nan")
                    asr_term = wer
                    g_term = 1.0 - acc_g if isinstance(acc_g, float) else 1.0
                    t_term = 1.0 - acc_t if isinstance(acc_t, float) else 1.0
                    val = 0.6 * asr_term + 0.2 * g_term + 0.2 * t_term
                    return safe_round(val * 100.0, 3)
                # fallback to total validation loss
                return meters["loss"].avg if meters.get("loss") else float("nan")
            except Exception:
                return float("nan")

        metrics.log_derived("best_val", _best_val)

        # No-KD best metric: prioritize val_total, lightly penalize WER/CER if available
        def _best_nokd(meters):
            try:
                val_loss = meters["loss"].avg if meters.get("loss") else float("nan")
                if not isinstance(val_loss, float):
                    return float("nan")
                has_wer = meters.get("_w_total") and meters["_w_total"].sum > 0
                has_cer = meters.get("_c_total") and meters["_c_total"].sum > 0
                if has_wer:
                    wer_ratio = (meters["_w_errors"].sum * 1.0 / meters["_w_total"].sum)
                else:
                    wer_ratio = 0.0
                if has_cer:
                    cer_ratio = (meters["_c_errors"].sum * 1.0 / meters["_c_total"].sum)
                else:
                    cer_ratio = 0.0
                # Smaller is better; loss dominates, WER/CER act as mild penalties
                return safe_round(val_loss * (1.0 + 0.3 * wer_ratio + 0.1 * cer_ratio), 6)
            except Exception:
                return float("nan")

        metrics.log_derived("best_nokd", _best_nokd)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Save criterion state including tone classifier"""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Save tone criterion state (includes tone_classifier)
        state['tone_criterion'] = self.tone_criterion.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load criterion state including tone classifier"""
        # Load tone criterion state
        if 'tone_criterion' in state_dict:
            self.tone_criterion.load_state_dict(state_dict['tone_criterion'], strict=False)
            print(f"[GenderToneRepulsiveCriterion] Loaded tone criterion state")

        # Remove tone_criterion from state_dict before calling super
        state_dict_copy = {k: v for k, v in state_dict.items() if k != 'tone_criterion'}
        super().load_state_dict(state_dict_copy, strict=False)
