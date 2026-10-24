#!/usr/bin/env python3
"""
Layer Freezing Hook for Wav2Vec2 Models
This script provides utilities to freeze specific layers in wav2vec2 models during training.
"""

import torch
import torch.nn as nn
from typing import Optional
import re


def freeze_encoder_layers(model, freeze_layers: int):
    """
    Freeze the first N encoder layers of a wav2vec2 model.
    
    Args:
        model: The wav2vec2 model (should have w2v_encoder attribute)
        freeze_layers: Number of encoder layers to freeze from the beginning
    """
    if not hasattr(model, 'w2v_encoder'):
        print("Warning: Model doesn't have w2v_encoder attribute. Skipping layer freezing.")
        return
        
    w2v_model = model.w2v_encoder.w2v_model
    
    # Get total number of encoder layers
    if hasattr(w2v_model, 'encoder') and hasattr(w2v_model.encoder, 'layers'):
        total_layers = len(w2v_model.encoder.layers)
        print(f"Found {total_layers} encoder layers in wav2vec2 model")
        
        if freeze_layers >= total_layers:
            print(f"Warning: freeze_layers ({freeze_layers}) >= total_layers ({total_layers})")
            freeze_layers = total_layers - 1
            
        # Freeze the first N layers
        for i in range(freeze_layers):
            layer = w2v_model.encoder.layers[i]
            for param in layer.parameters():
                param.requires_grad = False
            print(f"Frozen encoder layer {i}")
            
        # Ensure remaining layers are trainable
        for i in range(freeze_layers, total_layers):
            layer = w2v_model.encoder.layers[i]
            for param in layer.parameters():
                param.requires_grad = True
            print(f"Trainable encoder layer {i}")
            
        print(f"✅ Successfully froze first {freeze_layers} layers, training last {total_layers - freeze_layers} layers")
        
    else:
        print("Warning: Could not find encoder layers in model structure")


def freeze_layers_by_regex(model, pattern: str):
    """
    Freeze model parameters matching a regex pattern.
    
    Args:
        model: The model to freeze parameters in
        pattern: Regex pattern to match parameter names
    """
    frozen_params = []
    trainable_params = []
    
    for name, param in model.named_parameters():
        if re.search(pattern, name):
            param.requires_grad = False
            frozen_params.append(name)
        else:
            param.requires_grad = True
            trainable_params.append(name)
            
    print(f"Frozen {len(frozen_params)} parameters matching pattern '{pattern}':")
    for name in frozen_params[:5]:  # Show first 5
        print(f"  - {name}")
    if len(frozen_params) > 5:
        print(f"  ... and {len(frozen_params) - 5} more")
        
    print(f"Trainable {len(trainable_params)} parameters:")
    for name in trainable_params[:5]:  # Show first 5
        print(f"  - {name}")
    if len(trainable_params) > 5:
        print(f"  ... and {len(trainable_params) - 5} more")


def print_model_structure(model):
    """Print the structure of a wav2vec2 model to understand layer organization."""
    print("Model structure:")
    for name, module in model.named_modules():
        if 'encoder.layers' in name and len(name.split('.')) <= 4:
            print(f"  {name}: {type(module).__name__}")


def count_trainable_parameters(model):
    """Count and display trainable vs frozen parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"Parameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / total_params:.1f}%")
    
    return trainable_params, total_params


# Example freeze patterns for different scenarios
FREEZE_PATTERNS = {
    # Freeze first 18 layers (out of 24)
    "first_18_layers": r"w2v_encoder\.w2v_model\.encoder\.layers\.(?:[0-9]|1[0-7])\.",
    
    # Freeze first 20 layers (out of 24) 
    "first_20_layers": r"w2v_encoder\.w2v_model\.encoder\.layers\.(?:[0-9]|1[0-9])\.",
    
    # Freeze only feature extractor (CNN layers)
    "feature_extractor": r"w2v_encoder\.w2v_model\.feature_extractor\.",
    
    # Freeze feature extractor + first 12 transformer layers
    "feature_and_first_12": r"w2v_encoder\.w2v_model\.(feature_extractor|encoder\.layers\.(?:[0-9]|1[01]))\.",
    
    # Freeze everything except last 4 layers
    "all_except_last_4": r"w2v_encoder\.w2v_model\.(feature_extractor|encoder\.layers\.(?:[0-9]|1[0-9]))\.",
}


if __name__ == "__main__":
    print("Layer Freezing Utilities for Wav2Vec2 Models")
    print("Available freeze patterns:")
    for name, pattern in FREEZE_PATTERNS.items():
        print(f"  {name}: {pattern}")
