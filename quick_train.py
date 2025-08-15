#!/usr/bin/env python3
"""
Quick training script to generate a model for testing inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from train import ModelConfig, MinimalLLM, set_seed
import os

def create_test_model():
    """Create and save a minimal test model"""
    print("🔄 Creating test model for inference testing...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create minimal config
    config = ModelConfig(
        d_model=128,  # Smaller for quick testing
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=512,
        vocab_size=256
    )
    
    # Initialize model
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save the model (even untrained, for testing inference)
    model_save_path = "byte_llm_model.pt"
    print(f"💾 Saving test model to {model_save_path}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': {'val_loss': 999.0, 'val_accuracy': 0.0, 'val_perplexity': 999.0},
        'training_time': 0.0
    }, model_save_path)
    
    print(f"✅ Test model saved! You can now test inference with:")
    print(f"   python inference.py {model_save_path}")
    print(f"⚠️  Note: This is an untrained model, so output will be random.")

if __name__ == "__main__":
    create_test_model()