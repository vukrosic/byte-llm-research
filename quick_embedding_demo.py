#!/usr/bin/env python3
"""
Quick training demo to show embedding evolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from train import ModelConfig, MinimalLLM, TextByteDataset, set_seed
from embedding_tracker import EmbeddingTracker

def create_simple_dataset(size: int = 1000):
    """Create a simple dataset for quick training"""
    # Create some structured byte patterns
    patterns = [
        # Numbers
        b"0123456789" * 10,
        # Letters  
        b"abcdefghijklmnopqrstuvwxyz" * 10,
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10,
        # Common words
        b"the quick brown fox jumps over the lazy dog " * 20,
        b"hello world this is a test of byte level modeling " * 15,
        # Punctuation
        b".,!?;:()[]{}\"'-_/\\@#$%^&*+=<>|`~ " * 25,
    ]
    
    # Combine all patterns
    all_bytes = []
    for pattern in patterns:
        all_bytes.extend(list(pattern))
    
    # Repeat to get desired size
    while len(all_bytes) < size:
        all_bytes.extend(all_bytes[:min(size - len(all_bytes), len(all_bytes))])
    
    return all_bytes[:size]

def quick_train_demo():
    """Run a quick training demo to show embedding evolution"""
    print("🚀 Quick Embedding Evolution Demo")
    print("="*50)
    
    # Set seed
    set_seed(42)
    
    # Create minimal config
    config = ModelConfig(
        d_model=64,      # Small for quick training
        n_heads=4,
        n_layers=2,
        d_ff=256,
        batch_size=8,
        max_steps=500,   # Quick training
        max_seq_len=128,
        eval_every=100,
        vocab_size=256
    )
    
    print(f"📋 Config: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    
    # Create simple dataset
    bytes_data = create_simple_dataset(5000)
    dataset = TextByteDataset(bytes_data, config.max_seq_len)
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"📊 Dataset: {len(dataset)} samples")
    
    # Initialize model
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    print(f"🔧 Device: {device}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize embedding tracker
    tracker = EmbeddingTracker(save_dir="quick_demo_evolution")
    tracker.capture_embeddings(model, 0)
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\n🎯 Training for {config.max_steps} steps...")
    
    step = 0
    for epoch in range(10):  # Max epochs
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            if step % 50 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                print(f"Step {step:3d}: Loss {loss.item():.4f}, Acc {accuracy:.3f}")
            
            # Capture embeddings
            if step % 100 == 0 and step > 0:
                tracker.capture_embeddings(model, step)
                tracker.print_distance_summary(top_k=3)
            
            step += 1
        
        if step >= config.max_steps:
            break
    
    # Final capture
    tracker.capture_embeddings(model, step)
    
    print(f"\n🎨 Creating visualizations...")
    tracker.create_all_heatmaps()
    tracker.create_animated_gif(duration=1.0)  # Slower animation
    tracker.save_data()
    
    # Show evolution
    if len(tracker.step_history) > 1:
        print(f"\n📈 FINAL EVOLUTION ANALYSIS")
        tracker.compare_steps(0, -1)
    
    print(f"\n✅ Demo complete! Check '{tracker.save_dir}' for:")
    print(f"   - Individual heatmaps (.png files)")
    print(f"   - Animated evolution (embedding_evolution.gif)")
    print(f"   - Raw data (embedding_data.pkl)")

if __name__ == "__main__":
    quick_train_demo()