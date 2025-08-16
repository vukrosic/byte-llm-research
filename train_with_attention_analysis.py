#!/usr/bin/env python3
"""
Training script with integrated attention analysis
Tracks attention patterns during training and identifies optimization opportunities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from train import ModelConfig, MinimalLLM, TextByteDataset, set_seed
from attention_analyzer import AttentionAnalyzer

def create_attention_rich_dataset(size: int = 8000):
    """Create dataset that requires different types of attention patterns"""
    patterns = []
    
    # Patterns requiring local attention
    for _ in range(100):
        # Sequential patterns
        patterns.append(b"abcdefghijklmnopqrstuvwxyz" * 5)
        patterns.append(b"0123456789" * 10)
    
    # Patterns requiring long-range attention
    for _ in range(50):
        # Matching brackets/quotes
        patterns.append(b'start "this is a long quote with many words inside" end')
        patterns.append(b'begin (nested (structures) here) finish')
        patterns.append(b'if condition then do_something else do_other_thing endif')
    
    # Patterns that might not need attention (repetitive)
    for _ in range(100):
        patterns.append(b"hello world " * 20)
        patterns.append(b"test test test " * 15)
    
    # Mixed complexity patterns
    for _ in range(50):
        patterns.append(b"The quick brown fox jumps over the lazy dog. " * 3)
        patterns.append(b"Machine learning models process data efficiently. " * 2)
    
    # Convert to bytes
    all_bytes = []
    for pattern in patterns:
        all_bytes.extend(list(pattern))
    
    # Extend to desired size
    while len(all_bytes) < size:
        all_bytes.extend(all_bytes[:min(size - len(all_bytes), len(all_bytes))])
    
    return all_bytes[:size]

def train_with_attention_tracking():
    """Train model while tracking attention patterns"""
    print("🔍 Training with Attention Analysis")
    print("=" * 50)
    
    # Set seed
    set_seed(42)
    
    # Create config
    config = ModelConfig(
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        batch_size=12,
        max_steps=2000,
        max_seq_len=128,
        eval_every=400,  # Capture attention every 400 steps
        vocab_size=256
    )
    
    print(f"📋 Config: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    
    # Create dataset
    print("📚 Creating attention-rich dataset...")
    bytes_data = create_attention_rich_dataset(15000)
    dataset = TextByteDataset(bytes_data, config.max_seq_len)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Initialize model
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🔧 Device: {device}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize attention analyzer
    analyzer = AttentionAnalyzer(save_dir="training_attention_analysis")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps)
    
    # Get a sample batch for attention analysis
    sample_batch = next(iter(val_loader))
    sample_input = sample_batch[0][:1].to(device)  # Take first sample
    
    # Initial attention capture
    print(f"\n📸 Capturing initial attention patterns...")
    analyzer.capture_attention_patterns(model, sample_input, 0)
    
    # Training loop
    print(f"\n🚀 Starting training with attention tracking...")
    
    step = 0
    best_loss = float('inf')
    
    for epoch in range(20):
        model.train()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    lr = optimizer.param_groups[0]['lr']
                print(f"Step {step:4d}: Loss {loss.item():.4f}, Acc {accuracy:.3f}, LR {lr:.2e}")
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
            
            # Attention analysis
            if step % config.eval_every == 0 and step > 0:
                print(f"\n📸 Capturing attention patterns at step {step}...")
                analyzer.capture_attention_patterns(model, sample_input, step)
                analyzer.analyze_attention_patterns()
                
                # Visualize most and least useful heads
                if len(analyzer.attention_weights_history) >= 2:
                    # Visualize head 0,0 and head 1,3 as examples
                    analyzer.visualize_attention_head(0, 0)
                    analyzer.visualize_attention_head(1, 3)
                
                print()
            
            step += 1
        
        if step >= config.max_steps:
            break
    
    # Final attention analysis
    print(f"\n📸 Final attention capture...")
    analyzer.capture_attention_patterns(model, sample_input, step)
    
    # Create comprehensive analysis
    print(f"\n🎨 Creating attention visualizations...")
    
    # Analyze all captured patterns
    for i in range(len(analyzer.step_history)):
        print(f"\n📊 Analysis for step {analyzer.step_history[i]}:")
        analyzer.analyze_attention_patterns(i)
    
    # Create evolution GIFs for interesting heads
    print(f"\n🎬 Creating attention evolution animations...")
    
    # Create GIFs for first few heads of each layer
    for layer in range(min(config.n_layers, 2)):  # First 2 layers
        for head in range(min(config.n_heads, 4)):  # First 4 heads
            try:
                analyzer.create_attention_evolution_gif(layer, head, duration=1.0)
            except Exception as e:
                print(f"⚠️  Could not create GIF for L{layer}H{head}: {e}")
    
    # Save all data
    analyzer.save_data()
    
    # Final comprehensive analysis
    print(f"\n" + "="*80)
    print("🔍 COMPREHENSIVE ATTENTION ANALYSIS")
    print("="*80)
    
    if len(analyzer.step_history) > 1:
        print(f"\n📈 ATTENTION EVOLUTION SUMMARY:")
        
        # Compare initial vs final attention patterns
        initial_entropies = analyzer.attention_entropy_history[0]
        final_entropies = analyzer.attention_entropy_history[-1]
        
        print(f"\n🎯 ATTENTION HEAD EVOLUTION:")
        for layer_idx in range(len(initial_entropies)):
            print(f"\nLayer {layer_idx}:")
            for head_idx in range(len(initial_entropies[layer_idx])):
                initial_entropy = initial_entropies[layer_idx][head_idx]
                final_entropy = final_entropies[layer_idx][head_idx]
                change = final_entropy - initial_entropy
                
                if abs(change) > 0.5:
                    direction = "📈 More diffuse" if change > 0 else "📉 More focused"
                    print(f"  Head {head_idx}: {initial_entropy:.2f} → {final_entropy:.2f} ({change:+.2f}) {direction}")
        
        # Identify patterns
        print(f"\n🔍 ATTENTION PATTERN INSIGHTS:")
        
        # Find heads that became very focused (low entropy)
        focused_heads = []
        diffuse_heads = []
        
        for layer_idx in range(len(final_entropies)):
            for head_idx in range(len(final_entropies[layer_idx])):
                entropy = final_entropies[layer_idx][head_idx]
                if entropy < 1.0:
                    focused_heads.append((layer_idx, head_idx, entropy))
                elif entropy > 2.5:
                    diffuse_heads.append((layer_idx, head_idx, entropy))
        
        if focused_heads:
            print(f"\n🎯 HIGHLY FOCUSED HEADS (potential for optimization):")
            for layer, head, entropy in sorted(focused_heads, key=lambda x: x[2]):
                print(f"  L{layer}H{head}: entropy={entropy:.2f} (very focused)")
        
        if diffuse_heads:
            print(f"\n🌊 HIGHLY DIFFUSE HEADS (potential for removal):")
            for layer, head, entropy in sorted(diffuse_heads, key=lambda x: x[2], reverse=True):
                print(f"  L{layer}H{head}: entropy={entropy:.2f} (very diffuse)")
        
        # Optimization recommendations
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        if focused_heads:
            print(f"  • {len(focused_heads)} heads are highly focused - consider sparse attention")
        
        if diffuse_heads:
            print(f"  • {len(diffuse_heads)} heads are very diffuse - consider removal or replacement")
        
        if len(focused_heads) > len(diffuse_heads):
            print(f"  • Model shows local attention patterns - sparse attention might work well")
        else:
            print(f"  • Model uses diverse attention patterns - full attention may be necessary")
    
    print(f"\n✅ Training with attention analysis complete!")
    print(f"📁 Results saved in: {analyzer.save_dir}/")
    print(f"🎬 Check the GIF files for attention evolution animations")
    print(f"📊 Check PNG files for attention pattern snapshots")

if __name__ == "__main__":
    train_with_attention_tracking()