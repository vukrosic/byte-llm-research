#!/usr/bin/env python3
"""
Training script focused on letter evolution analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from train import ModelConfig, MinimalLLM, TextByteDataset, set_seed
from letter_tracker import LetterTracker

def create_letter_rich_dataset(size: int = 10000):
    """Create dataset rich in letter patterns for better letter learning"""
    patterns = []
    
    # Common English words (lots of letter patterns)
    common_words = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
        "hello", "world", "computer", "science", "machine", "learning", "artificial", "intelligence", "language", "model", "training", "data", "algorithm", "neural", "network", "deep", "python", "programming", "software", "development", "technology", "innovation"
    ]
    
    # Create sentences with these words
    for _ in range(100):
        sentence = " ".join(np.random.choice(common_words, size=np.random.randint(5, 15)))
        patterns.append(sentence.encode('utf-8'))
    
    # Add alphabet sequences
    for _ in range(50):
        patterns.append(b"abcdefghijklmnopqrstuvwxyz " * 2)
        patterns.append(b"zyxwvutsrqponmlkjihgfedcba " * 2)
    
    # Add vowel-heavy and consonant-heavy patterns
    vowel_pattern = b"aeiouaeiouaeiou " * 20
    consonant_pattern = b"bcdfghjklmnpqrstvwxyz " * 10
    
    for _ in range(30):
        patterns.append(vowel_pattern)
        patterns.append(consonant_pattern)
    
    # Add common letter combinations
    combinations = [b"th", b"he", b"in", b"er", b"an", b"re", b"ed", b"nd", b"on", b"en", b"at", b"ou", b"it", b"is", b"or", b"ti", b"hi", b"st", b"io", b"le", b"ar", b"ne", b"as", b"to"]
    
    for combo in combinations:
        for _ in range(20):
            patterns.append((combo + b" ") * 50)
    
    # Combine all patterns
    all_bytes = []
    for pattern in patterns:
        all_bytes.extend(list(pattern))
    
    # Repeat to get desired size
    while len(all_bytes) < size:
        all_bytes.extend(all_bytes[:min(size - len(all_bytes), len(all_bytes))])
    
    return all_bytes[:size]

def train_with_letter_tracking():
    """Train model with focused letter tracking"""
    print("🔤 Training with Letter Evolution Tracking")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create config optimized for letter learning
    config = ModelConfig(
        d_model=128,     # Reasonable size for letter patterns
        n_heads=8,
        n_layers=4,
        d_ff=512,
        batch_size=16,
        max_steps=5000,  # Exactly 5000 steps as requested
        max_seq_len=256,
        eval_every=250,  # More frequent tracking
        vocab_size=256
    )
    
    print(f"📋 Config: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"🎯 Training for {config.max_steps} steps with letter tracking every {config.eval_every} steps")
    
    # Create letter-rich dataset
    print("📚 Creating letter-rich dataset...")
    bytes_data = create_letter_rich_dataset(20000)
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
    
    # Initialize letter tracker
    tracker = LetterTracker(save_dir="letter_5000_steps")
    tracker.capture_letters(model, 0)  # Initial state
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    
    step = 0
    best_loss = float('inf')
    
    for epoch in range(100):  # Max epochs
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
            
            # Letter tracking
            if step % config.eval_every == 0 and step > 0:
                print(f"\n📸 Capturing letter state at step {step}")
                tracker.capture_letters(model, step)
                tracker.print_letter_analysis(top_k=3)
                print()
            
            step += 1
        
        if step >= config.max_steps:
            break
    
    # Final capture
    print(f"\n📸 Final letter capture at step {step}")
    tracker.capture_letters(model, step)
    
    # Create all visualizations
    print(f"\n🎨 Creating letter evolution visualizations...")
    
    # Individual heatmaps for key steps
    key_steps = [0, len(tracker.step_history)//4, len(tracker.step_history)//2, 
                 3*len(tracker.step_history)//4, -1]
    
    for step_idx in key_steps:
        tracker.create_letter_heatmap(step_idx)
    
    # Create animated GIFs
    print("🎬 Creating letter evolution GIF...")
    tracker.create_letter_evolution_gif(duration=0.6)
    
    print("🎵 Creating vowel-consonant analysis GIF...")
    tracker.create_vowel_consonant_gif(duration=0.6)
    
    # Save data
    tracker.save_data()
    
    # Final analysis
    print(f"\n" + "="*70)
    print("📊 FINAL LETTER EVOLUTION ANALYSIS")
    print("="*70)
    
    print("\n🔤 INITIAL STATE (Step 0):")
    tracker.print_letter_analysis(0, top_k=5)
    
    print(f"\n🔤 FINAL STATE (Step {tracker.step_history[-1]}):")
    tracker.print_letter_analysis(-1, top_k=5)
    
    print(f"\n🔄 EVOLUTION SUMMARY:")
    tracker.compare_letter_evolution(0, -1)
    
    print(f"\n✅ Letter evolution analysis complete!")
    print(f"📁 Results saved in: {tracker.save_dir}/")
    print(f"🎬 Key files:")
    print(f"   - letter_evolution.gif (main animation)")
    print(f"   - vowel_consonant_evolution.gif (clustering analysis)")
    print(f"   - letter_heatmap_step_*.png (individual snapshots)")
    print(f"   - letter_data.pkl (raw data)")

if __name__ == "__main__":
    train_with_letter_tracking()