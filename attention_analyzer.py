#!/usr/bin/env python3
"""
Attention Mechanism Analysis and Optimization
Analyzes what attention computations are actually useful and tests efficient alternatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os
from typing import List, Dict, Tuple, Optional
import pickle
from dataclasses import dataclass

class AttentionAnalyzer:
    def __init__(self, save_dir: str = "attention_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for attention patterns
        self.attention_weights_history = []  # [step][layer][head][seq_len, seq_len]
        self.attention_entropy_history = []  # Entropy of attention distributions
        self.attention_sparsity_history = []  # How sparse are attention patterns
        self.step_history = []
        
        print(f"🔍 AttentionAnalyzer initialized, saving to {save_dir}")
    
    def capture_attention_patterns(self, model, input_ids: torch.Tensor, step: int):
        """Capture attention patterns from all layers and heads"""
        model.eval()
        attention_weights = []
        
        with torch.no_grad():
            x = model.token_embedding(input_ids) * (model.config.d_model ** 0.5)
            x = model.position_dropout(x)
            
            for layer_idx, block in enumerate(model.transformer_blocks):
                # Hook into attention to capture weights
                attention_layer = block.attention
                
                batch_size, seq_len = x.size(0), x.size(1)
                qkv = attention_layer.qkv(x).reshape(batch_size, seq_len, 3, attention_layer.n_heads, attention_layer.d_k)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                Q, K, V = qkv[0], qkv[1], qkv[2]
                
                # Apply rotary embeddings
                Q = attention_layer.rotary(Q)
                K = attention_layer.rotary(K)
                
                # Compute attention weights manually
                scores = torch.matmul(Q, K.transpose(-2, -1)) / (attention_layer.d_k ** 0.5)
                
                # Apply causal mask
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                scores.masked_fill_(causal_mask, float('-inf'))
                
                # Get attention weights
                attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, seq_len]
                
                attention_weights.append(attn_weights.cpu().numpy())
                
                # Continue forward pass
                attn_output = torch.matmul(attn_weights, V)
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, attention_layer.d_model)
                x = x + block.dropout(attention_layer.w_o(attn_output))
                
                # Feed forward
                ff_out = block.feed_forward(block.norm2(x))
                x = x + block.dropout(ff_out)
        
        # Store attention patterns
        self.attention_weights_history.append(attention_weights)
        self.step_history.append(step)
        
        # Compute metrics
        self.compute_attention_metrics(attention_weights)
        
        model.train()
        print(f"📸 Captured attention patterns at step {step}")
    
    def compute_attention_metrics(self, attention_weights: List[np.ndarray]):
        """Compute various metrics for attention patterns"""
        layer_entropies = []
        layer_sparsities = []
        
        for layer_idx, layer_attn in enumerate(attention_weights):
            # layer_attn shape: [batch, heads, seq_len, seq_len]
            head_entropies = []
            head_sparsities = []
            
            for head_idx in range(layer_attn.shape[1]):
                head_attn = layer_attn[0, head_idx]  # Take first batch
                
                # Compute entropy for each position
                entropies = []
                sparsities = []
                
                for pos in range(head_attn.shape[0]):
                    attn_dist = head_attn[pos, :pos+1]  # Only look at valid positions (causal)
                    if len(attn_dist) > 1:
                        # Entropy
                        entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-10))
                        entropies.append(entropy)
                        
                        # Sparsity (how concentrated is attention)
                        sparsity = np.sum(attn_dist > 0.1)  # Count positions with >10% attention
                        sparsities.append(sparsity / len(attn_dist))
                
                head_entropies.append(np.mean(entropies) if entropies else 0)
                head_sparsities.append(np.mean(sparsities) if sparsities else 0)
            
            layer_entropies.append(head_entropies)
            layer_sparsities.append(head_sparsities)
        
        self.attention_entropy_history.append(layer_entropies)
        self.attention_sparsity_history.append(layer_sparsities)
    
    def analyze_attention_patterns(self, step_idx: int = -1):
        """Analyze attention patterns for a specific step"""
        if not self.attention_weights_history:
            print("❌ No attention data captured yet")
            return
        
        attention_weights = self.attention_weights_history[step_idx]
        entropies = self.attention_entropy_history[step_idx]
        sparsities = self.attention_sparsity_history[step_idx]
        step = self.step_history[step_idx]
        
        print(f"\n🔍 ATTENTION ANALYSIS - Step {step}")
        print("=" * 60)
        
        # Analyze each layer
        for layer_idx, (layer_attn, layer_entropy, layer_sparsity) in enumerate(zip(attention_weights, entropies, sparsities)):
            print(f"\n📊 Layer {layer_idx}:")
            
            for head_idx in range(len(layer_entropy)):
                entropy = layer_entropy[head_idx]
                sparsity = layer_sparsity[head_idx]
                
                # Classify attention pattern
                if entropy < 1.0:
                    pattern = "🎯 Focused"
                elif entropy > 2.5:
                    pattern = "🌊 Diffuse"
                else:
                    pattern = "⚖️  Balanced"
                
                if sparsity < 0.3:
                    concentration = "Very Concentrated"
                elif sparsity > 0.7:
                    concentration = "Very Spread"
                else:
                    concentration = "Moderate"
                
                print(f"  Head {head_idx}: {pattern} (entropy={entropy:.2f}), {concentration} (sparsity={sparsity:.2f})")
        
        # Find most/least useful heads
        self.identify_useful_heads(attention_weights, entropies, sparsities)
    
    def identify_useful_heads(self, attention_weights: List[np.ndarray], entropies: List[List[float]], sparsities: List[List[float]]):
        """Identify which attention heads seem most/least useful"""
        print(f"\n🎯 ATTENTION HEAD UTILITY ANALYSIS:")
        
        all_heads = []
        for layer_idx, (layer_entropy, layer_sparsity) in enumerate(zip(entropies, sparsities)):
            for head_idx, (entropy, sparsity) in enumerate(zip(layer_entropy, layer_sparsity)):
                all_heads.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'entropy': entropy,
                    'sparsity': sparsity,
                    'utility_score': entropy * (1 - sparsity)  # High entropy + low sparsity = more useful
                })
        
        # Sort by utility
        all_heads.sort(key=lambda x: x['utility_score'], reverse=True)
        
        print("🏆 MOST USEFUL HEADS (high entropy, focused attention):")
        for head in all_heads[:5]:
            print(f"  Layer {head['layer']}, Head {head['head']}: "
                  f"entropy={head['entropy']:.2f}, sparsity={head['sparsity']:.2f}, "
                  f"utility={head['utility_score']:.2f}")
        
        print("\n🗑️  LEAST USEFUL HEADS (low entropy, scattered attention):")
        for head in all_heads[-5:]:
            print(f"  Layer {head['layer']}, Head {head['head']}: "
                  f"entropy={head['entropy']:.2f}, sparsity={head['sparsity']:.2f}, "
                  f"utility={head['utility_score']:.2f}")
    
    def visualize_attention_head(self, layer_idx: int, head_idx: int, step_idx: int = -1, max_seq_len: int = 50):
        """Visualize a specific attention head"""
        if not self.attention_weights_history:
            print("❌ No attention data captured yet")
            return
        
        attention_weights = self.attention_weights_history[step_idx]
        step = self.step_history[step_idx]
        
        if layer_idx >= len(attention_weights) or head_idx >= attention_weights[layer_idx].shape[1]:
            print(f"❌ Invalid layer {layer_idx} or head {head_idx}")
            return
        
        # Get attention matrix for this head
        attn_matrix = attention_weights[layer_idx][0, head_idx]  # First batch
        seq_len = min(attn_matrix.shape[0], max_seq_len)
        attn_matrix = attn_matrix[:seq_len, :seq_len]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_matrix, cmap='Blues', square=True, 
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f'Attention Head L{layer_idx}H{head_idx} - Step {step}', fontsize=14)
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        filename = f"{self.save_dir}/attention_L{layer_idx}H{head_idx}_step_{step:06d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved attention visualization: {filename}")
    
    def create_attention_evolution_gif(self, layer_idx: int, head_idx: int, duration: float = 0.8):
        """Create GIF showing how a specific attention head evolves"""
        if len(self.attention_weights_history) < 2:
            print("❌ Need at least 2 snapshots for animation")
            return
        
        print(f"🎬 Creating attention evolution GIF for L{layer_idx}H{head_idx}...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Find global min/max for consistent color scale
        all_weights = []
        for attn_weights in self.attention_weights_history:
            if layer_idx < len(attn_weights) and head_idx < attn_weights[layer_idx].shape[1]:
                all_weights.append(attn_weights[layer_idx][0, head_idx])
        
        if not all_weights:
            print(f"❌ No data for L{layer_idx}H{head_idx}")
            return
        
        vmin, vmax = 0, np.max([np.max(w) for w in all_weights])
        
        def animate(frame):
            ax.clear()
            
            if frame >= len(self.attention_weights_history):
                return
            
            attn_weights = self.attention_weights_history[frame]
            step = self.step_history[frame]
            
            if layer_idx >= len(attn_weights) or head_idx >= attn_weights[layer_idx].shape[1]:
                return
            
            attn_matrix = attn_weights[layer_idx][0, head_idx]
            seq_len = min(attn_matrix.shape[0], 50)
            attn_matrix = attn_matrix[:seq_len, :seq_len]
            
            im = ax.imshow(attn_matrix, cmap='Blues', vmin=vmin, vmax=vmax, aspect='equal')
            ax.set_title(f'Attention Head L{layer_idx}H{head_idx} Evolution - Step {step}', fontsize=14)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=len(self.attention_weights_history), 
                           interval=duration*1000, blit=False, repeat=True)
        
        gif_filename = f"{self.save_dir}/attention_L{layer_idx}H{head_idx}_evolution.gif"
        anim.save(gif_filename, writer='pillow', fps=1/duration)
        plt.close()
        
        print(f"🎉 Attention evolution GIF saved: {gif_filename}")
    
    def save_data(self):
        """Save all attention analysis data"""
        data = {
            'attention_weights_history': self.attention_weights_history,
            'attention_entropy_history': self.attention_entropy_history,
            'attention_sparsity_history': self.attention_sparsity_history,
            'step_history': self.step_history
        }
        
        filename = f"{self.save_dir}/attention_data.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Saved attention analysis data: {filename}")

# Alternative attention mechanisms for testing
class LinearAttention(nn.Module):
    """Linear attention - O(n) instead of O(n²)"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Linear attention: use feature maps instead of softmax
        Q = F.elu(Q) + 1  # Ensure positive
        K = F.elu(K) + 1
        
        # Compute linear attention
        KV = torch.einsum('bhnd,bhne->bhde', K, V)
        Z = torch.einsum('bhnd->bhd', K)
        
        attn_output = torch.einsum('bhnd,bhde->bhne', Q, KV) / (torch.einsum('bhnd,bhd->bhn', Q, Z).unsqueeze(-1) + 1e-6)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class SparseAttention(nn.Module):
    """Sparse attention - only attend to nearby positions"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, window_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Create sparse attention mask (only attend to nearby positions)
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + 1)  # Causal
            mask[i, start:end] = 1
        
        # Compute attention with sparse mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class NoAttention(nn.Module):
    """No attention - just feed forward (baseline)"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.linear(x))

def main():
    """Demo attention analysis"""
    print("🔍 Attention Analysis Demo")
    print("=" * 40)
    
    # This would be integrated with actual training
    # For now, just show the structure
    analyzer = AttentionAnalyzer()
    
    print("✅ AttentionAnalyzer ready for integration with training!")
    print("📋 Available analysis methods:")
    print("   - capture_attention_patterns()")
    print("   - analyze_attention_patterns()")
    print("   - visualize_attention_head()")
    print("   - create_attention_evolution_gif()")
    print("   - Alternative mechanisms: LinearAttention, SparseAttention, NoAttention")

if __name__ == "__main__":
    main()