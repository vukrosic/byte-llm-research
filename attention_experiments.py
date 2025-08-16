#!/usr/bin/env python3
"""
Attention Mechanism Experiments
Tests different attention mechanisms to find more efficient alternatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json

from train import ModelConfig, MinimalLLM, TextByteDataset, set_seed
from attention_analyzer import AttentionAnalyzer, LinearAttention, SparseAttention, NoAttention

@dataclass
class ExperimentConfig:
    """Configuration for attention experiments"""
    # Model settings
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    vocab_size: int = 256
    
    # Training settings
    batch_size: int = 16
    max_steps: int = 1000
    learning_rate: float = 0.001
    
    # Experiment settings
    test_mechanisms: List[str] = None
    
    def __post_init__(self):
        if self.test_mechanisms is None:
            self.test_mechanisms = ['standard', 'linear', 'sparse_16', 'sparse_32', 'no_attention']

class AttentionExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        
    def create_model_with_attention(self, attention_type: str) -> MinimalLLM:
        """Create model with specified attention mechanism"""
        model_config = ModelConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size
        )
        
        model = MinimalLLM(model_config)
        
        # Replace attention mechanisms
        if attention_type != 'standard':
            for block in model.transformer_blocks:
                if attention_type == 'linear':
                    block.attention = LinearAttention(
                        self.config.d_model, self.config.n_heads, 
                        self.config.max_seq_len, 0.1
                    )
                elif attention_type.startswith('sparse_'):
                    window_size = int(attention_type.split('_')[1])
                    block.attention = SparseAttention(
                        self.config.d_model, self.config.n_heads, 
                        self.config.max_seq_len, window_size, 0.1
                    )
                elif attention_type == 'no_attention':
                    block.attention = NoAttention(
                        self.config.d_model, self.config.n_heads, 
                        self.config.max_seq_len, 0.1
                    )
        
        return model
    
    def create_simple_dataset(self, size: int = 5000) -> TextByteDataset:
        """Create simple dataset for experiments"""
        # Create patterns that require different types of attention
        patterns = []
        
        # Local patterns (good for sparse attention)
        for _ in range(100):
            patterns.append(b"abcdefghijklmnopqrstuvwxyz" * 3)
        
        # Long-range dependencies (good for full attention)
        for _ in range(50):
            patterns.append(b"start" + b"x" * 50 + b"middle" + b"y" * 50 + b"end")
        
        # Repetitive patterns (might not need attention)
        for _ in range(100):
            patterns.append(b"hello world " * 20)
        
        # Random text
        for _ in range(50):
            random_bytes = np.random.randint(97, 123, size=200)  # Random letters
            patterns.append(bytes(random_bytes))
        
        # Combine patterns
        all_bytes = []
        for pattern in patterns:
            all_bytes.extend(list(pattern))
        
        # Extend to desired size
        while len(all_bytes) < size:
            all_bytes.extend(all_bytes[:min(size - len(all_bytes), len(all_bytes))])
        
        return TextByteDataset(all_bytes[:size], self.config.max_seq_len)
    
    def measure_performance(self, model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict:
        """Measure model performance and efficiency"""
        model.eval()
        
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        forward_times = []
        memory_usage = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                if i >= 50:  # Limit for speed
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Measure forward pass time
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                
                # Measure memory before forward pass
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated()
                
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                
                # Measure memory after forward pass
                if device.type == 'cuda':
                    memory_after = torch.cuda.memory_allocated()
                    memory_usage.append(memory_after - memory_before)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                forward_times.append(time.time() - start_time)
                
                # Compute metrics
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
                
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == y).sum().item()
        
        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        avg_forward_time = np.mean(forward_times)
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'forward_time_ms': avg_forward_time * 1000,
            'memory_mb': avg_memory / (1024 * 1024),
            'perplexity': min(np.exp(avg_loss), 1000)
        }
    
    def train_and_evaluate(self, attention_type: str) -> Dict:
        """Train and evaluate a model with specific attention mechanism"""
        print(f"\n🧪 Testing {attention_type} attention...")
        
        # Set seed for fair comparison
        set_seed(42)
        
        # Create model and data
        model = self.create_model_with_attention(attention_type)
        dataset = self.create_simple_dataset()
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        
        print(f"  📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        model.train()
        training_losses = []
        training_times = []
        
        step = 0
        start_time = time.time()
        
        for epoch in range(10):
            for x, y in train_loader:
                if step >= self.config.max_steps:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Time the training step
                step_start = time.time()
                
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
                
                step_time = time.time() - step_start
                training_times.append(step_time)
                training_losses.append(loss.item())
                
                if step % 200 == 0:
                    print(f"    Step {step}: Loss {loss.item():.4f}, Time {step_time*1000:.1f}ms")
                
                step += 1
            
            if step >= self.config.max_steps:
                break
        
        total_training_time = time.time() - start_time
        
        # Evaluate
        performance = self.measure_performance(model, val_loader, device)
        
        # Compile results
        results = {
            'attention_type': attention_type,
            'final_loss': performance['loss'],
            'final_accuracy': performance['accuracy'],
            'final_perplexity': performance['perplexity'],
            'avg_forward_time_ms': performance['forward_time_ms'],
            'avg_memory_mb': performance['memory_mb'],
            'total_training_time_s': total_training_time,
            'avg_training_step_ms': np.mean(training_times) * 1000,
            'training_loss_curve': training_losses[-100:],  # Last 100 steps
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"  ✅ {attention_type}: Loss {results['final_loss']:.4f}, "
              f"Acc {results['final_accuracy']:.3f}, "
              f"Time {results['avg_forward_time_ms']:.1f}ms")
        
        return results
    
    def run_all_experiments(self) -> Dict:
        """Run experiments for all attention mechanisms"""
        print("🚀 Running Attention Mechanism Experiments")
        print("=" * 60)
        
        all_results = {}
        
        for attention_type in self.config.test_mechanisms:
            try:
                results = self.train_and_evaluate(attention_type)
                all_results[attention_type] = results
            except Exception as e:
                print(f"❌ Failed to test {attention_type}: {e}")
                all_results[attention_type] = {'error': str(e)}
        
        self.results = all_results
        return all_results
    
    def analyze_results(self):
        """Analyze and compare results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print(f"\n📊 ATTENTION MECHANISM COMPARISON")
        print("=" * 80)
        
        # Create comparison table
        mechanisms = []
        losses = []
        accuracies = []
        forward_times = []
        memory_usage = []
        training_times = []
        
        for mechanism, results in self.results.items():
            if 'error' in results:
                continue
            
            mechanisms.append(mechanism)
            losses.append(results['final_loss'])
            accuracies.append(results['final_accuracy'])
            forward_times.append(results['avg_forward_time_ms'])
            memory_usage.append(results['avg_memory_mb'])
            training_times.append(results['avg_training_step_ms'])
        
        # Print comparison table
        print(f"{'Mechanism':<15} {'Loss':<8} {'Accuracy':<8} {'Forward(ms)':<12} {'Memory(MB)':<12} {'Train(ms)':<10}")
        print("-" * 80)
        
        for i, mechanism in enumerate(mechanisms):
            print(f"{mechanism:<15} {losses[i]:<8.4f} {accuracies[i]:<8.3f} "
                  f"{forward_times[i]:<12.1f} {memory_usage[i]:<12.1f} {training_times[i]:<10.1f}")
        
        # Find best mechanisms
        if losses:
            best_accuracy_idx = np.argmax(accuracies)
            best_speed_idx = np.argmin(forward_times)
            best_memory_idx = np.argmin(memory_usage)
            
            print(f"\n🏆 WINNERS:")
            print(f"  Best Accuracy: {mechanisms[best_accuracy_idx]} ({accuracies[best_accuracy_idx]:.3f})")
            print(f"  Fastest: {mechanisms[best_speed_idx]} ({forward_times[best_speed_idx]:.1f}ms)")
            print(f"  Most Memory Efficient: {mechanisms[best_memory_idx]} ({memory_usage[best_memory_idx]:.1f}MB)")
            
            # Efficiency score (accuracy / time)
            efficiency_scores = [acc / time for acc, time in zip(accuracies, forward_times)]
            best_efficiency_idx = np.argmax(efficiency_scores)
            print(f"  Best Efficiency: {mechanisms[best_efficiency_idx]} (score: {efficiency_scores[best_efficiency_idx]:.4f})")
    
    def create_comparison_plots(self):
        """Create visualization comparing mechanisms"""
        if not self.results:
            return
        
        # Filter out error results
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if len(valid_results) < 2:
            print("❌ Need at least 2 valid results for plotting")
            return
        
        mechanisms = list(valid_results.keys())
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Accuracy vs Speed
        accuracies = [valid_results[m]['final_accuracy'] for m in mechanisms]
        forward_times = [valid_results[m]['avg_forward_time_ms'] for m in mechanisms]
        
        ax1.scatter(forward_times, accuracies, s=100, alpha=0.7)
        for i, mechanism in enumerate(mechanisms):
            ax1.annotate(mechanism, (forward_times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Forward Time (ms)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Speed Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss comparison
        losses = [valid_results[m]['final_loss'] for m in mechanisms]
        ax2.bar(mechanisms, losses, alpha=0.7)
        ax2.set_ylabel('Final Loss')
        ax2.set_title('Final Loss Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Memory usage
        memory_usage = [valid_results[m]['avg_memory_mb'] for m in mechanisms]
        ax3.bar(mechanisms, memory_usage, alpha=0.7, color='orange')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Training time
        training_times = [valid_results[m]['avg_training_step_ms'] for m in mechanisms]
        ax4.bar(mechanisms, training_times, alpha=0.7, color='green')
        ax4.set_ylabel('Training Step Time (ms)')
        ax4.set_title('Training Speed Comparison')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('attention_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("📊 Saved comparison plots: attention_comparison.png")
    
    def save_results(self, filename: str = "attention_experiment_results.json"):
        """Save results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Saved results: {filename}")

def main():
    """Run attention experiments"""
    config = ExperimentConfig(
        max_steps=500,  # Quick experiments
        test_mechanisms=['standard', 'linear', 'sparse_16', 'sparse_32', 'no_attention']
    )
    
    experiment = AttentionExperiment(config)
    results = experiment.run_all_experiments()
    
    experiment.analyze_results()
    experiment.create_comparison_plots()
    experiment.save_results()
    
    print(f"\n✅ Attention experiments complete!")
    print(f"📁 Check attention_comparison.png and attention_experiment_results.json")

if __name__ == "__main__":
    main()