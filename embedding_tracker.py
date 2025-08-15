#!/usr/bin/env python3
"""
Embedding Evolution Tracker
Monitors and visualizes how byte embeddings change during training
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os
from typing import List, Dict, Tuple
import pickle
from datetime import datetime

class EmbeddingTracker:
    def __init__(self, vocab_size: int = 256, save_dir: str = "embedding_evolution"):
        self.vocab_size = vocab_size
        self.save_dir = save_dir
        self.embedding_history = []  # List of embedding matrices over time
        self.distance_history = []   # List of distance matrices over time
        self.step_history = []       # Training steps
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # ASCII characters for visualization
        self.ascii_chars = [chr(i) if 32 <= i <= 126 else f"\\x{i:02x}" for i in range(256)]
        
        print(f"📊 EmbeddingTracker initialized, saving to {save_dir}")
    
    def compute_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine distances between embeddings"""
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Convert to distance (1 - similarity)
        distance = 1 - similarity
        
        return distance
    
    def capture_embeddings(self, model, step: int):
        """Capture current embedding state"""
        with torch.no_grad():
            # Get embedding weights
            embeddings = model.token_embedding.weight.clone().cpu()
            
            # Compute distance matrix
            distances = self.compute_pairwise_distances(embeddings)
            
            # Store in history
            self.embedding_history.append(embeddings.numpy())
            self.distance_history.append(distances.numpy())
            self.step_history.append(step)
            
            print(f"📸 Captured embeddings at step {step}")
    
    def print_distance_summary(self, step_idx: int = -1, top_k: int = 10):
        """Print text summary of embedding distances"""
        if not self.distance_history:
            print("❌ No embedding data captured yet")
            return
        
        distances = self.distance_history[step_idx]
        step = self.step_history[step_idx]
        
        print(f"\n🔍 EMBEDDING DISTANCE ANALYSIS - Step {step}")
        print("=" * 60)
        
        # Find most similar pairs (excluding diagonal)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        masked_distances = np.where(mask, distances, np.inf)
        
        # Get indices of smallest distances
        flat_indices = np.argpartition(masked_distances.flatten(), top_k)[:top_k]
        indices = np.unravel_index(flat_indices, distances.shape)
        
        print(f"🔗 TOP {top_k} MOST SIMILAR BYTE PAIRS:")
        for i in range(top_k):
            byte1, byte2 = indices[0][i], indices[1][i]
            dist = distances[byte1, byte2]
            char1 = self.ascii_chars[byte1]
            char2 = self.ascii_chars[byte2]
            print(f"  {byte1:3d}({char1:>4}) ↔ {byte2:3d}({char2:>4}): {dist:.4f}")
        
        # Find most dissimilar pairs
        flat_indices = np.argpartition(-masked_distances.flatten(), top_k)[:top_k]
        indices = np.unravel_index(flat_indices, distances.shape)
        
        print(f"\n🔀 TOP {top_k} MOST DISSIMILAR BYTE PAIRS:")
        for i in range(top_k):
            byte1, byte2 = indices[0][i], indices[1][i]
            dist = distances[byte1, byte2]
            char1 = self.ascii_chars[byte1]
            char2 = self.ascii_chars[byte2]
            print(f"  {byte1:3d}({char1:>4}) ↔ {byte2:3d}({char2:>4}): {dist:.4f}")
        
        # Statistics
        mean_dist = np.mean(distances[mask])
        std_dist = np.std(distances[mask])
        min_dist = np.min(distances[mask])
        max_dist = np.max(distances[mask])
        
        print(f"\n📈 DISTANCE STATISTICS:")
        print(f"  Mean: {mean_dist:.4f} ± {std_dist:.4f}")
        print(f"  Range: [{min_dist:.4f}, {max_dist:.4f}]")
        
        # Character class analysis
        self.analyze_character_classes(distances)
    
    def analyze_character_classes(self, distances: np.ndarray):
        """Analyze distances within character classes"""
        # Define character classes
        classes = {
            'digits': list(range(48, 58)),      # 0-9
            'uppercase': list(range(65, 91)),   # A-Z
            'lowercase': list(range(97, 123)),  # a-z
            'punctuation': [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                           58, 59, 60, 61, 62, 63, 64, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126],
            'whitespace': [9, 10, 13, 32],      # tab, newline, carriage return, space
            'control': list(range(0, 32)) + [127]  # control characters
        }
        
        print(f"\n🏷️  CHARACTER CLASS ANALYSIS:")
        for class_name, byte_list in classes.items():
            if len(byte_list) < 2:
                continue
            
            # Get distances within this class
            class_distances = []
            for i in range(len(byte_list)):
                for j in range(i + 1, len(byte_list)):
                    byte1, byte2 = byte_list[i], byte_list[j]
                    class_distances.append(distances[byte1, byte2])
            
            if class_distances:
                mean_dist = np.mean(class_distances)
                print(f"  {class_name:12}: {mean_dist:.4f} (avg within-class distance)")
    
    def create_heatmap(self, step_idx: int = -1, figsize: Tuple[int, int] = (12, 10)):
        """Create distance heatmap for a specific step"""
        if not self.distance_history:
            print("❌ No embedding data captured yet")
            return None
        
        distances = self.distance_history[step_idx]
        step = self.step_history[step_idx]
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(distances, 
                   cmap='viridis', 
                   square=True,
                   cbar_kws={'label': 'Cosine Distance'},
                   xticklabels=False,
                   yticklabels=False)
        
        plt.title(f'Byte Embedding Distance Matrix - Step {step}', fontsize=16)
        plt.xlabel('Byte Value', fontsize=12)
        plt.ylabel('Byte Value', fontsize=12)
        
        # Add some tick labels for reference
        ticks = [0, 32, 48, 65, 97, 128, 192, 255]  # Key ASCII boundaries
        tick_labels = ['\\x00', 'space', '0', 'A', 'a', '\\x80', '\\xC0', '\\xFF']
        plt.xticks(ticks, tick_labels, rotation=45)
        plt.yticks(ticks, tick_labels, rotation=0)
        
        plt.tight_layout()
        
        # Save
        filename = f"{self.save_dir}/heatmap_step_{step:06d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved heatmap: {filename}")
        return filename
    
    def create_all_heatmaps(self):
        """Create heatmaps for all captured steps"""
        print(f"🎨 Creating heatmaps for {len(self.distance_history)} steps...")
        
        filenames = []
        for i in range(len(self.distance_history)):
            filename = self.create_heatmap(i)
            filenames.append(filename)
        
        return filenames
    
    def create_animated_gif(self, duration: float = 0.5, figsize: Tuple[int, int] = (12, 10)):
        """Create animated GIF showing embedding evolution"""
        if len(self.distance_history) < 2:
            print("❌ Need at least 2 snapshots to create animation")
            return
        
        print(f"🎬 Creating animated GIF with {len(self.distance_history)} frames...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Find global min/max for consistent color scale
        all_distances = np.concatenate([d.flatten() for d in self.distance_history])
        vmin, vmax = np.min(all_distances), np.max(all_distances)
        
        def animate(frame):
            ax.clear()
            
            distances = self.distance_history[frame]
            step = self.step_history[frame]
            
            # Create heatmap
            im = ax.imshow(distances, cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
            
            ax.set_title(f'Byte Embedding Distance Matrix - Step {step}', fontsize=16)
            ax.set_xlabel('Byte Value', fontsize=12)
            ax.set_ylabel('Byte Value', fontsize=12)
            
            # Add tick labels
            ticks = [0, 32, 48, 65, 97, 128, 192, 255]
            tick_labels = ['\\x00', 'space', '0', 'A', 'a', '\\x80', '\\xC0', '\\xFF']
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=45)
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels, rotation=0)
            
            return [im]
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.distance_history), 
                           interval=duration*1000, blit=False, repeat=True)
        
        # Save as GIF
        gif_filename = f"{self.save_dir}/embedding_evolution.gif"
        anim.save(gif_filename, writer='pillow', fps=1/duration)
        plt.close()
        
        print(f"🎉 Animated GIF saved: {gif_filename}")
        return gif_filename
    
    def save_data(self):
        """Save all tracking data"""
        data = {
            'embedding_history': self.embedding_history,
            'distance_history': self.distance_history,
            'step_history': self.step_history,
            'vocab_size': self.vocab_size,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.save_dir}/embedding_data.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Saved tracking data: {filename}")
    
    def load_data(self, filename: str = None):
        """Load tracking data"""
        if filename is None:
            filename = f"{self.save_dir}/embedding_data.pkl"
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.embedding_history = data['embedding_history']
        self.distance_history = data['distance_history']
        self.step_history = data['step_history']
        self.vocab_size = data['vocab_size']
        
        print(f"📦 Loaded tracking data: {filename}")
        print(f"   Steps: {len(self.step_history)}")
    
    def compare_steps(self, step1_idx: int = 0, step2_idx: int = -1):
        """Compare embeddings between two steps"""
        if len(self.distance_history) < 2:
            print("❌ Need at least 2 snapshots to compare")
            return
        
        dist1 = self.distance_history[step1_idx]
        dist2 = self.distance_history[step2_idx]
        step1 = self.step_history[step1_idx]
        step2 = self.step_history[step2_idx]
        
        # Compute difference
        diff = dist2 - dist1
        
        print(f"\n🔄 EMBEDDING EVOLUTION: Step {step1} → Step {step2}")
        print("=" * 60)
        
        # Find biggest changes
        mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
        masked_diff = np.where(mask, np.abs(diff), 0)
        
        # Get indices of biggest changes
        flat_indices = np.argpartition(-masked_diff.flatten(), 10)[:10]
        indices = np.unravel_index(flat_indices, diff.shape)
        
        print("🚀 BIGGEST DISTANCE CHANGES:")
        for i in range(10):
            byte1, byte2 = indices[0][i], indices[1][i]
            change = diff[byte1, byte2]
            char1 = self.ascii_chars[byte1]
            char2 = self.ascii_chars[byte2]
            direction = "↗️" if change > 0 else "↘️"
            print(f"  {byte1:3d}({char1:>4}) ↔ {byte2:3d}({char2:>4}): {change:+.4f} {direction}")
        
        # Overall statistics
        mean_change = np.mean(np.abs(diff[mask]))
        print(f"\n📊 Average absolute change: {mean_change:.4f}")

def main():
    """Demo the embedding tracker"""
    tracker = EmbeddingTracker()
    
    # Create some dummy data for demonstration
    print("🎭 Creating demo data...")
    
    # Simulate embedding evolution
    torch.manual_seed(42)
    d_model = 128
    
    for step in [0, 100, 500, 1000, 2000, 5000]:
        # Create mock model with evolving embeddings
        embeddings = torch.randn(256, d_model)
        
        # Simulate learning: make similar bytes more similar over time
        if step > 0:
            # Make digits more similar to each other
            digit_indices = list(range(48, 58))  # 0-9
            digit_center = embeddings[digit_indices].mean(dim=0, keepdim=True)
            for idx in digit_indices:
                embeddings[idx] = 0.7 * embeddings[idx] + 0.3 * digit_center.squeeze()
            
            # Make letters more similar to each other
            letter_indices = list(range(65, 91)) + list(range(97, 123))  # A-Z, a-z
            letter_center = embeddings[letter_indices].mean(dim=0, keepdim=True)
            for idx in letter_indices:
                embeddings[idx] = 0.8 * embeddings[idx] + 0.2 * letter_center.squeeze()
        
        # Create mock model
        class MockModel:
            def __init__(self, embeddings):
                self.token_embedding = type('obj', (object,), {'weight': embeddings})()
        
        model = MockModel(embeddings)
        tracker.capture_embeddings(model, step)
    
    # Analyze evolution
    print("\n" + "="*80)
    print("📊 INITIAL STATE")
    tracker.print_distance_summary(0)
    
    print("\n" + "="*80)
    print("📊 FINAL STATE")
    tracker.print_distance_summary(-1)
    
    print("\n" + "="*80)
    print("📊 EVOLUTION COMPARISON")
    tracker.compare_steps(0, -1)
    
    # Create visualizations
    tracker.create_all_heatmaps()
    tracker.create_animated_gif()
    tracker.save_data()
    
    print(f"\n✅ Demo complete! Check the '{tracker.save_dir}' directory for outputs.")

if __name__ == "__main__":
    main()