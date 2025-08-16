#!/usr/bin/env python3
"""
Focused tracker for lowercase letters (a-z) evolution during training
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

class LetterTracker:
    def __init__(self, save_dir: str = "letter_evolution"):
        self.save_dir = save_dir
        self.letter_indices = list(range(97, 123))  # a-z (97-122)
        self.letters = [chr(i) for i in self.letter_indices]
        
        # Storage for tracking
        self.embedding_history = []  # Full embeddings over time
        self.letter_embeddings_history = []  # Just letter embeddings
        self.letter_distances_history = []   # Letter-to-letter distances
        self.step_history = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"📝 LetterTracker initialized for letters: {self.letters}")
        print(f"💾 Saving to: {save_dir}")
    
    def compute_letter_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine distances between letter embeddings"""
        # Extract letter embeddings
        letter_embeddings = embeddings[self.letter_indices]
        
        # Normalize
        letter_embeddings_norm = F.normalize(letter_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(letter_embeddings_norm, letter_embeddings_norm.t())
        
        # Convert to distance
        distance = 1 - similarity
        
        return distance, letter_embeddings
    
    def capture_letters(self, model, step: int):
        """Capture current letter embedding state"""
        with torch.no_grad():
            # Get full embedding weights
            embeddings = model.token_embedding.weight.clone().cpu()
            
            # Compute letter-specific distances and embeddings
            letter_distances, letter_embeddings = self.compute_letter_distances(embeddings)
            
            # Store in history
            self.embedding_history.append(embeddings.numpy())
            self.letter_embeddings_history.append(letter_embeddings.numpy())
            self.letter_distances_history.append(letter_distances.numpy())
            self.step_history.append(step)
            
            print(f"📸 Captured letter embeddings at step {step}")
    
    def print_letter_analysis(self, step_idx: int = -1, top_k: int = 5):
        """Print detailed analysis of letter relationships"""
        if not self.letter_distances_history:
            print("❌ No letter data captured yet")
            return
        
        distances = self.letter_distances_history[step_idx]
        step = self.step_history[step_idx]
        
        print(f"\n🔤 LETTER ANALYSIS - Step {step}")
        print("=" * 50)
        
        # Find most similar letter pairs
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        masked_distances = np.where(mask, distances, np.inf)
        
        # Get indices of smallest distances
        flat_indices = np.argpartition(masked_distances.flatten(), top_k)[:top_k]
        indices = np.unravel_index(flat_indices, distances.shape)
        
        print(f"🤝 TOP {top_k} MOST SIMILAR LETTER PAIRS:")
        for i in range(top_k):
            idx1, idx2 = indices[0][i], indices[1][i]
            letter1, letter2 = self.letters[idx1], self.letters[idx2]
            dist = distances[idx1, idx2]
            print(f"  {letter1} ↔ {letter2}: {dist:.4f}")
        
        # Find most dissimilar pairs
        flat_indices = np.argpartition(-masked_distances.flatten(), top_k)[:top_k]
        indices = np.unravel_index(flat_indices, distances.shape)
        
        print(f"\n🔀 TOP {top_k} MOST DISSIMILAR LETTER PAIRS:")
        for i in range(top_k):
            idx1, idx2 = indices[0][i], indices[1][i]
            letter1, letter2 = self.letters[idx1], self.letters[idx2]
            dist = distances[idx1, idx2]
            print(f"  {letter1} ↔ {letter2}: {dist:.4f}")
        
        # Statistics
        mean_dist = np.mean(distances[mask])
        std_dist = np.std(distances[mask])
        min_dist = np.min(distances[mask])
        max_dist = np.max(distances[mask])
        
        print(f"\n📊 LETTER DISTANCE STATISTICS:")
        print(f"  Mean: {mean_dist:.4f} ± {std_dist:.4f}")
        print(f"  Range: [{min_dist:.4f}, {max_dist:.4f}]")
        
        # Vowel vs consonant analysis
        self.analyze_vowel_consonant(distances)
    
    def analyze_vowel_consonant(self, distances: np.ndarray):
        """Analyze vowel vs consonant clustering"""
        vowels = ['a', 'e', 'i', 'o', 'u']
        vowel_indices = [self.letters.index(v) for v in vowels]
        consonant_indices = [i for i in range(26) if i not in vowel_indices]
        
        # Vowel-vowel distances
        vowel_distances = []
        for i in range(len(vowel_indices)):
            for j in range(i + 1, len(vowel_indices)):
                idx1, idx2 = vowel_indices[i], vowel_indices[j]
                vowel_distances.append(distances[idx1, idx2])
        
        # Consonant-consonant distances
        consonant_distances = []
        for i in range(len(consonant_indices)):
            for j in range(i + 1, len(consonant_indices)):
                idx1, idx2 = consonant_indices[i], consonant_indices[j]
                consonant_distances.append(distances[idx1, idx2])
        
        # Vowel-consonant distances
        vowel_consonant_distances = []
        for v_idx in vowel_indices:
            for c_idx in consonant_indices:
                vowel_consonant_distances.append(distances[v_idx, c_idx])
        
        print(f"\n🎵 VOWEL vs CONSONANT ANALYSIS:")
        print(f"  Vowel-Vowel avg distance:     {np.mean(vowel_distances):.4f}")
        print(f"  Consonant-Consonant avg:      {np.mean(consonant_distances):.4f}")
        print(f"  Vowel-Consonant avg:          {np.mean(vowel_consonant_distances):.4f}")
        
        # Check if vowels cluster together
        vowel_internal = np.mean(vowel_distances)
        vowel_external = np.mean(vowel_consonant_distances)
        vowel_clustering = vowel_external - vowel_internal
        
        print(f"  Vowel clustering score:       {vowel_clustering:.4f} {'✅' if vowel_clustering > 0 else '❌'}")
    
    def create_letter_heatmap(self, step_idx: int = -1, figsize: Tuple[int, int] = (10, 8)):
        """Create heatmap for letter distances"""
        if not self.letter_distances_history:
            print("❌ No letter data captured yet")
            return None
        
        distances = self.letter_distances_history[step_idx]
        step = self.step_history[step_idx]
        
        plt.figure(figsize=figsize)
        
        # Create heatmap with letter labels
        sns.heatmap(distances, 
                   annot=False,
                   cmap='viridis_r',  # Reversed so dark = similar
                   square=True,
                   cbar_kws={'label': 'Cosine Distance'},
                   xticklabels=self.letters,
                   yticklabels=self.letters)
        
        plt.title(f'Lowercase Letter Distance Matrix - Step {step}', fontsize=16, pad=20)
        plt.xlabel('Letters', fontsize=12)
        plt.ylabel('Letters', fontsize=12)
        
        # Highlight vowels
        vowel_positions = [self.letters.index(v) for v in ['a', 'e', 'i', 'o', 'u']]
        for pos in vowel_positions:
            plt.axhline(y=pos+0.5, color='red', linewidth=2, alpha=0.7)
            plt.axvline(x=pos+0.5, color='red', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        
        # Save
        filename = f"{self.save_dir}/letter_heatmap_step_{step:06d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved letter heatmap: {filename}")
        return filename
    
    def create_letter_evolution_gif(self, duration: float = 0.8, figsize: Tuple[int, int] = (10, 8)):
        """Create animated GIF of letter evolution"""
        if len(self.letter_distances_history) < 2:
            print("❌ Need at least 2 snapshots for animation")
            return
        
        print(f"🎬 Creating letter evolution GIF with {len(self.letter_distances_history)} frames...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Find global min/max for consistent color scale
        all_distances = np.concatenate([d.flatten() for d in self.letter_distances_history])
        vmin, vmax = np.min(all_distances), np.max(all_distances)
        
        def animate(frame):
            ax.clear()
            
            distances = self.letter_distances_history[frame]
            step = self.step_history[frame]
            
            # Create heatmap
            im = ax.imshow(distances, cmap='viridis_r', vmin=vmin, vmax=vmax, aspect='equal')
            
            ax.set_title(f'Letter Distance Evolution - Step {step}', fontsize=16, pad=20)
            ax.set_xlabel('Letters', fontsize=12)
            ax.set_ylabel('Letters', fontsize=12)
            
            # Set letter labels
            ax.set_xticks(range(26))
            ax.set_xticklabels(self.letters)
            ax.set_yticks(range(26))
            ax.set_yticklabels(self.letters)
            
            # Highlight vowels
            vowel_positions = [self.letters.index(v) for v in ['a', 'e', 'i', 'o', 'u']]
            for pos in vowel_positions:
                ax.axhline(y=pos-0.5, color='red', linewidth=2, alpha=0.7)
                ax.axvline(x=pos-0.5, color='red', linewidth=2, alpha=0.7)
            
            return [im]
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.letter_distances_history), 
                           interval=duration*1000, blit=False, repeat=True)
        
        # Save as GIF
        gif_filename = f"{self.save_dir}/letter_evolution.gif"
        anim.save(gif_filename, writer='pillow', fps=1/duration)
        plt.close()
        
        print(f"🎉 Letter evolution GIF saved: {gif_filename}")
        return gif_filename
    
    def create_vowel_consonant_gif(self, duration: float = 0.8, figsize: Tuple[int, int] = (12, 5)):
        """Create GIF showing vowel vs consonant clustering over time"""
        if len(self.letter_distances_history) < 2:
            print("❌ Need at least 2 snapshots for animation")
            return
        
        print(f"🎵 Creating vowel-consonant analysis GIF...")
        
        # Prepare data
        vowels = ['a', 'e', 'i', 'o', 'u']
        vowel_indices = [self.letters.index(v) for v in vowels]
        consonant_indices = [i for i in range(26) if i not in vowel_indices]
        
        vowel_vowel_history = []
        consonant_consonant_history = []
        vowel_consonant_history = []
        
        for distances in self.letter_distances_history:
            # Vowel-vowel distances
            vv_distances = []
            for i in range(len(vowel_indices)):
                for j in range(i + 1, len(vowel_indices)):
                    idx1, idx2 = vowel_indices[i], vowel_indices[j]
                    vv_distances.append(distances[idx1, idx2])
            
            # Consonant-consonant distances
            cc_distances = []
            for i in range(len(consonant_indices)):
                for j in range(i + 1, len(consonant_indices)):
                    idx1, idx2 = consonant_indices[i], consonant_indices[j]
                    cc_distances.append(distances[idx1, idx2])
            
            # Vowel-consonant distances
            vc_distances = []
            for v_idx in vowel_indices:
                for c_idx in consonant_indices:
                    vc_distances.append(distances[v_idx, c_idx])
            
            vowel_vowel_history.append(np.mean(vv_distances))
            consonant_consonant_history.append(np.mean(cc_distances))
            vowel_consonant_history.append(np.mean(vc_distances))
        
        # Create animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            step = self.step_history[frame]
            
            # Plot 1: Distance trends over time
            steps_so_far = self.step_history[:frame+1]
            vv_so_far = vowel_vowel_history[:frame+1]
            cc_so_far = consonant_consonant_history[:frame+1]
            vc_so_far = vowel_consonant_history[:frame+1]
            
            ax1.plot(steps_so_far, vv_so_far, 'ro-', label='Vowel-Vowel', linewidth=2)
            ax1.plot(steps_so_far, cc_so_far, 'bo-', label='Consonant-Consonant', linewidth=2)
            ax1.plot(steps_so_far, vc_so_far, 'go-', label='Vowel-Consonant', linewidth=2)
            
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Average Distance')
            ax1.set_title(f'Letter Clustering Evolution - Step {step}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Current clustering scores
            current_vv = vowel_vowel_history[frame]
            current_cc = consonant_consonant_history[frame]
            current_vc = vowel_consonant_history[frame]
            
            categories = ['Vowel-Vowel', 'Consonant-Consonant', 'Vowel-Consonant']
            values = [current_vv, current_cc, current_vc]
            colors = ['red', 'blue', 'green']
            
            bars = ax2.bar(categories, values, color=colors, alpha=0.7)
            ax2.set_ylabel('Average Distance')
            ax2.set_title(f'Current Clustering - Step {step}')
            ax2.set_ylim(0, max(max(vowel_vowel_history), max(consonant_consonant_history), max(vowel_consonant_history)) * 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.letter_distances_history), 
                           interval=duration*1000, blit=False, repeat=True)
        
        # Save as GIF
        gif_filename = f"{self.save_dir}/vowel_consonant_evolution.gif"
        anim.save(gif_filename, writer='pillow', fps=1/duration)
        plt.close()
        
        print(f"🎉 Vowel-consonant GIF saved: {gif_filename}")
        return gif_filename
    
    def save_data(self):
        """Save all letter tracking data"""
        data = {
            'letter_embeddings_history': self.letter_embeddings_history,
            'letter_distances_history': self.letter_distances_history,
            'step_history': self.step_history,
            'letters': self.letters,
            'letter_indices': self.letter_indices,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.save_dir}/letter_data.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Saved letter tracking data: {filename}")
    
    def compare_letter_evolution(self, step1_idx: int = 0, step2_idx: int = -1):
        """Compare letter relationships between two steps"""
        if len(self.letter_distances_history) < 2:
            print("❌ Need at least 2 snapshots to compare")
            return
        
        dist1 = self.letter_distances_history[step1_idx]
        dist2 = self.letter_distances_history[step2_idx]
        step1 = self.step_history[step1_idx]
        step2 = self.step_history[step2_idx]
        
        diff = dist2 - dist1
        
        print(f"\n🔄 LETTER EVOLUTION: Step {step1} → Step {step2}")
        print("=" * 50)
        
        # Find biggest changes
        mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
        masked_diff = np.where(mask, np.abs(diff), 0)
        
        # Get indices of biggest changes
        flat_indices = np.argpartition(-masked_diff.flatten(), 10)[:10]
        indices = np.unravel_index(flat_indices, diff.shape)
        
        print("🚀 BIGGEST LETTER RELATIONSHIP CHANGES:")
        for i in range(10):
            idx1, idx2 = indices[0][i], indices[1][i]
            letter1, letter2 = self.letters[idx1], self.letters[idx2]
            change = diff[idx1, idx2]
            direction = "↗️ (more distant)" if change > 0 else "↘️ (more similar)"
            print(f"  {letter1} ↔ {letter2}: {change:+.4f} {direction}")

def main():
    """Demo the letter tracker"""
    from train import ModelConfig, MinimalLLM, set_seed
    
    print("🔤 Letter Evolution Demo")
    print("=" * 40)
    
    # Create tracker
    tracker = LetterTracker()
    
    # Create demo model evolution
    set_seed(42)
    config = ModelConfig(d_model=128, vocab_size=256)
    
    print("🎭 Simulating letter evolution over 5000 steps...")
    
    for step in [0, 500, 1000, 2000, 3000, 4000, 5000]:
        # Create mock model
        embeddings = torch.randn(256, config.d_model)
        
        # Simulate learning: make letters more structured over time
        if step > 0:
            # Make vowels cluster together
            vowel_indices = [97, 101, 105, 111, 117]  # a, e, i, o, u
            vowel_center = embeddings[vowel_indices].mean(dim=0, keepdim=True)
            strength = min(step / 5000.0, 0.8)  # Gradually increase clustering
            
            for idx in vowel_indices:
                embeddings[idx] = (1 - strength) * embeddings[idx] + strength * vowel_center.squeeze()
            
            # Make consonants cluster (but less than vowels)
            consonant_indices = [i for i in range(97, 123) if i not in vowel_indices]
            consonant_center = embeddings[consonant_indices].mean(dim=0, keepdim=True)
            consonant_strength = strength * 0.3  # Weaker clustering
            
            for idx in consonant_indices:
                embeddings[idx] = (1 - consonant_strength) * embeddings[idx] + consonant_strength * consonant_center.squeeze()
        
        # Mock model
        class MockModel:
            def __init__(self, embeddings):
                self.token_embedding = type('obj', (object,), {'weight': embeddings})()
        
        model = MockModel(embeddings)
        tracker.capture_letters(model, step)
    
    # Analysis
    print("\n" + "="*60)
    print("📊 INITIAL LETTER STATE")
    tracker.print_letter_analysis(0)
    
    print("\n" + "="*60)
    print("📊 FINAL LETTER STATE")
    tracker.print_letter_analysis(-1)
    
    print("\n" + "="*60)
    print("📊 LETTER EVOLUTION COMPARISON")
    tracker.compare_letter_evolution(0, -1)
    
    # Create all visualizations
    print(f"\n🎨 Creating letter visualizations...")
    
    # Individual heatmaps
    for i in range(len(tracker.step_history)):
        tracker.create_letter_heatmap(i)
    
    # Animated GIFs
    tracker.create_letter_evolution_gif()
    tracker.create_vowel_consonant_gif()
    tracker.save_data()
    
    print(f"\n✅ Letter analysis complete! Check '{tracker.save_dir}' for:")
    print(f"   - letter_evolution.gif (main heatmap animation)")
    print(f"   - vowel_consonant_evolution.gif (clustering analysis)")
    print(f"   - Individual heatmaps for each step")

if __name__ == "__main__":
    main()