#!/usr/bin/env python3
"""
Analyze embeddings from a trained model
"""

import torch
import sys
import os
from train import ModelConfig, MinimalLLM
from embedding_tracker import EmbeddingTracker

def analyze_model_embeddings(model_path: str):
    """Analyze embeddings from a saved model"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"🔍 Analyzing embeddings from {model_path}")
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = ModelConfig()
            print("⚠️  Using default config")
        
        # Initialize model
        model = MinimalLLM(config)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create tracker and analyze
    tracker = EmbeddingTracker(save_dir=f"analysis_{os.path.basename(model_path).split('.')[0]}")
    tracker.capture_embeddings(model, 0)
    
    # Print detailed analysis
    print(f"\n" + "="*80)
    print("🔬 DETAILED EMBEDDING ANALYSIS")
    print("="*80)
    
    tracker.print_distance_summary(top_k=15)
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    tracker.create_heatmap()
    tracker.save_data()
    
    print(f"\n✅ Analysis complete! Check '{tracker.save_dir}' for outputs.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_embeddings.py <model_path>")
        print("Example: python analyze_embeddings.py byte_llm_model.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    analyze_model_embeddings(model_path)

if __name__ == "__main__":
    main()