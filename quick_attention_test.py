#!/usr/bin/env python3
"""
Quick attention mechanism test to validate our analysis approach
"""

import torch
from attention_experiments import AttentionExperiment, ExperimentConfig

def main():
    """Run quick attention experiments"""
    print("⚡ Quick Attention Mechanism Test")
    print("=" * 50)
    
    # Create lightweight config for quick testing
    config = ExperimentConfig(
        d_model=64,      # Smaller for speed
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=128,
        batch_size=8,
        max_steps=300,   # Very quick
        test_mechanisms=['standard', 'linear', 'sparse_16', 'no_attention']
    )
    
    print(f"🔧 Quick test config: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"🎯 Testing: {config.test_mechanisms}")
    
    # Run experiments
    experiment = AttentionExperiment(config)
    results = experiment.run_all_experiments()
    
    # Analyze results
    experiment.analyze_results()
    experiment.create_comparison_plots()
    experiment.save_results("quick_attention_results.json")
    
    print(f"\n🎉 Quick test complete!")
    print(f"📊 Key insights:")
    
    # Extract key insights
    if results:
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if len(valid_results) >= 2:
            # Find fastest and most accurate
            accuracies = {k: v['final_accuracy'] for k, v in valid_results.items()}
            speeds = {k: v['avg_forward_time_ms'] for k, v in valid_results.items()}
            
            best_acc = max(accuracies, key=accuracies.get)
            fastest = min(speeds, key=speeds.get)
            
            print(f"   🏆 Most accurate: {best_acc} ({accuracies[best_acc]:.3f})")
            print(f"   ⚡ Fastest: {fastest} ({speeds[fastest]:.1f}ms)")
            
            # Speed comparison
            standard_speed = speeds.get('standard', 0)
            if standard_speed > 0:
                for mechanism, speed in speeds.items():
                    if mechanism != 'standard':
                        speedup = standard_speed / speed
                        print(f"   📈 {mechanism}: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")

if __name__ == "__main__":
    main()