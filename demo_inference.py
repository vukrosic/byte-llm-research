#!/usr/bin/env python3
"""
Demo script showing how to use the byte-level LLM for inference
"""

import torch
from inference import ByteLevelGenerator, InferenceConfig

def demo_generation():
    """Demo text generation with various prompts"""
    
    # Check if model exists
    model_path = "byte_llm_model.pt"
    try:
        # Create inference config
        config = InferenceConfig(
            max_new_bytes=512,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        # Initialize generator
        generator = ByteLevelGenerator(model_path, config)
        
        # Demo prompts
        prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The key to understanding quantum physics",
            "In the year 2050, technology will",
            "The most important lesson I learned"
        ]
        
        print("\n" + "="*60)
        print("🎭 BYTE-LEVEL LLM DEMO GENERATIONS")
        print("="*60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 DEMO {i}/5")
            print(f"🎯 PROMPT: {prompt}")
            print("-" * 40)
            
            try:
                generated = generator.generate(prompt, max_new_bytes=256)
                print(f"🤖 GENERATED: {generated}")
            except Exception as e:
                print(f"❌ Generation failed: {e}")
            
            print("-" * 40)
        
        print(f"\n✅ Demo completed! Run 'python inference.py {model_path}' for interactive mode.")
        
    except FileNotFoundError:
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train the model first by running: python train.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_generation()