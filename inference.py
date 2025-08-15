#!/usr/bin/env python3
"""
Interactive Byte-Level LLM Inference Script
Generates text using the full context window for maximum coherence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Import model components from train.py
from train import ModelConfig, MinimalLLM, Rotary, MultiHeadAttention, FeedForward, TransformerBlock

@dataclass
class InferenceConfig:
    # Generation parameters
    max_new_bytes: int = 256   # Maximum bytes to generate (reduced to fit context)
    temperature: float = 0.8   # Sampling temperature
    top_k: int = 50           # Top-k sampling
    top_p: float = 0.9        # Nucleus sampling
    repetition_penalty: float = 1.1  # Penalize repetition
    
    # Model parameters (should match training config)
    max_seq_len: int = 512
    vocab_size: int = 256

class ByteLevelGenerator:
    def __init__(self, model_path: str, config: InferenceConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"🔄 Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model config from checkpoint if available
        if 'config' in checkpoint:
            self.model_config = checkpoint['config']
        else:
            # Fallback to default config
            self.model_config = ModelConfig()
            print("⚠️  Using default model config (checkpoint config not found)")
        
        # Initialize model
        self.model = MinimalLLM(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def text_to_bytes(self, text: str) -> list:
        """Convert text to list of byte values"""
        return list(text.encode('utf-8'))
    
    def bytes_to_text(self, byte_list: list) -> str:
        """Convert list of byte values to text, handling decode errors gracefully"""
        try:
            return bytes(byte_list).decode('utf-8')
        except UnicodeDecodeError:
            # Handle partial UTF-8 sequences by trying to decode what we can
            result = ""
            i = 0
            while i < len(byte_list):
                try:
                    # Try to decode from current position
                    result += bytes(byte_list[i:]).decode('utf-8')
                    break
                except UnicodeDecodeError as e:
                    # Add the valid part and skip the problematic byte
                    if e.start > 0:
                        result += bytes(byte_list[i:i+e.start]).decode('utf-8')
                        i += e.start
                    else:
                        i += 1
            return result
    
    def apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.1):
        """Apply repetition penalty to logits"""
        if penalty == 1.0:
            return logits
        
        # Get unique tokens in input and apply penalty
        for token_id in set(input_ids.tolist()):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        
        return logits
    
    def sample_next_byte(self, logits: torch.Tensor, input_ids: torch.Tensor) -> int:
        """Sample next byte using temperature, top-k, and top-p"""
        # Apply repetition penalty
        logits = self.apply_repetition_penalty(logits, input_ids, self.config.repetition_penalty)
        
        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        else:
            # Greedy sampling
            return logits.argmax().item()
        
        # Apply top-k filtering
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_byte = torch.multinomial(probs, num_samples=1).item()
        
        return next_byte
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_bytes: Optional[int] = None) -> str:
        """Generate text continuation for the given prompt"""
        if max_new_bytes is None:
            max_new_bytes = self.config.max_new_bytes
        
        # Ensure we don't try to generate more than the context window allows
        max_new_bytes = min(max_new_bytes, self.config.max_seq_len - 1)
        
        # Convert prompt to bytes
        input_bytes = self.text_to_bytes(prompt)
        
        # Truncate to fit in context window, leaving room for generation
        max_prompt_len = max(1, self.config.max_seq_len - max_new_bytes)
        if len(input_bytes) > max_prompt_len:
            input_bytes = input_bytes[-max_prompt_len:]
            print(f"⚠️  Prompt truncated to {len(input_bytes)} bytes to fit context window")
        elif len(input_bytes) == 0:
            # Handle empty prompt case
            input_bytes = [32]  # Start with a space character
            print("⚠️  Empty prompt detected, starting with space character")
        
        # Convert to tensor
        input_ids = torch.tensor(input_bytes, dtype=torch.long, device=self.device).unsqueeze(0)
        
        print(f"🎯 Generating {max_new_bytes} bytes from {len(input_bytes)} byte prompt...")
        print(f"🔄 Using temperature={self.config.temperature}, top_k={self.config.top_k}, top_p={self.config.top_p}")
        
        generated_bytes = []
        
        for i in range(max_new_bytes):
            # Get current sequence length
            current_len = input_ids.size(1)
            
            # Truncate if we exceed max sequence length
            if current_len >= self.config.max_seq_len:
                # Keep the most recent tokens to maintain context
                input_ids = input_ids[:, -self.config.max_seq_len + 1:]
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = self.model(input_ids)
            
            # Get logits for the last position
            next_byte_logits = logits[0, -1, :]
            
            # Sample next byte
            next_byte = self.sample_next_byte(next_byte_logits, input_ids[0])
            
            # Add to generated sequence
            generated_bytes.append(next_byte)
            
            # Update input_ids for next iteration
            next_byte_tensor = torch.tensor([[next_byte]], dtype=torch.long, device=self.device)
            input_ids = torch.cat([input_ids, next_byte_tensor], dim=1)
            
            # Show progress every 100 bytes
            if (i + 1) % 100 == 0:
                partial_text = self.bytes_to_text(generated_bytes)
                print(f"Generated {i + 1}/{max_new_bytes} bytes...")
        
        # Convert generated bytes to text
        generated_text = self.bytes_to_text(generated_bytes)
        return generated_text

def interactive_mode(generator: ByteLevelGenerator):
    """Run interactive text generation"""
    print("\n" + "="*60)
    print("🤖 BYTE-LEVEL LLM INTERACTIVE MODE")
    print("="*60)
    print("Commands:")
    print("  /help     - Show this help")
    print("  /config   - Show current generation config")
    print("  /temp X   - Set temperature to X")
    print("  /topk X   - Set top-k to X")
    print("  /topp X   - Set top-p to X")
    print("  /length X - Set max generation length to X")
    print("  /quit     - Exit")
    print("  Just type text to generate continuation!")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n💬 Enter prompt (or command): ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()
                
                if cmd == '/help':
                    print("Commands: /help, /config, /temp, /topk, /topp, /length, /quit")
                
                elif cmd == '/config':
                    print(f"Current config:")
                    print(f"  Temperature: {generator.config.temperature}")
                    print(f"  Top-k: {generator.config.top_k}")
                    print(f"  Top-p: {generator.config.top_p}")
                    print(f"  Max length: {generator.config.max_new_bytes}")
                    print(f"  Repetition penalty: {generator.config.repetition_penalty}")
                
                elif cmd == '/temp' and len(cmd_parts) > 1:
                    try:
                        generator.config.temperature = float(cmd_parts[1])
                        print(f"✅ Temperature set to {generator.config.temperature}")
                    except ValueError:
                        print("❌ Invalid temperature value")
                
                elif cmd == '/topk' and len(cmd_parts) > 1:
                    try:
                        generator.config.top_k = int(cmd_parts[1])
                        print(f"✅ Top-k set to {generator.config.top_k}")
                    except ValueError:
                        print("❌ Invalid top-k value")
                
                elif cmd == '/topp' and len(cmd_parts) > 1:
                    try:
                        generator.config.top_p = float(cmd_parts[1])
                        print(f"✅ Top-p set to {generator.config.top_p}")
                    except ValueError:
                        print("❌ Invalid top-p value")
                
                elif cmd == '/length' and len(cmd_parts) > 1:
                    try:
                        new_length = int(cmd_parts[1])
                        max_allowed = generator.config.max_seq_len - 1
                        if new_length > max_allowed:
                            print(f"⚠️  Length {new_length} exceeds max context {max_allowed}, setting to {max_allowed}")
                            generator.config.max_new_bytes = max_allowed
                        else:
                            generator.config.max_new_bytes = new_length
                        print(f"✅ Max length set to {generator.config.max_new_bytes}")
                    except ValueError:
                        print("❌ Invalid length value")
                
                elif cmd == '/quit':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Unknown command. Type /help for available commands.")
                
                continue
            
            # Generate text
            print(f"\n🎨 Generating continuation...")
            print("-" * 40)
            
            generated = generator.generate(user_input)
            
            print(f"📝 PROMPT: {user_input}")
            print(f"🤖 GENERATED: {generated}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python inference.py <model_path>")
        print("Example: python inference.py model_checkpoint.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Create inference config
    config = InferenceConfig()
    
    # Initialize generator
    try:
        generator = ByteLevelGenerator(model_path, config)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Run interactive mode
    interactive_mode(generator)

if __name__ == "__main__":
    main()