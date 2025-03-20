"""
Benchmark script to measure the speedup provided by KV caching in the attention layer.
This script tests only the attention layer with random initialization.
"""

import torch
import argparse
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

from model_550m import Attention
from kv_cache import KVCache


@dataclass
class AttentionConfig:
    """Configuration for the attention layer."""
    dim: int
    num_heads: int
    bias: bool = False
    use_alibi: bool = False
    window_size: int = 0  # Set to 0 instead of None for Flash Attention
    softcap: float = 0.0  # Set to 0.0 instead of None
    dropout: float = 0.0
    use_rotary: bool = True
    max_seq_len: int = 8192


def precompute_freqs_cis(dim, end, theta=10000.0, device="cuda", dtype=torch.bfloat16):
    """
    Precompute the frequency cis for rotary embeddings.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(dtype=dtype)
    return freqs_cis


def benchmark_attention(
    batch_size=1,
    seq_lengths=[128, 256, 512, 1024],
    dim=2048,
    num_heads=32,
    window_size=0,
    softcap=0.0,
    num_runs=3,
    device="cuda",
    max_new_tokens=50,
    dtype=torch.bfloat16,
):
    """Benchmark the attention layer with and without KV caching."""
    
    # Create config for attention layer
    head_dim = dim // num_heads
    max_possible_len = max(seq_lengths) + max_new_tokens
    
    config = AttentionConfig(
        dim=dim,
        num_heads=num_heads,
        bias=False,
        use_alibi=False,
        window_size=window_size,
        softcap=softcap,
        dropout=0.0,
        use_rotary=True,
        max_seq_len=max_possible_len
    )
    
    print(f"Initializing attention layer with dim={dim}, num_heads={num_heads}, window_size={window_size}...")
    print(f"Using data type: {dtype}")
    
    attention = Attention(config)
    attention = attention.to(device=device, dtype=dtype)
    attention.eval()
    
    # Precompute freqs_cis for rotary embeddings with correct dtype
    freqs_cis = precompute_freqs_cis(head_dim, max_possible_len, device=device, dtype=dtype)
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nBenchmarking with sequence length: {seq_len}")
        
        # Results for this sequence length
        seq_results = {
            "seq_len": seq_len,
            "with_cache": [],
            "without_cache": [],
        }
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            
            # Create random input with correct dtype
            x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
            
            # Clear caches
            torch.cuda.empty_cache()
            
            # ----- Test without KV cache -----
            print("    Testing without KV cache...")
            start_time = time.time()
            
            # Process the prompt-like initial sequence
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = attention(x, freqs_cis=freqs_cis)
            
            prompt_time_no_cache = time.time() - start_time
            gen_start_time = time.time()
            
            # Simulate token-by-token generation without cache
            for i in range(max_new_tokens):
                # For each new token, process the entire growing sequence
                new_x = torch.randn(batch_size, seq_len + i + 1, dim, device=device, dtype=dtype)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _ = attention(new_x, freqs_cis=freqs_cis)
            
            gen_time_no_cache = time.time() - gen_start_time
            total_time_no_cache = time.time() - start_time
            
            # ----- Test with KV cache -----
            print("    Testing with KV cache...")
            if hasattr(attention, 'reset_kv_cache'):
                attention.reset_kv_cache()  # Ensure cache is reset
            
            # Setup caches for this test
            max_seq_len = seq_len + max_new_tokens
            if hasattr(attention, 'setup_kv_cache'):
                attention.setup_kv_cache(batch_size=batch_size, max_seq_len=max_seq_len)
            
            start_time = time.time()
            
            # Process the prompt-like initial sequence with cache
            input_pos = torch.arange(seq_len, device=device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = attention(x, input_pos=input_pos, freqs_cis=freqs_cis)
            
            prompt_time_with_cache = time.time() - start_time
            gen_start_time = time.time()
            
            # Simulate token-by-token generation with cache
            curr_seq_len = seq_len
            for i in range(max_new_tokens):
                # For cached generation, we only need to process the new token
                new_token = torch.randn(batch_size, 1, dim, device=device, dtype=dtype)
                curr_pos = torch.tensor([curr_seq_len], device=device)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _ = attention(new_token, input_pos=curr_pos, freqs_cis=freqs_cis)
                curr_seq_len += 1
            
            gen_time_with_cache = time.time() - gen_start_time
            total_time_with_cache = time.time() - start_time
            
            # Clean up
            if hasattr(attention, 'reset_kv_cache'):
                attention.reset_kv_cache()
            torch.cuda.empty_cache()
            
            # Record results for this run
            seq_results["with_cache"].append({
                "prompt_time": prompt_time_with_cache,
                "generation_time": gen_time_with_cache,
                "total_time": total_time_with_cache,
                "tokens_per_second": max_new_tokens / gen_time_with_cache if gen_time_with_cache > 0 else 0,
            })
            
            seq_results["without_cache"].append({
                "prompt_time": prompt_time_no_cache,
                "generation_time": gen_time_no_cache,
                "total_time": total_time_no_cache,
                "tokens_per_second": max_new_tokens / gen_time_no_cache if gen_time_no_cache > 0 else 0,
            })
        
        # Calculate averages for this sequence length
        avg_with_cache = {
            "prompt_time": np.mean([r["prompt_time"] for r in seq_results["with_cache"]]),
            "generation_time": np.mean([r["generation_time"] for r in seq_results["with_cache"]]),
            "total_time": np.mean([r["total_time"] for r in seq_results["with_cache"]]),
            "tokens_per_second": np.mean([r["tokens_per_second"] for r in seq_results["with_cache"]]),
        }
        
        avg_without_cache = {
            "prompt_time": np.mean([r["prompt_time"] for r in seq_results["without_cache"]]),
            "generation_time": np.mean([r["generation_time"] for r in seq_results["without_cache"]]),
            "total_time": np.mean([r["total_time"] for r in seq_results["without_cache"]]),
            "tokens_per_second": np.mean([r["tokens_per_second"] for r in seq_results["without_cache"]]),
        }
        
        # Calculate speedup
        speedup = avg_with_cache["tokens_per_second"] / avg_without_cache["tokens_per_second"] if avg_without_cache["tokens_per_second"] > 0 else float('inf')
        
        # Print comparison
        print(f"\nResults for sequence length {seq_len}:")
        print(f"  With KV cache: {avg_with_cache['tokens_per_second']:.2f} tokens/sec")
        print(f"  Without KV cache: {avg_without_cache['tokens_per_second']:.2f} tokens/sec")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Store results
        results.append({
            "seq_len": seq_len,
            "with_cache": avg_with_cache,
            "without_cache": avg_without_cache,
            "speedup": speedup,
        })
    
    return results


def plot_results(results):
    """Plot the benchmark results."""
    seq_lengths = [r["seq_len"] for r in results]
    speedups = [r["speedup"] for r in results]
    tokens_per_sec_with_cache = [r["with_cache"]["tokens_per_second"] for r in results]
    tokens_per_sec_without_cache = [r["without_cache"]["tokens_per_second"] for r in results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot speedup
    ax1.plot(seq_lengths, speedups, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Speedup Factor (x)')
    ax1.set_title('KV Cache Speedup vs. Sequence Length')
    ax1.grid(True)
    
    # Plot tokens per second
    ax2.plot(seq_lengths, tokens_per_sec_with_cache, 'o-', linewidth=2, label='With KV Cache', color='green')
    ax2.plot(seq_lengths, tokens_per_sec_without_cache, 'o-', linewidth=2, label='Without KV Cache', color='red')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Tokens per Second')
    ax2.set_title('Generation Speed vs. Sequence Length')
    ax2.legend()
    ax2.grid(True)
    
    # Add logarithmic scale for y-axis if values vary a lot
    if max(tokens_per_sec_with_cache) / min(tokens_per_sec_without_cache) > 10:
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('attention_kvcache_benchmark.png')
    print("Saved benchmark plot to 'attention_kvcache_benchmark.png'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache acceleration for attention layer")
    parser.add_argument("--dim", type=int, default=2048, help="Hidden dimension size")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of benchmark runs per setting")
    parser.add_argument("--window_size", type=int, default=0, help="Window size for attention (0 for no window)")
    parser.add_argument("--softcap", type=float, default=0.0, help="Softcap value (0.0 for no softcap)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, cpu)")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="bf16", 
                        help="Data type to use (bf16 or fp16)")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Set dtype
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    # Define sequence lengths to test - reduced max length for faster testing
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Run benchmark
    results = benchmark_attention(
        batch_size=args.batch_size,
        seq_lengths=seq_lengths,
        dim=args.dim,
        num_heads=args.num_heads,
        window_size=args.window_size,
        softcap=args.softcap,
        num_runs=args.num_runs,
        device=device,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
    )
    
    # Print overall results
    print("\n" + "="*50)
    print("OVERALL BENCHMARK RESULTS")
    print("="*50)
    
    for result in results:
        seq_len = result["seq_len"]
        speedup = result["speedup"]
        tokens_with_cache = result["with_cache"]["tokens_per_second"]
        tokens_without_cache = result["without_cache"]["tokens_per_second"]
        
        print(f"Sequence length {seq_len}:")
        print(f"  With KV cache: {tokens_with_cache:.2f} tokens/sec")
        print(f"  Without KV cache: {tokens_without_cache:.2f} tokens/sec")
        print(f"  Speedup: {speedup:.2f}x")
        print()
    
    # Calculate overall average speedup
    avg_speedup = np.mean([r["speedup"] for r in results])
    print(f"Average speedup across all sequence lengths: {avg_speedup:.2f}x")
    
    # Plot results
    plot_results(results)


if __name__ == "__main__":
    main() 