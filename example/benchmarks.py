"""
Benchmarks for quantization functions.
"""

import torch
import time
import numpy as np
from bytesandbits.functional.quantization import (
    quantize_8bit, dequantize_8bit,
    quantize_4bit, dequantize_4bit
)
from bytesandbits.functional.base import BaseQuantizer
import psutil
import os

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def benchmark_quantization_speed(tensor, num_runs=100):
    """Benchmark quantization and dequantization speed."""
    results = {}
    
    # 8-bit quantization
    start_time = time.time()
    for _ in range(num_runs):
        q_tensor, scale, zero_point = quantize_8bit(tensor)
        deq_tensor = dequantize_8bit(q_tensor, scale, zero_point)
    end_time = time.time()
    results['8bit'] = (end_time - start_time) / num_runs
    
    # 4-bit quantization
    start_time = time.time()
    for _ in range(num_runs):
        q_tensor, scale, zero_point = quantize_4bit(tensor)
        deq_tensor = dequantize_4bit(q_tensor, scale, zero_point)
    end_time = time.time()
    results['4bit'] = (end_time - start_time) / num_runs
    
    # BaseQuantizer symmetric
    quantizer_sym = BaseQuantizer(num_bits=8, symmetric=True)
    start_time = time.time()
    for _ in range(num_runs):
        q_tensor, scale, zero_point = quantizer_sym.quantize(tensor)
        deq_tensor = quantizer_sym.dequantize(q_tensor, scale, zero_point)
    end_time = time.time()
    results['base_symmetric'] = (end_time - start_time) / num_runs
    
    # BaseQuantizer asymmetric
    quantizer_asym = BaseQuantizer(num_bits=8, symmetric=False)
    start_time = time.time()
    for _ in range(num_runs):
        q_tensor, scale, zero_point = quantizer_asym.quantize(tensor)
        deq_tensor = quantizer_asym.dequantize(q_tensor, scale, zero_point)
    end_time = time.time()
    results['base_asymmetric'] = (end_time - start_time) / num_runs
    
    return results

def benchmark_memory_usage(tensor):
    """Benchmark memory usage of different quantization methods."""
    results = {}
    
    # Measure memory for 8-bit quantization
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    initial_memory = get_memory_usage()
    q_tensor, scale, zero_point = quantize_8bit(tensor)
    deq_tensor = dequantize_8bit(q_tensor, scale, zero_point)
    results['8bit'] = get_memory_usage() - initial_memory
    
    # Measure memory for 4-bit quantization
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    initial_memory = get_memory_usage()
    q_tensor, scale, zero_point = quantize_4bit(tensor)
    deq_tensor = dequantize_4bit(q_tensor, scale, zero_point)
    results['4bit'] = get_memory_usage() - initial_memory
    
    # Measure memory for BaseQuantizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    initial_memory = get_memory_usage()
    quantizer = BaseQuantizer(num_bits=8, symmetric=True)
    q_tensor, scale, zero_point = quantizer.quantize(tensor)
    deq_tensor = quantizer.dequantize(q_tensor, scale, zero_point)
    results['base_quantizer'] = get_memory_usage() - initial_memory
    
    return results

def benchmark_accuracy(tensor):
    """Benchmark accuracy of different quantization methods."""
    results = {}
    
    # 8-bit quantization accuracy
    q_tensor, scale, zero_point = quantize_8bit(tensor)
    deq_tensor = dequantize_8bit(q_tensor, scale, zero_point)
    mae_8bit = torch.mean(torch.abs(tensor - deq_tensor)).item()
    mse_8bit = torch.mean((tensor - deq_tensor) ** 2).item()
    results['8bit'] = {'mae': mae_8bit, 'mse': mse_8bit}
    
    # 4-bit quantization accuracy
    q_tensor, scale, zero_point = quantize_4bit(tensor)
    deq_tensor = dequantize_4bit(q_tensor, scale, zero_point)
    mae_4bit = torch.mean(torch.abs(tensor - deq_tensor)).item()
    mse_4bit = torch.mean((tensor - deq_tensor) ** 2).item()
    results['4bit'] = {'mae': mae_4bit, 'mse': mse_4bit}
    
    # BaseQuantizer accuracy (symmetric)
    quantizer_sym = BaseQuantizer(num_bits=8, symmetric=True)
    q_tensor, scale, zero_point = quantizer_sym.quantize(tensor)
    deq_tensor = quantizer_sym.dequantize(q_tensor, scale, zero_point)
    mae_sym = torch.mean(torch.abs(tensor - deq_tensor)).item()
    mse_sym = torch.mean((tensor - deq_tensor) ** 2).item()
    results['base_symmetric'] = {'mae': mae_sym, 'mse': mse_sym}
    
    # BaseQuantizer accuracy (asymmetric)
    quantizer_asym = BaseQuantizer(num_bits=8, symmetric=False)
    q_tensor, scale, zero_point = quantizer_asym.quantize(tensor)
    deq_tensor = quantizer_asym.dequantize(q_tensor, scale, zero_point)
    mae_asym = torch.mean(torch.abs(tensor - deq_tensor)).item()
    mse_asym = torch.mean((tensor - deq_tensor) ** 2).item()
    results['base_asymmetric'] = {'mae': mae_asym, 'mse': mse_asym}
    
    return results

def run_benchmarks():
    """Run all benchmarks with different tensor sizes and types."""
    print("Running quantization benchmarks...")
    
    # Test different tensor sizes
    tensor_sizes = [
        (100, 100),      # Small
        (1000, 1000),    # Medium
        (5000, 5000)     # Large
    ]
    
    for size in tensor_sizes:
        print(f"\nBenchmarking with tensor size {size}:")
        
        # Generate random tensor
        tensor = torch.randn(size, dtype=torch.float32)
        
        # Speed benchmarks
        speed_results = benchmark_quantization_speed(tensor)
        print("\nSpeed (seconds per operation):")
        for method, time_taken in speed_results.items():
            print(f"{method}: {time_taken:.6f}")
        
        # Memory benchmarks
        memory_results = benchmark_memory_usage(tensor)
        print("\nMemory Usage (MB):")
        for method, memory in memory_results.items():
            print(f"{method}: {memory:.2f}")
        
        # Accuracy benchmarks
        accuracy_results = benchmark_accuracy(tensor)
        print("\nAccuracy Metrics:")
        for method, metrics in accuracy_results.items():
            print(f"{method}:")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  MSE: {metrics['mse']:.6f}")

def benchmark_per_channel():
    """Benchmark per-channel quantization performance."""
    print("\nBenchmarking per-channel quantization...")
    
    # Create tensor with different ranges per channel
    tensor = torch.tensor([
        [1.0, 20.0, 300.0],
        [2.0, 30.0, 400.0],
        [3.0, 40.0, 500.0]
    ], dtype=torch.float32)
    
    # Compare global vs per-channel quantization
    print("\nGlobal vs Per-channel Quantization:")
    
    # Global quantization
    q_tensor_global, scale_global, zero_point_global = quantize_8bit(tensor)
    deq_tensor_global = dequantize_8bit(q_tensor_global, scale_global, zero_point_global)
    mae_global = torch.mean(torch.abs(tensor - deq_tensor_global)).item()
    
    # Per-channel quantization
    q_tensor_per_channel, scale_per_channel, zero_point_per_channel = quantize_8bit(tensor, per_channel=True)
    deq_tensor_per_channel = dequantize_8bit(q_tensor_per_channel, scale_per_channel, zero_point_per_channel)
    mae_per_channel = torch.mean(torch.abs(tensor - deq_tensor_per_channel)).item()
    
    print(f"Global quantization MAE: {mae_global:.6f}")
    print(f"Per-channel quantization MAE: {mae_per_channel:.6f}")
    print(f"Improvement: {(mae_global - mae_per_channel) / mae_global * 100:.2f}%")

if __name__ == "__main__":
    run_benchmarks()
    benchmark_per_channel() 