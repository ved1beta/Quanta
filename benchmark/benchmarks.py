"""
Benchmarks for quantization functions.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
from bytesandbits.functional.quantization import (
    quantize_8bit, dequantize_8bit,
    quantize_4bit, dequantize_4bit,
    quantize_8bit_linear, quantize_8bit_nf8, quantize_8bit_fp8,
    quantize_4bit_linear, quantize_4bit_nf4, quantize_4bit_fp4
)
from bytesandbits.functional.base import BaseQuantizer
import psutil

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def benchmark_quantization_speed(tensor, num_runs=100):
    """Benchmark quantization and dequantization speed."""
    results = {}
    
    # Adjust number of runs based on tensor size
    tensor_size = tensor.numel()
    if tensor_size > 1000000:  # For large tensors
        num_runs = 10
    elif tensor_size > 100000:  # For medium tensors
        num_runs = 50
    
    # 8-bit quantization methods
    for quant_type in ["linear", "nf8", "fp8"]:
        try:
            start_time = time.time()
            for _ in range(num_runs):
                q_tensor, scale, zero_point = quantize_8bit(tensor, quant_type=quant_type)
                deq_tensor = dequantize_8bit(q_tensor, scale, zero_point, quant_type=quant_type)
            end_time = time.time()
            results[f'8bit_{quant_type}'] = (end_time - start_time) / num_runs
        except RuntimeError as e:
            if "can't allocate memory" in str(e):
                results[f'8bit_{quant_type}'] = "OOM"  # Out of Memory
            else:
                raise e
    
    # 4-bit quantization methods
    for quant_type in ["linear", "nf4", "fp4"]:
        try:
            start_time = time.time()
            for _ in range(num_runs):
                q_tensor, scale, zero_point = quantize_4bit(tensor, quant_type=quant_type)
                deq_tensor = dequantize_4bit(q_tensor, scale, zero_point, quant_type=quant_type)
            end_time = time.time()
            results[f'4bit_{quant_type}'] = (end_time - start_time) / num_runs
        except RuntimeError as e:
            if "can't allocate memory" in str(e):
                results[f'4bit_{quant_type}'] = "OOM"  # Out of Memory
            else:
                raise e
    
    # BaseQuantizer symmetric
    try:
        quantizer_sym = BaseQuantizer(num_bits=8, symmetric=True)
        start_time = time.time()
        for _ in range(num_runs):
            q_tensor, scale, zero_point = quantizer_sym.quantize(tensor)
            deq_tensor = quantizer_sym.dequantize(q_tensor, scale, zero_point)
        end_time = time.time()
        results['base_symmetric'] = (end_time - start_time) / num_runs
    except RuntimeError as e:
        if "can't allocate memory" in str(e):
            results['base_symmetric'] = "OOM"
        else:
            raise e
    
    # BaseQuantizer asymmetric
    try:
        quantizer_asym = BaseQuantizer(num_bits=8, symmetric=False)
        start_time = time.time()
        for _ in range(num_runs):
            q_tensor, scale, zero_point = quantizer_asym.quantize(tensor)
            deq_tensor = quantizer_asym.dequantize(q_tensor, scale, zero_point)
        end_time = time.time()
        results['base_asymmetric'] = (end_time - start_time) / num_runs
    except RuntimeError as e:
        if "can't allocate memory" in str(e):
            results['base_asymmetric'] = "OOM"
        else:
            raise e
    
    return results

def benchmark_memory_usage(tensor):
    """Benchmark memory usage of different quantization methods."""
    results = {}
    
    # 8-bit quantization methods
    for quant_type in ["linear", "nf8", "fp8"]:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = get_memory_usage()
        q_tensor, scale, zero_point = quantize_8bit(tensor, quant_type=quant_type)
        deq_tensor = dequantize_8bit(q_tensor, scale, zero_point, quant_type=quant_type)
        results[f'8bit_{quant_type}'] = get_memory_usage() - initial_memory
    
    # 4-bit quantization methods
    for quant_type in ["linear", "nf4", "fp4"]:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = get_memory_usage()
        q_tensor, scale, zero_point = quantize_4bit(tensor, quant_type=quant_type)
        deq_tensor = dequantize_4bit(q_tensor, scale, zero_point, quant_type=quant_type)
        results[f'4bit_{quant_type}'] = get_memory_usage() - initial_memory
    
    # BaseQuantizer
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
    
    # 8-bit quantization methods
    for quant_type in ["linear", "nf8", "fp8"]:
        q_tensor, scale, zero_point = quantize_8bit(tensor, quant_type=quant_type)
        deq_tensor = dequantize_8bit(q_tensor, scale, zero_point, quant_type=quant_type)
        mae = torch.mean(torch.abs(tensor - deq_tensor)).item()
        mse = torch.mean((tensor - deq_tensor) ** 2).item()
        results[f'8bit_{quant_type}'] = {'mae': mae, 'mse': mse}
    
    # 4-bit quantization methods
    for quant_type in ["linear", "nf4", "fp4"]:
        q_tensor, scale, zero_point = quantize_4bit(tensor, quant_type=quant_type)
        deq_tensor = dequantize_4bit(q_tensor, scale, zero_point, quant_type=quant_type)
        mae = torch.mean(torch.abs(tensor - deq_tensor)).item()
        mse = torch.mean((tensor - deq_tensor) ** 2).item()
        results[f'4bit_{quant_type}'] = {'mae': mae, 'mse': mse}
    
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
            if time_taken == "OOM":
                print(f"{method}: Out of Memory")
            else:
                print(f"{method}: {time_taken:.6f}")
        
        # Memory benchmarks
        try:
            memory_results = benchmark_memory_usage(tensor)
            print("\nMemory Usage (MB):")
            for method, memory in memory_results.items():
                print(f"{method}: {memory:.2f}")
        except RuntimeError as e:
            if "can't allocate memory" in str(e):
                print("\nMemory Usage: Out of Memory")
            else:
                raise e
        
        # Accuracy benchmarks
        try:
            accuracy_results = benchmark_accuracy(tensor)
            print("\nAccuracy Metrics:")
            for method, metrics in accuracy_results.items():
                print(f"{method}:")
                print(f"  MAE: {metrics['mae']:.6f}")
                print(f"  MSE: {metrics['mse']:.6f}")
        except RuntimeError as e:
            if "can't allocate memory" in str(e):
                print("\nAccuracy Metrics: Out of Memory")
            else:
                raise e

def benchmark_per_channel():
    """Benchmark per-channel quantization performance."""
    print("\nBenchmarking per-channel quantization...")
    
    # Create tensor with different ranges per channel
    tensor = torch.tensor([
        [1.0, 20.0, 300.0],
        [2.0, 30.0, 400.0],
        [3.0, 40.0, 500.0]
    ], dtype=torch.float32)
    
    # Compare global vs per-channel quantization for each method
    print("\nGlobal vs Per-channel Quantization:")
    
    for quant_type in ["linear", "nf8", "fp8"]:
        # Global quantization
        q_tensor_global, scale_global, zero_point_global = quantize_8bit(tensor, quant_type=quant_type)
        deq_tensor_global = dequantize_8bit(q_tensor_global, scale_global, zero_point_global, quant_type=quant_type)
        mae_global = torch.mean(torch.abs(tensor - deq_tensor_global)).item()
        
        # Per-channel quantization
        q_tensor_per_channel, scale_per_channel, zero_point_per_channel = quantize_8bit(tensor, quant_type=quant_type, per_channel=True)
        deq_tensor_per_channel = dequantize_8bit(q_tensor_per_channel, scale_per_channel, zero_point_per_channel, quant_type=quant_type)
        mae_per_channel = torch.mean(torch.abs(tensor - deq_tensor_per_channel)).item()
        
        print(f"\n{quant_type.upper()} Quantization:")
        print(f"Global quantization MAE: {mae_global:.6f}")
        print(f"Per-channel quantization MAE: {mae_per_channel:.6f}")
        print(f"Improvement: {(mae_global - mae_per_channel) / mae_global * 100:.2f}%")

if __name__ == "__main__":
    run_benchmarks()
    benchmark_per_channel() 