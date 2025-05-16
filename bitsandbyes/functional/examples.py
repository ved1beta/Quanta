"""
Examples demonstrating the usage of quantization functions.
"""

import torch
import numpy as np
from .quantization import quantize_8bit, dequantize_8bit, quantize_4bit, dequantize_4bit
from .base import BaseQuantizer

def example_8bit_quantization():
    """Example of basic 8-bit quantization and dequantization."""
    # Create a sample tensor
    tensor = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]], dtype=torch.float32)
    
    # Quantize
    q_tensor, scale, zero_point = quantize_8bit(tensor)
    print("\n8-bit Quantization Example:")
    print("Original tensor:\n", tensor)
    print("Quantized tensor:\n", q_tensor)
    print("Scale:", scale)
    print("Zero point:", zero_point)
    
    # Dequantize
    deq_tensor = dequantize_8bit(q_tensor, scale, zero_point)
    print("Dequantized tensor:\n", deq_tensor)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor)).item())

def example_per_channel_8bit():
    """Example of per-channel 8-bit quantization."""
    # Create a sample tensor with different ranges per channel
    tensor = torch.tensor([[1.0, 20.0, 300.0],
                          [2.0, 30.0, 400.0],
                          [3.0, 40.0, 500.0]], dtype=torch.float32)
    
    # Quantize per channel
    q_tensor, scale, zero_point = quantize_8bit(tensor, per_channel=True)
    print("\nPer-channel 8-bit Quantization Example:")
    print("Original tensor:\n", tensor)
    print("Quantized tensor:\n", q_tensor)
    print("Scale per channel:", scale)
    print("Zero point per channel:", zero_point)
    
    # Dequantize
    deq_tensor = dequantize_8bit(q_tensor, scale, zero_point)
    print("Dequantized tensor:\n", deq_tensor)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor)).item())

def example_4bit_quantization():
    """Example of 4-bit quantization."""
    # Create a sample tensor
    tensor = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]], dtype=torch.float32)
    
    # Quantize
    q_tensor, scale, zero_point = quantize_4bit(tensor)
    print("\n4-bit Quantization Example:")
    print("Original tensor:\n", tensor)
    print("Quantized tensor:\n", q_tensor)
    print("Scale:", scale)
    print("Zero point:", zero_point)
    
    # Dequantize
    deq_tensor = dequantize_4bit(q_tensor, scale, zero_point)
    print("Dequantized tensor:\n", deq_tensor)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor)).item())

def example_base_quantizer():
    """Example using the BaseQuantizer class."""
    # Create a sample tensor
    tensor = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]], dtype=torch.float32)
    
    # Create quantizer instances
    symmetric_quantizer = BaseQuantizer(num_bits=8, symmetric=True)
    asymmetric_quantizer = BaseQuantizer(num_bits=8, symmetric=False)
    
    # Quantize using symmetric quantizer
    q_tensor_sym, scale_sym, zero_point_sym = symmetric_quantizer.quantize(tensor)
    deq_tensor_sym = symmetric_quantizer.dequantize(q_tensor_sym, scale_sym, zero_point_sym)
    
    # Quantize using asymmetric quantizer
    q_tensor_asym, scale_asym, zero_point_asym = asymmetric_quantizer.quantize(tensor)
    deq_tensor_asym = asymmetric_quantizer.dequantize(q_tensor_asym, scale_asym, zero_point_asym)
    
    print("\nBaseQuantizer Example:")
    print("Original tensor:\n", tensor)
    print("\nSymmetric quantization:")
    print("Quantized tensor:\n", q_tensor_sym)
    print("Scale:", scale_sym)
    print("Zero point:", zero_point_sym)
    print("Dequantized tensor:\n", deq_tensor_sym)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor_sym)).item())
    
    print("\nAsymmetric quantization:")
    print("Quantized tensor:\n", q_tensor_asym)
    print("Scale:", scale_asym)
    print("Zero point:", zero_point_asym)
    print("Dequantized tensor:\n", deq_tensor_asym)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor_asym)).item())

def run_all_examples():
    """Run all quantization examples."""
    example_8bit_quantization()
    example_per_channel_8bit()
    example_4bit_quantization()
    example_base_quantizer()

if __name__ == "__main__":
    run_all_examples() 