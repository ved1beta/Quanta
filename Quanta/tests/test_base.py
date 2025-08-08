"""
Tests and examples for the BaseQuantizer class.
"""

import torch
import pytest
from Quanta.functional.base import BaseQuantizer

def test_symmetric_quantization():
    """Test symmetric quantization with different bit widths."""
    # Test with 8-bit quantization
    quantizer_8bit = BaseQuantizer(num_bits=8, symmetric=True)
    tensor = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
    
    q_tensor, scale, zero_point = quantizer_8bit.quantize(tensor)
    deq_tensor = quantizer_8bit.dequantize(q_tensor, scale, zero_point)
    
    print("\nSymmetric 8-bit Quantization Test:")
    print("Original tensor:", tensor)
    print("Quantized tensor:", q_tensor)
    print("Scale:", scale)
    print("Zero point:", zero_point)
    print("Dequantized tensor:", deq_tensor)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor)).item())
    
    # Test with 4-bit quantization
    quantizer_4bit = BaseQuantizer(num_bits=4, symmetric=True)
    q_tensor_4bit, scale_4bit, zero_point_4bit = quantizer_4bit.quantize(tensor)
    deq_tensor_4bit = quantizer_4bit.dequantize(q_tensor_4bit, scale_4bit, zero_point_4bit)
    
    print("\nSymmetric 4-bit Quantization Test:")
    print("Original tensor:", tensor)
    print("Quantized tensor:", q_tensor_4bit)
    print("Scale:", scale_4bit)
    print("Zero point:", zero_point_4bit)
    print("Dequantized tensor:", deq_tensor_4bit)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor_4bit)).item())

def test_asymmetric_quantization():
    """Test asymmetric quantization with different bit widths."""
    # Test with 8-bit quantization
    quantizer_8bit = BaseQuantizer(num_bits=8, symmetric=False)
    tensor = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    
    q_tensor, scale, zero_point = quantizer_8bit.quantize(tensor)
    deq_tensor = quantizer_8bit.dequantize(q_tensor, scale, zero_point)
    
    print("\nAsymmetric 8-bit Quantization Test:")
    print("Original tensor:", tensor)
    print("Quantized tensor:", q_tensor)
    print("Scale:", scale)
    print("Zero point:", zero_point)
    print("Dequantized tensor:", deq_tensor)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor)).item())

def test_per_channel_quantization():
    """Test per-channel quantization with a 2D tensor."""
    quantizer = BaseQuantizer(num_bits=8, symmetric=True)
    tensor = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]], dtype=torch.float32)
    
    q_tensor, scale, zero_point = quantizer.quantize(tensor, per_channel=True)
    deq_tensor = quantizer.dequantize(q_tensor, scale, zero_point)
    
    print("\nPer-channel Quantization Test:")
    print("Original tensor:\n", tensor)
    print("Quantized tensor:\n", q_tensor)
    print("Scale per channel:", scale)
    print("Zero point per channel:", zero_point)
    print("Dequantized tensor:\n", deq_tensor)
    print("Mean absolute error:", torch.mean(torch.abs(tensor - deq_tensor)).item())

def test_edge_cases():
    """Test edge cases and error handling."""
    quantizer = BaseQuantizer(num_bits=8, symmetric=True)
    
    # Test with constant tensor
    constant_tensor = torch.ones(3, 3) * 2.0
    q_tensor, scale, zero_point = quantizer.quantize(constant_tensor)
    deq_tensor = quantizer.dequantize(q_tensor, scale, zero_point)
    
    print("\nEdge Case Test - Constant Tensor:")
    print("Original tensor:\n", constant_tensor)
    print("Quantized tensor:\n", q_tensor)
    print("Scale:", scale)
    print("Zero point:", zero_point)
    print("Dequantized tensor:\n", deq_tensor)
    print("Mean absolute error:", torch.mean(torch.abs(constant_tensor - deq_tensor)).item())
    
    # Test with very small values
    small_tensor = torch.tensor([1e-6, 2e-6, 3e-6], dtype=torch.float32)
    q_tensor, scale, zero_point = quantizer.quantize(small_tensor)
    deq_tensor = quantizer.dequantize(q_tensor, scale, zero_point)
    
    print("\nEdge Case Test - Small Values:")
    print("Original tensor:", small_tensor)
    print("Quantized tensor:", q_tensor)
    print("Scale:", scale)
    print("Zero point:", zero_point)
    print("Dequantized tensor:", deq_tensor)
    print("Mean absolute error:", torch.mean(torch.abs(small_tensor - deq_tensor)).item())

def run_all_tests():
    """Run all tests and demonstrations."""
    test_symmetric_quantization()
    test_asymmetric_quantization()
    test_per_channel_quantization()
    test_edge_cases()

if __name__ == "__main__":
    run_all_tests() 