"""
Tests for quantization and dequantization functions.
"""

import torch
import pytest
from Quanta.functional.quantization import (
    quantize_8bit,
    dequantize_8bit,
    quantize_4bit,
    dequantize_4bit,
    quantize_8bit_linear,
    quantize_8bit_nf8,
    quantize_8bit_fp8,
    quantize_4bit_linear,
    quantize_4bit_nf4,
    quantize_4bit_fp4
)

def test_8bit_quantization_cpu():
    # Test with a simple tensor on CPU
    tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    q_tensor, scale, zero_point = quantize_8bit(tensor)
    
    # Check quantization properties
    assert q_tensor.dtype == torch.uint8
    assert q_tensor.min() >= 0
    assert q_tensor.max() <= 255
    
    # Test dequantization
    deq_tensor = dequantize_8bit(q_tensor, scale, zero_point)
    assert torch.allclose(tensor, deq_tensor, rtol=1e-2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_8bit_quantization_cuda():
    # Test with a simple tensor on CUDA
    tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0], device='cuda')
    q_tensor, scale, zero_point = quantize_8bit(tensor)
    
    # Check quantization properties
    assert q_tensor.dtype == torch.uint8
    assert q_tensor.min() >= 0
    assert q_tensor.max() <= 255
    assert q_tensor.is_cuda
    
    # Test dequantization
    deq_tensor = dequantize_8bit(q_tensor, scale, zero_point)
    # Use both rtol and atol for CUDA tensors
    assert torch.allclose(tensor, deq_tensor, rtol=1e-2, atol=1e-5)

def test_8bit_quantization_per_channel_cpu():
    # Test with a 2D tensor on CPU
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    q_tensor, scale, zero_point = quantize_8bit(tensor, per_channel=True)
    
    assert q_tensor.dtype == torch.uint8
    assert q_tensor.shape == tensor.shape
    assert q_tensor.min() >= 0
    assert q_tensor.max() <= 255

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_8bit_quantization_per_channel_cuda():
    # Test with a 2D tensor on CUDA
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    q_tensor, scale, zero_point = quantize_8bit(tensor, per_channel=True)
    
    assert q_tensor.dtype == torch.uint8
    assert q_tensor.shape == tensor.shape
    assert q_tensor.min() >= 0
    assert q_tensor.max() <= 255
    assert q_tensor.is_cuda

def test_4bit_quantization_cpu():
    # Test with a simple tensor on CPU
    tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    q_tensor, scale, zero_point = quantize_4bit(tensor)
    
    # Check quantization properties
    assert q_tensor.dtype == torch.uint8
    assert q_tensor.min() >= 0
    assert q_tensor.max() <= 15
    
    # Test dequantization
    deq_tensor = dequantize_4bit(q_tensor, scale, zero_point)
    assert torch.allclose(tensor, deq_tensor, rtol=0.15, atol=0.15)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_4bit_quantization_cuda():
    # Test with a simple tensor on CUDA
    tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0], device='cuda')
    q_tensor, scale, zero_point = quantize_4bit(tensor)
    
    # Check quantization properties
    assert q_tensor.dtype == torch.uint8
    assert q_tensor.min() >= 0
    assert q_tensor.max() <= 15
    assert q_tensor.is_cuda
    
    # Test dequantization
    deq_tensor = dequantize_4bit(q_tensor, scale, zero_point)
    assert torch.allclose(tensor, deq_tensor, rtol=0.15, atol=0.15)

def test_quantization_edge_cases_cpu():
    # Test with zero tensor on CPU
    zero_tensor = torch.zeros(5)
    q_tensor, scale, zero_point = quantize_8bit(zero_tensor)
    assert torch.all(q_tensor == 0)
    
    # Test with constant tensor on CPU
    const_tensor = torch.ones(5) * 2.0
    q_tensor, scale, zero_point = quantize_8bit(const_tensor)
    assert torch.all(q_tensor == q_tensor[0])

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_quantization_edge_cases_cuda():
    # Test with zero tensor on CUDA
    zero_tensor = torch.zeros(5, device='cuda')
    q_tensor, scale, zero_point = quantize_8bit(zero_tensor)
    assert torch.all(q_tensor == 0)
    
    # Test with constant tensor on CUDA
    const_tensor = torch.ones(5, device='cuda') * 2.0
    q_tensor, scale, zero_point = quantize_8bit(const_tensor)
    assert torch.all(q_tensor == q_tensor[0])

if __name__ == "__main__":
    pytest.main([__file__]) 