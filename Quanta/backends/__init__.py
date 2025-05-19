"""
Backend dispatcher for quantization operations.
This module automatically selects the appropriate backend (CPU or CUDA)
based on device availability and tensor location.
"""

import torch
from typing import Tuple, Optional
from .cpu.quantization import (
    quantize_8bit_cpu,
    dequantize_8bit_cpu,
    quantize_4bit_cpu,
    dequantize_4bit_cpu
)

# Try to import CUDA implementations
try:
    from .cuda.quantization import (
        quantize_8bit_cuda,
        dequantize_8bit_cuda,
        quantize_4bit_cuda,
        dequantize_4bit_cuda
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

def _get_backend(tensor: torch.Tensor) -> str:
    """
    Determine the appropriate backend for a tensor.
    
    Args:
        tensor: Input tensor
    
    Returns:
        'cuda' if CUDA is available and tensor is on GPU, 'cpu' otherwise
    """
    if CUDA_AVAILABLE and tensor.is_cuda:
        return 'cuda'
    return 'cpu'

def quantize_8bit(
    tensor: torch.Tensor,
    per_channel: bool = False,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to 8-bit precision using the appropriate backend.
    
    Args:
        tensor: Input tensor to quantize
        per_channel: Whether to quantize per channel
        symmetric: Whether to use symmetric quantization
    
    Returns:
        Tuple of (quantized_tensor, scale, zero_point)
    """
    backend = _get_backend(tensor)
    
    if backend == 'cuda':
        return quantize_8bit_cuda(tensor, per_channel, symmetric)
    return quantize_8bit_cpu(tensor, per_channel, symmetric)

def dequantize_8bit(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize an 8-bit tensor using the appropriate backend.
    
    Args:
        q_tensor: Quantized tensor
        scale: Scale factor
        zero_point: Zero point
    
    Returns:
        Dequantized tensor
    """
    backend = _get_backend(q_tensor)
    
    if backend == 'cuda':
        return dequantize_8bit_cuda(q_tensor, scale, zero_point)
    return dequantize_8bit_cpu(q_tensor, scale, zero_point)

def quantize_4bit(
    tensor: torch.Tensor,
    per_channel: bool = False,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to 4-bit precision using the appropriate backend.
    
    Args:
        tensor: Input tensor to quantize
        per_channel: Whether to quantize per channel
        symmetric: Whether to use symmetric quantization
    
    Returns:
        Tuple of (quantized_tensor, scale, zero_point)
    """
    backend = _get_backend(tensor)
    
    if backend == 'cuda':
        return quantize_4bit_cuda(tensor, per_channel, symmetric)
    return quantize_4bit_cpu(tensor, per_channel, symmetric)

def dequantize_4bit(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize a 4-bit tensor using the appropriate backend.
    
    Args:
        q_tensor: Quantized tensor
        scale: Scale factor
        zero_point: Zero point
    
    Returns:
        Dequantized tensor
    """
    backend = _get_backend(q_tensor)
    
    if backend == 'cuda':
        return dequantize_4bit_cuda(q_tensor, scale, zero_point)
    return dequantize_4bit_cpu(q_tensor, scale, zero_point) 