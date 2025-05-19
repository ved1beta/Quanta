"""
CPU implementation of quantization operations.
This module provides pure CPU implementations of quantization operations
that can be used as fallbacks when CUDA is not available.
"""

import torch
from typing import Tuple, Optional

def quantize_8bit_cpu(
    tensor: torch.Tensor,
    per_channel: bool = False,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU implementation of 8-bit quantization.
    
    Args:
        tensor: Input tensor to quantize
        per_channel: Whether to quantize per channel
        symmetric: Whether to use symmetric quantization
    
    Returns:
        Tuple of (quantized_tensor, scale, zero_point)
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    if per_channel:
        dim = 0 if tensor.dim() > 1 else None
        min_val = tensor.min(dim=dim, keepdim=True).values
        max_val = tensor.max(dim=dim, keepdim=True).values
    else:
        min_val = tensor.min()
        max_val = tensor.max()
    
    # Handle edge case where min_val == max_val
    if torch.allclose(min_val, max_val):
        return torch.zeros_like(tensor, dtype=torch.uint8), torch.ones_like(min_val), min_val
    
    if symmetric:
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = (2**7 - 1) / abs_max
        zero_point = torch.zeros_like(min_val)
        q_tensor = torch.clamp(
            torch.round(tensor * scale),
            -127, 127
        ).to(torch.int8)
        # Convert to unsigned
        q_tensor = (q_tensor + 128).to(torch.uint8)
    else:
        scale = (2**8 - 1) / (max_val - min_val)
        zero_point = torch.round(-min_val * scale)
        q_tensor = torch.clamp(
            torch.round(tensor * scale + zero_point),
            0, 255
        ).to(torch.uint8)
    
    return q_tensor, scale, zero_point

def dequantize_8bit_cpu(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    CPU implementation of 8-bit dequantization.
    
    Args:
        q_tensor: Quantized tensor
        scale: Scale factor
        zero_point: Zero point
    
    Returns:
        Dequantized tensor
    """
    if not q_tensor.is_contiguous():
        q_tensor = q_tensor.contiguous()
    
    # Convert from unsigned to signed if zero_point is 0 (symmetric case)
    if torch.allclose(zero_point, torch.zeros_like(zero_point)):
        q_tensor = q_tensor.to(torch.int8) - 128
    
    return (q_tensor.float() - zero_point) / scale

def quantize_4bit_cpu(
    tensor: torch.Tensor,
    per_channel: bool = False,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU implementation of 4-bit quantization.
    
    Args:
        tensor: Input tensor to quantize
        per_channel: Whether to quantize per channel
        symmetric: Whether to use symmetric quantization
    
    Returns:
        Tuple of (quantized_tensor, scale, zero_point)
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    if per_channel:
        dim = 0 if tensor.dim() > 1 else None
        min_val = tensor.min(dim=dim, keepdim=True).values
        max_val = tensor.max(dim=dim, keepdim=True).values
    else:
        min_val = tensor.min()
        max_val = tensor.max()
    
    # Handle edge case where min_val == max_val
    if torch.allclose(min_val, max_val):
        return torch.zeros_like(tensor, dtype=torch.uint8), torch.ones_like(min_val), min_val
    
    if symmetric:
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = (2**3 - 1) / abs_max
        zero_point = torch.zeros_like(min_val)
        q_tensor = torch.clamp(
            torch.round(tensor * scale),
            -7, 7
        ).to(torch.int8)
        # Convert to unsigned
        q_tensor = (q_tensor + 8).to(torch.uint8)
    else:
        scale = (2**4 - 1) / (max_val - min_val)
        zero_point = torch.round(-min_val * scale)
        q_tensor = torch.clamp(
            torch.round(tensor * scale + zero_point),
            0, 15
        ).to(torch.uint8)
    
    return q_tensor, scale, zero_point

def dequantize_4bit_cpu(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    CPU implementation of 4-bit dequantization.
    
    Args:
        q_tensor: Quantized tensor
        scale: Scale factor
        zero_point: Zero point
    
    Returns:
        Dequantized tensor
    """
    if not q_tensor.is_contiguous():
        q_tensor = q_tensor.contiguous()
    
    # Convert from unsigned to signed if zero_point is 0 (symmetric case)
    if torch.allclose(zero_point, torch.zeros_like(zero_point)):
        q_tensor = q_tensor.to(torch.int8) - 8
    
    return (q_tensor.float() - zero_point) / scale 