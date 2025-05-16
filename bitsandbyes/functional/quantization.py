"""
Quantization and dequantization functions.
"""

import torch

def quantize_8bit(tensor, per_channel=False):
    """
    Quantize a floating-point tensor to 8-bit precision.
    """
    if per_channel:
        dim = 0 if tensor.dim() > 1 else None
        min_val = tensor.min(dim=dim, keepdim=True).values
        max_val = tensor.max(dim=dim, keepdim=True).values
    else:
        min_val = tensor.min()
        max_val = tensor.max()
    
    # Handle edge case where min_val == max_val
    if torch.allclose(min_val, max_val):
        return torch.zeros_like(tensor, dtype=torch.uint8), 1.0, min_val
    
    scale = (max_val - min_val) / 255
    zero_point = min_val
    
    # Ensure scale is not zero
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    q_tensor = torch.clamp(torch.round((tensor - min_val) / scale), 0, 255).to(torch.uint8)
    return q_tensor, scale, zero_point

def dequantize_8bit(q_tensor, scale, zero_point):
    """
    Dequantize an 8-bit tensor back to floating-point.
    """
    return q_tensor.float() * scale + zero_point

def quantize_4bit(tensor, quant_type="nf4"):
    """
    Quantize a floating-point tensor to 4-bit precision.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Handle edge case where min_val == max_val
    if torch.allclose(min_val, max_val):
        return torch.zeros_like(tensor, dtype=torch.uint8), 1.0, min_val
    
    scale = (max_val - min_val) / 15
    zero_point = min_val
    
    # Ensure scale is not zero
    if scale == 0:
        scale = 1.0
    
    q_tensor = torch.clamp(torch.round((tensor - min_val) / scale), 0, 15).to(torch.uint8)
    return q_tensor, scale, zero_point

def dequantize_4bit(q_tensor, scale, zero_point):
    """
    Dequantize a 4-bit tensor back to floating-point.
    """
    return q_tensor.float() * scale + zero_point 