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

def quantize_4bit_linear(tensor , per_channel = False):

    if per_channel:
        dim = 0 if tensor.dim() > 1 else None 
        min_val = tensor.min(dim=dim,keepdim=True).values
        max_val = tensor.max(dim=dim,keepdim=True).values

    else :
        min_val = tensor.min()
        max_val = tensor.max()

    scale = (max_val - min_val)/15
    zero_point = min_val

    q_tensor = torch.clamp(torch.round((tensor - min_val) / scale), 0, 15).to(torch.uint8)

    return q_tensor , zero_point , scale

def quantize_4bit_nf4(tensor):

    nf4_levels = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])

    abs_max = torch.max(torch.abs(tensor))
    normalized = tensor / abs_max
    expanded = normalized.unsqueeze(-1)
    distances = torch.abs(expanded - nf4_levels)
    indices = torch.argmin(distances, dim=-1)
    
    return indices.to(torch.uint8), abs_max, nf4_levels