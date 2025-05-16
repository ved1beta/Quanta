"""
Quantization and dequantization functions.
"""

import torch

def quantize_8bit(tensor, quant_type="linear", per_channel=False):
    """
    Quantize a floating-point tensor to 8-bit precision.
    """
    if quant_type == "linear":
        return quantize_8bit_linear(tensor, per_channel)
    elif quant_type == "nf8":
        return quantize_8bit_nf8(tensor)
    elif quant_type == "fp8":
        return quantize_8bit_fp8(tensor)
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")

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

def quantize_8bit_nf8(tensor):

    nf8_levels = torch.linspace(-1 , 1, 256)
    nf8_levels = torch.tanh(nf8_levels * 2)

    abs_max = torch.max(torch.abs(tensor))
    normalized = tensor / abs_max
    expanded = normalized.unsqueeze(-1)
    distances = torch.abs(expanded - nf8_levels)
    indices = torch.argmin(distances, dim=-1)
    
    return indices.to(torch.uint8), abs_max, nf8_levels

def quantize_4bit_fp4(tensor):
    signs = torch.sign(tensor)
    abs_values = torch.abs(tensor)

    log_values  =torch.log2(abs_values + ( abs_values == 0 ).float())
    exp_bias = 1 
    exp_values = torch.clamp(torch.round(log_values + exp_bias), 0, 3)
    mantissa_values = torch.round((abs_values / (2 ** (exp_values - exp_bias))) - 1)
    mantissa_values = torch.clamp(mantissa_values, 0, 1)

    q_tensor = ((exp_values << 1) | mantissa_values).to(torch.uint8)
    q_tensor = torch.where(signs < 0, q_tensor | 0x8, q_tensor)
    
    return q_tensor, exp_bias

def quantize_8bit_fp8(tensor):
    signs = torch.sign(tensor)
    abs_values = torch.abs(tensor)

    log_values = torch.log2(abs_values + (abs_values == 0).float())

    exp_bias = 7 
    exp_values = torch.clamp( torch.round(log_values + exp_bias), 0 , 15)

    mantissa_values = torch.round((abs_values / (2 ** (exp_values - exp_bias))) * 8 - 8)
    mantissa_values = torch.clamp(mantissa_values, 0, 7)

    q_tensor = ((exp_values << 3) | mantissa_values).to(torch.uint8)
    
    q_tensor = torch.where(signs < 0, q_tensor | 0x80, q_tensor)
    
    return q_tensor, exp_bias


def quantize_8bit_linear(tensor, per_channel=False):
    """
    Linear 8-bit quantization with 256 levels (0-255).
    """
    if per_channel:
        dim = 0 if tensor.dim() > 1 else None
        min_val = tensor.min(dim=dim, keepdim=True).values
        max_val = tensor.max(dim=dim, keepdim=True).values
    else:
        min_val = tensor.min()
        max_val = tensor.max()
    
    scale = (max_val - min_val) / 255
    zero_point = min_val
    
    q_tensor = torch.clamp(torch.round((tensor - min_val) / scale), 0, 255).to(torch.uint8)
    return q_tensor, scale, zero_point
    