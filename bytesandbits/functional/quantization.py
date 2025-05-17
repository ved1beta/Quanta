import torch

def quantize_4bit(tensor, quant_type="linear", per_channel=False, symmetric=False):
    """
    Quantize a floating-point tensor to 4-bit precision.
    
    Args:
        tensor: Input tensor to quantize
        quant_type: Type of quantization ("linear", "nf4", "fp4")
        per_channel: Whether to quantize per channel
        symmetric: Whether to use symmetric quantization (for linear only)
    """
    if quant_type == "linear":
        return quantize_4bit_linear(tensor, per_channel, symmetric)
    elif quant_type == "nf4":
        return quantize_4bit_nf4(tensor)
    elif quant_type == "fp4":
        return quantize_4bit_fp4(tensor)
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")

def quantize_8bit(tensor, quant_type="linear", per_channel=False, symmetric=False):
    """
    Quantize a floating-point tensor to 8-bit precision.
    
    Args:
        tensor: Input tensor to quantize
        quant_type: Type of quantization ("linear", "nf8", "fp8")
        per_channel: Whether to quantize per channel
        symmetric: Whether to use symmetric quantization (for linear only)
    """
    if quant_type == "linear":
        return quantize_8bit_linear(tensor, per_channel, symmetric)
    elif quant_type == "nf8":
        return quantize_8bit_nf8(tensor)
    elif quant_type == "fp8":
        return quantize_8bit_fp8(tensor)
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")

def dequantize_8bit(q_tensor, scale, zero_point, quant_type="linear", symmetric=False):
    """
    Dequantize an 8-bit tensor back to floating point.
    
    Args:
        q_tensor: Quantized tensor
        scale: Scale factor or levels for dequantization
        zero_point: Zero point offset or bias
        quant_type: Type of quantization used ("linear", "nf8", "fp8")
        symmetric: Whether symmetric quantization was used (for linear only)
    """
    if quant_type == "linear":
        if symmetric:
            return q_tensor.float() * scale
        else:
            return q_tensor.float() * scale + zero_point
    elif quant_type == "nf8":
        return scale[q_tensor.long()] * zero_point
    elif quant_type == "fp8":
        # Convert to uint8 for bitwise operations
        q_tensor = q_tensor.to(torch.uint8)
        signs = torch.where(q_tensor & 0x80, -1.0, 1.0)
        exp_values = (q_tensor >> 3) & 0x0F
        mantissa_values = q_tensor & 0x07
        values = (1.0 + mantissa_values.float() / 8.0) * (2.0 ** (exp_values.float() - zero_point))
        return values * signs
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")

def dequantize_4bit(q_tensor, scale, zero_point, quant_type="linear", symmetric=False):
    """
    Dequantize a 4-bit tensor back to floating point.
    
    Args:
        q_tensor: Quantized tensor
        scale: Scale factor or levels for dequantization
        zero_point: Zero point offset or bias
        quant_type: Type of quantization used ("linear", "nf4", "fp4")
        symmetric: Whether symmetric quantization was used (for linear only)
    """
    if quant_type == "linear":
        if symmetric:
            return q_tensor.float() * scale
        else:
            return q_tensor.float() * scale + zero_point
    elif quant_type == "nf4":
        # scale is the nf4_levels tensor
        return scale[q_tensor.long()] * zero_point
    elif quant_type == "fp4":
        # Convert to uint8 for bitwise operations
        q_tensor = q_tensor.to(torch.uint8)
        signs = torch.where(q_tensor & 0x8, -1.0, 1.0)
        exp_values = (q_tensor >> 1) & 0x3
        mantissa_values = q_tensor & 0x1
        values = (1.0 + mantissa_values.float()) * (2.0 ** (exp_values.float() - zero_point))
        return values * signs
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")

def quantize_4bit_linear(tensor, per_channel=False, symmetric=False):
    """
    Linear 4-bit quantization with 16 levels.
    
    Args:
        tensor: Input tensor to quantize
        per_channel: Whether to quantize per channel
        symmetric: Whether to use symmetric quantization
    """
    if symmetric:
        if per_channel:
            dim = 0 if tensor.dim() > 1 else None
            abs_max = torch.max(torch.abs(tensor), dim=dim, keepdim=True).values
            
            abs_max = torch.clamp(abs_max, min=1e-6)
        else:
            abs_max = torch.max(torch.abs(tensor))
            
            if abs_max == 0:
                abs_max = 1e-6
                
        scale = abs_max / 7.0  # Use 7 as max to avoid overflow
        q_tensor = torch.clamp(torch.round(tensor / scale) + 8, 0, 15).to(torch.uint8)
        zero_point = 0  # In symmetric, conceptually zero_point is 0
        
        return q_tensor, scale, zero_point
    else:
        if per_channel:
            dim = 0 if tensor.dim() > 1 else None 
            min_val = tensor.min(dim=dim, keepdim=True).values
            max_val = tensor.max(dim=dim, keepdim=True).values
            
            # Ensure we have a non-zero range for each channel
            mask = (max_val == min_val)
            max_val = torch.where(mask, min_val + 1e-6, max_val)
        else:
            min_val = tensor.min()
            max_val = tensor.max()
            
            # Ensure we have a non-zero range
            if max_val == min_val:
                max_val = min_val + 1e-6

        scale = (max_val - min_val) / 15
        zero_point = min_val

        q_tensor = torch.clamp(torch.round((tensor - min_val) / scale), 0, 15).to(torch.uint8)

        return q_tensor, scale, zero_point

def quantize_8bit_linear(tensor, per_channel=False, symmetric=False):
    """
    Linear 8-bit quantization with 256 levels.
    
    Args:
        tensor: Input tensor to quantize
        per_channel: Whether to quantize per channel
        symmetric: Whether to use symmetric quantization
    """
    if symmetric:
        if per_channel:
            dim = 0 if tensor.dim() > 1 else None
            abs_max = torch.max(torch.abs(tensor), dim=dim, keepdim=True).values
            
            abs_max = torch.clamp(abs_max, min=1e-6)
        else:
            abs_max = torch.max(torch.abs(tensor))
            
            if abs_max == 0:
                abs_max = 1e-6
                
        scale = abs_max / 127.0  # Use 127 as max to avoid overflow
        q_tensor = torch.clamp(torch.round(tensor / scale) + 128, 0, 255).to(torch.uint8)
        zero_point = 0  # In symmetric, conceptually zero_point is 0
        
        return q_tensor, scale, zero_point
    else:
        if per_channel:
            dim = 0 if tensor.dim() > 1 else None
            min_val = tensor.min(dim=dim, keepdim=True).values
            max_val = tensor.max(dim=dim, keepdim=True).values
            
            mask = (max_val == min_val)
            max_val = torch.where(mask, min_val + 1e-6, max_val)
        else:
            min_val = tensor.min()
            max_val = tensor.max()
            
            if max_val == min_val:
                max_val = min_val + 1e-6
        
        scale = (max_val - min_val) / 255
        zero_point = min_val
        q_tensor = torch.clamp(torch.round((tensor - min_val) / scale), 0, 255).to(torch.uint8)
        return q_tensor, scale, zero_point

def quantize_4bit_nf4(tensor):
    """
    Normalized float 4-bit quantization using predefined levels.
    """
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
    
    return indices.to(torch.uint8), nf4_levels, abs_max

def quantize_4bit_fp4(tensor):
    """
    Floating point 4-bit quantization.
    """
    signs = torch.sign(tensor)
    abs_values = torch.abs(tensor)

    log_values = torch.log2(abs_values + (abs_values == 0).float())
    exp_bias = 1 
    exp_values = torch.clamp(torch.round(log_values + exp_bias), 0, 3)
    mantissa_values = torch.round((abs_values / (2 ** (exp_values - exp_bias))) - 1)
    mantissa_values = torch.clamp(mantissa_values, 0, 1)

    # Convert to uint8 before bitwise operations
    exp_values = exp_values.to(torch.uint8)
    mantissa_values = mantissa_values.to(torch.uint8)
    q_tensor = ((exp_values << 1) | mantissa_values).to(torch.uint8)
    
    # Convert signs to uint8 for bitwise operation
    signs = (signs < 0).to(torch.uint8)
    q_tensor = torch.where(signs, q_tensor | 0x8, q_tensor)
    
    return q_tensor, None, exp_bias

def quantize_8bit_fp8(tensor):
    """
    Floating point 8-bit quantization.
    """
    signs = torch.sign(tensor)
    abs_values = torch.abs(tensor)

    log_values = torch.log2(abs_values + (abs_values == 0).float())

    exp_bias = 7 
    exp_values = torch.clamp(torch.round(log_values + exp_bias), 0, 15)

    mantissa_values = torch.round((abs_values / (2 ** (exp_values - exp_bias))) * 8 - 8)
    mantissa_values = torch.clamp(mantissa_values, 0, 7)

    # Convert to uint8 before bitwise operations
    exp_values = exp_values.to(torch.uint8)
    mantissa_values = mantissa_values.to(torch.uint8)
    q_tensor = ((exp_values << 3) | mantissa_values).to(torch.uint8)
    
    # Convert signs to uint8 for bitwise operation
    signs = (signs < 0).to(torch.uint8)
    q_tensor = torch.where(signs, q_tensor | 0x80, q_tensor)
    
    return q_tensor, None, exp_bias

def quantize_8bit_nf8(tensor):
    """
    Normalized float 8-bit quantization using tanh-based levels.
    """
    nf8_levels = torch.linspace(-1, 1, 256)
    nf8_levels = torch.tanh(nf8_levels * 2)

    abs_max = torch.max(torch.abs(tensor))
    normalized = tensor / abs_max
    expanded = normalized.unsqueeze(-1)
    distances = torch.abs(expanded - nf8_levels)
    indices = torch.argmin(distances, dim=-1)
    
    return indices.to(torch.uint8), nf8_levels, abs_max