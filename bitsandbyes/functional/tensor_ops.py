import torch 
from torch.autograd import Function
from .base import BaseQuantizer
from typing import Tuple

class Quantizer(BaseQuantizer):
    """Quantization operations using the base quantizer."""
    
    def quantize_8bit(
        self,
        tensor: torch.Tensor,
        per_channel: bool = False,
        symmetric: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """8-bit quantization."""
        self.num_bits = 8
        self.symmetric = symmetric
        return self.quantize(tensor, per_channel)
    
    def quantize_4bit(
        self,
        tensor: torch.Tensor,
        per_channel: bool = False,
        symmetric: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """4-bit quantization."""
        self.num_bits = 4
        self.symmetric = symmetric
        return self.quantize(tensor, per_channel)

_quantizer = Quantizer()

def quantize_8bit(tensor, per_channel=False, symmetric=True):
    return _quantizer.quantize_8bit(tensor, per_channel, symmetric)

def quantize_4bit(tensor, per_channel=False, symmetric=True):
    return _quantizer.quantize_4bit(tensor, per_channel, symmetric)

def dequantize_8bit(q_tensor, scale, zero_point):
    return _quantizer.dequantize(q_tensor, scale, zero_point)

def dequantize_4bit(q_tensor, scale, zero_point):
    return _quantizer.dequantize(q_tensor, scale, zero_point)

def quantize(tensor , num_bits = 8 , symmetric = True):
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        scale = (2**(num_bits-1) - 1) / abs_max
        zero_point = 0
    else:
        scale = (2**num_bits - 1) / (max_val - min_val)
        zero_point = -round(min_val * scale)

    quantize = torch.clamp(
        torch.round(tensor * scale + zero_point) , 0 , 2**num_bits - 1).to(torch.int8 if num_bits <= 8 else torch.int16)
    return quantize , scale , zero_point

def dequantize(tensor , scale , zero_point):
    return (tensor.float() - zero_point)/scale

def quantize_add(a, a_scale, a_zero_point, b, b_scale, b_zero_point):
    a_float = dequantize(a , a_scale , a_zero_point)
    b_float = dequantize(b , b_scale , b_zero_point)

    result_float = a_float + b_float 

    return quantize(result_float)

def quantized_matmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point):

    # Dequantize inputs to floating point
    a_float = dequantize(a, a_scale, a_zero_point)
    b_float = dequantize(b, b_scale, b_zero_point)
    
    # Perform matrix multiplication in floating point
    result_float = torch.matmul(a_float, b_float)
    
    # Requantize the result
    return quantize(result_float)

def quantized_mul(a, a_scale, a_zero_point, b, b_scale, b_zero_point):

    # Dequantize inputs to floating point
    a_float = dequantize(a, a_scale, a_zero_point)
    b_float = dequantize(b, b_scale, b_zero_point)
    
    # Perform element-wise multiplication in floating point
    result_float = a_float * b_float
    
    # Requantize the result
    return quantize(result_float)

def quantized_relu(x, scale, zero_point):

    # Dequantize input to floating point
    x_float = dequantize(x, scale, zero_point)
    
    # Apply ReLU in floating point
    result_float = torch.relu(x_float)
    
    # Requantize the result
    return quantize(result_float)