import torch 
from torch.autograd import Function

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