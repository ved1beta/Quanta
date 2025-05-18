"""
Utility functions for BytesAndBits.

This module provides various utility functions for tensor operations, serialization,
and precision conversion.
"""

import torch
import numpy as np
import json
import os
from ..functional.quantization import (
    quantize_8bit, 
    quantize_4bit, 
    dequantize_8bit, 
    dequantize_4bit
)

#
# Tensor packing/unpacking
#

def pack_4bit_tensor(tensor):
    """Pack 4-bit values into a tensor with half the size."""
    if tensor.dtype != torch.uint8:
        raise ValueError("Input tensor must be uint8")
    
    origin_shape = tensor.shape
    tensor = tensor.reshape(-1)
    
    if tensor.numel() % 2 == 1:
        tensor = torch.cat([tensor, torch.zeros(1, dtype=torch.uint8)])
    tensor = tensor.reshape(-1, 2)
    packed = (tensor[:, 0] | (tensor[:, 1] << 4))
    return packed, origin_shape

def unpack_4bit_tensor(packed_tensor):
    """Unpack a tensor where each byte contains two 4-bit values."""
    packed_flat = packed_tensor.reshape(-1)
    
    low_bits = packed_flat & 0x0F
    high_bits = (packed_flat >> 4) & 0x0F
    
    unpacked = torch.empty(packed_flat.numel() * 2, dtype=torch.uint8)
    unpacked[0::2] = low_bits
    unpacked[1::2] = high_bits
    
    return unpacked

def tensor_bits_to_bytes(tensor, bits):
    """Convert tensor size from bits to bytes."""
    num_elements = torch.tensor(tensor.shape).prod().item()
    total_bits = num_elements * bits
    return (total_bits + 7) // 8 

#
# Serialization functions
#

def save_quantized_tensor(q_tensor, scale, zero_point, params, file_path):
    """
    Save a quantized tensor to a file.
    
    Args:
        q_tensor: The quantized tensor (uint8)
        scale: The scale factor used in quantization
        zero_point: The zero point used in quantization
        params: Dictionary with metadata like bits, scheme, shape
        file_path: Path where to save the file
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    if not file_path.endswith('.qtn'):
        file_path = file_path + '.qtn'
    
    q_tensor_np = q_tensor.cpu().numpy()
    
    if isinstance(scale, torch.Tensor):
        scale_np = scale.cpu().numpy()
    else:
        scale_np = scale
        
    if isinstance(zero_point, torch.Tensor):
        zero_point_np = zero_point.cpu().numpy()
    else:
        zero_point_np = zero_point

    dtype_str = str(q_tensor.dtype)
    if dtype_str.startswith('torch.'):
        dtype_str = dtype_str[6:]
    
    metadata = {
        'bits': params.get('bits', 8),
        'scheme': params.get('scheme', 'symmetric'),
        'type': params.get('type', 'linear'),
        'shape': list(q_tensor.shape),
        'dtype': dtype_str
    }
    
    with open(file_path, 'wb') as f:
        metadata_str = json.dumps(metadata)
        header_len = len(metadata_str)
        f.write(header_len.to_bytes(8, byteorder='little'))
        f.write(metadata_str.encode('utf-8'))
        
        f.write(q_tensor_np.tobytes())
        f.write(scale_np.tobytes())
        f.write(zero_point_np.tobytes())

def load_quantized_tensor(file_path):
    """
    Load a quantized tensor from a file.
    
    Args:
        file_path: Path to the saved quantized tensor
        
    Returns:
        Tuple of (q_tensor, scale, zero_point, params)
    """
    if not file_path.endswith('.qtn'):
        file_path = file_path + '.qtn'
    
    with open(file_path, 'rb') as f:
        # Read metadata
        header_len = int.from_bytes(f.read(8), byteorder='little')
        metadata_str = f.read(header_len).decode('utf-8')
        metadata = json.loads(metadata_str)
        
        # Get tensor info from metadata
        shape = tuple(metadata['shape'])
        bits = metadata['bits']
        dtype_name = metadata['dtype']
        
        # Create a numpy-compatible dtype
        if dtype_name == 'uint8':
            numpy_dtype = np.uint8
        elif dtype_name == 'int8':
            numpy_dtype = np.int8
        elif dtype_name == 'float32':
            numpy_dtype = np.float32
        else:
            numpy_dtype = np.uint8  
        
        q_tensor_size = np.prod(shape) * (np.dtype(np.uint8).itemsize)
        q_tensor_bytes = f.read(int(q_tensor_size))
        
        q_tensor_np = np.frombuffer(q_tensor_bytes, dtype=np.uint8).copy()
        q_tensor_np = q_tensor_np.reshape(shape)
        
        q_tensor = torch.from_numpy(q_tensor_np)
        
        if bits == 8:
            scale_dtype = np.float32
            zero_point_dtype = np.float32 if metadata['scheme'] == 'asymmetric' else np.float32
        else:
            scale_dtype = np.float32
            zero_point_dtype = np.float32
            
        scale_np = np.frombuffer(f.read(np.dtype(scale_dtype).itemsize), dtype=scale_dtype).copy()
        zero_point_np = np.frombuffer(f.read(np.dtype(zero_point_dtype).itemsize), dtype=zero_point_dtype).copy()
        
        scale = torch.tensor(scale_np.item())
        zero_point = torch.tensor(zero_point_np.item())
        
        return q_tensor, scale, zero_point, metadata
            
def save_quantized_tensor_torch(q_tensor, scale, zero_point, params, file_path):
    """
    Save a quantized tensor using PyTorch's native serialization.
    
    Args:
        q_tensor: The quantized tensor
        scale: The scale factor used in quantization
        zero_point: The zero point used in quantization
        params: Dictionary with metadata like bits, scheme, shape
        file_path: Path where to save the file
    """
    if not file_path.endswith('.pt'):
        file_path = file_path + '.pt'
    
    data_dict = {
        'q_tensor': q_tensor,
        'scale': scale, 
        'zero_point': zero_point,
        'params': params
    }
    
    torch.save(data_dict, file_path)

def load_quantized_tensor_torch(file_path):
    """
    Load a quantized tensor saved with PyTorch's native serialization.
    
    Args:
        file_path: Path to the saved file
        
    Returns:
        Tuple of (q_tensor, scale, zero_point, params)
    """
    if not file_path.endswith('.pt'):
        file_path = file_path + '.pt'
    
    data_dict = torch.load(file_path)
    
    return (
        data_dict['q_tensor'], 
        data_dict['scale'], 
        data_dict['zero_point'], 
        data_dict['params']
    )

#
# Precision Conversion Functions
#

def convert_precision(q_tensor, source_params, target_bits, target_type="linear", 
                     target_scheme=None):
    """
    Convert a quantized tensor from one precision format to another.
    
    Args:
        q_tensor: The quantized tensor to convert
        source_params: Dictionary with source quantization parameters
        target_bits: Target bit depth (4 or 8)
        target_type: Target quantization type (linear, nf4, fp4, etc.)
        target_scheme: Target quantization scheme (if None, use source scheme)
        
    Returns:
        Tuple of (converted quantized tensor, new scale, new zero_point, new params)
    """
    # Get source parameters
    source_bits = source_params.get('bits', 8)
    source_type = source_params.get('type', 'linear')
    source_scale = source_params.get('scale')
    source_zero_point = source_params.get('zero_point')
    source_scheme = source_params.get('scheme', 'symmetric')
    
    # If target scheme not specified, keep the same
    if target_scheme is None:
        target_scheme = source_scheme
    
    # First dequantize to full precision
    if source_bits == 8:
        fp_tensor = dequantize_8bit(
            q_tensor,
            source_scale,
            source_zero_point,
            quant_type=source_type
        )
    elif source_bits == 4:
        fp_tensor = dequantize_4bit(
            q_tensor,
            source_scale,
            source_zero_point,
            quant_type=source_type
        )
    else:
        raise ValueError(f"Unsupported source bit depth: {source_bits}")
    
    # Then requantize to target precision
    if target_bits == 8:
        new_q_tensor, new_scale, new_zero_point = quantize_8bit(
            fp_tensor,
            quant_type=target_type
        )
    elif target_bits == 4:
        new_q_tensor, new_scale, new_zero_point = quantize_4bit(
            fp_tensor,
            quant_type=target_type
        )
    else:
        raise ValueError(f"Unsupported target bit depth: {target_bits}")
    
    # Create new parameters dict
    new_params = {
        'bits': target_bits,
        'type': target_type,
        'scheme': target_scheme,
        'scale': new_scale,
        'zero_point': new_zero_point,
        'shape': tuple(new_q_tensor.shape)
    }
    
    return new_q_tensor, new_scale, new_zero_point, new_params

def convert_8bit_to_4bit(q_tensor, source_params, target_type="linear"):
    """
    Convert 8-bit quantized tensor to 4-bit.
    
    Args:
        q_tensor: The 8-bit quantized tensor
        source_params: Source quantization parameters
        target_type: Target quantization type (linear, nf4, fp4)
        
    Returns:
        Tuple of (4-bit tensor, scale, zero_point, params)
    """
    return convert_precision(q_tensor, source_params, 4, target_type)

def convert_4bit_to_8bit(q_tensor, source_params, target_type="linear"):
    """
    Convert 4-bit quantized tensor to 8-bit.
    
    Args:
        q_tensor: The 4-bit quantized tensor
        source_params: Source quantization parameters
        target_type: Target quantization type (linear, nf8, fp8)
        
    Returns:
        Tuple of (8-bit tensor, scale, zero_point, params)
    """
    return convert_precision(q_tensor, source_params, 8, target_type)

def optimize_for_target_hardware(q_tensor, source_params, target_hardware):
    """
    Convert tensor to the optimal precision for specific hardware.
    
    Args:
        q_tensor: The quantized tensor
        source_params: Source quantization parameters 
        target_hardware: String identifying target hardware ("cpu", "gpu", "mobile", etc.)
        
    Returns:
        Optimized tensor with parameters
    """
    # Define hardware-specific configuration
    hw_config = {
        "cpu": {"bits": 8, "type": "linear"},
        "gpu": {"bits": 8, "type": "linear"},
        "mobile": {"bits": 4, "type": "nf4"},
        "edge": {"bits": 4, "type": "linear"},
    }
    
    # Get configuration or use default
    config = hw_config.get(target_hardware, {"bits": 8, "type": "linear"})
    
    # Convert to target precision
    return convert_precision(
        q_tensor, 
        source_params, 
        config["bits"], 
        config["type"]
    ) 