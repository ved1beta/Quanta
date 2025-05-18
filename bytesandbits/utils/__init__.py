"""
Utility functions for BytesAndBits.
"""

from .utils import (
    # Tensor packing/unpacking
    pack_4bit_tensor,
    unpack_4bit_tensor,
    tensor_bits_to_bytes,
    
    # Serialization
    save_quantized_tensor,
    load_quantized_tensor,
    save_quantized_tensor_torch,
    load_quantized_tensor_torch,
    
    # Precision conversion
    convert_precision,
    convert_8bit_to_4bit,
    convert_4bit_to_8bit,
    optimize_for_target_hardware
)

__all__ = [
    # Tensor packing/unpacking
    "pack_4bit_tensor",
    "unpack_4bit_tensor",
    "tensor_bits_to_bytes",
    
    # Serialization
    "save_quantized_tensor",
    "load_quantized_tensor",
    "save_quantized_tensor_torch",
    "load_quantized_tensor_torch",
    
    # Precision conversion
    "convert_precision",
    "convert_8bit_to_4bit",
    "convert_4bit_to_8bit", 
    "optimize_for_target_hardware"
]
