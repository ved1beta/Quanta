import torch
import numpy as np

def pack_4bit_tensor(tensor):
    """Pack 4-bit values into a tensor with half the size."""
    if tensor.dtype != torch.uint8:
        raise ValueError("Input tensor must be uint8")
    
    origin_shape = tensor.shape()
    tensor = tensor.reshape(-1)
    
    if tensor.numel() % 2 == 1:
        tensor = torch.cat([tensor, torch.zeros(1, dtype=torch.uint8)])
    tensor = tensor.reshape(-1, 2)
    packed = (tensor[:, 0] | (tensor[:, 1] << 4))
    return packed , origin_shape

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