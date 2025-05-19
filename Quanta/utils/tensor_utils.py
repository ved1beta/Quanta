import torch
import numpy as np
import json
import os

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

def save_quantized_tensor(q_tensor, scale, zero_point, params, file_path):
    """Save a quantized tensor to a file."""
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
    """Load a quantized tensor from a file."""
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
    """Save a quantized tensor using PyTorch's native serialization."""
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
    """Load a quantized tensor saved with PyTorch's native serialization."""
    if not file_path.endswith('.pt'):
        file_path = file_path + '.pt'
    
    data_dict = torch.load(file_path)
    
    return (
        data_dict['q_tensor'], 
        data_dict['scale'], 
        data_dict['zero_point'], 
        data_dict['params']
    )