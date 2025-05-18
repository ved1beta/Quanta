import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytesandbits.utils.tensor_utils import pack_4bit_tensor, unpack_4bit_tensor, tensor_bits_to_bytes

def example_4bit_packing():
    print("=== 4-bit Tensor Packing Example ===")
    
    # Create a tensor with 4-bit values (0-15)
    values = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)
    print(f"Original values: {values}")
    
    # Pack the tensor
    packed = pack_4bit_tensor(values)
    print(f"Packed values: {packed}")
    print(f"Packed tensor shape: {packed.shape} (half the original size)")
    
    # Unpack the tensor
    unpacked = unpack_4bit_tensor(packed)
    print(f"Unpacked values: {unpacked}")
    
    # Verify the values match
    match = torch.all(values == unpacked[:len(values)])
    print(f"Values match: {match}")
    
    # Calculate storage savings
    original_size = len(values)
    packed_size = len(packed)
    savings_pct = (1 - packed_size / original_size) * 100
    print(f"Storage savings: {savings_pct:.2f}%")

def example_quantization_storage():
    print("=== Quantization Storage Example ===")
    
    # Create a sample model weight tensor
    weight = torch.randn(256, 1024)  # 256x1024 weight matrix
    
    # Calculate storage sizes for different bit precisions
    fp32_size = weight.nelement() * 4  # 4 bytes per element
    int8_size = weight.nelement() * 1  # 1 byte per element
    int4_size = tensor_bits_to_bytes(weight, 4)  # 4 bits per element
    
    print(f"Weight tensor shape: {weight.shape} ({weight.nelement()} elements)")
    print(f"FP32 storage size: {fp32_size:,} bytes")
    print(f"INT8 storage size: {int8_size:,} bytes ({fp32_size/int8_size:.2f}x reduction)")
    print(f"INT4 storage size: {int4_size:,} bytes ({fp32_size/int4_size:.2f}x reduction)")

def main():
    example_4bit_packing()
    print()
    example_quantization_storage()

if __name__ == "__main__":
    main()
