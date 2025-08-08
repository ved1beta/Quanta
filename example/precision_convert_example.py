"""
Example demonstrating precision conversion between different bit-widths and formats.
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Quanta.functional.state import QuantizationState, QuantizationScheme, QuantizationType
from Quanta.functional.quantization import (
    quantize_8bit, 
    quantize_4bit, 
    dequantize_8bit, 
    dequantize_4bit
)
from Quanta.utils.utils import (
    convert_precision,
    convert_8bit_to_4bit,
    convert_4bit_to_8bit,
    optimize_for_target_hardware
)

def example_precision_conversion():
    """
    Demonstrates direct precision conversion between different bit-widths.
    """
    print("\n=== Precision Conversion Example ===\n")
    
    # Create a sample tensor
    original_tensor = torch.randn(64, 128)
    print(f"Original tensor shape: {original_tensor.shape}")
    
    # First quantize to 8-bit
    q_tensor_8bit, scale_8bit, zero_point_8bit = quantize_8bit(original_tensor, quant_type="linear")
    
    # Create parameter dictionary for 8-bit tensor
    params_8bit = {
        "bits": 8,
        "scheme": QuantizationScheme.SYMMETRIC.value,
        "type": QuantizationType.LINEAR.value,
        "scale": scale_8bit,
        "zero_point": zero_point_8bit,
        "shape": original_tensor.shape
    }
    
    print(f"8-bit quantized tensor size: {q_tensor_8bit.numel() * 1} bytes")
    
    # Convert 8-bit to 4-bit
    q_tensor_4bit, scale_4bit, zero_point_4bit, params_4bit = convert_8bit_to_4bit(
        q_tensor_8bit, params_8bit, target_type="linear"
    )
    
    print(f"4-bit quantized tensor size: {q_tensor_4bit.numel() * 0.5} bytes")
    print(f"Storage reduction: {q_tensor_8bit.numel() / (q_tensor_4bit.numel() * 0.5):.2f}x")
    
    # Dequantize both for comparison
    deq_tensor_8bit = dequantize_8bit(
        q_tensor_8bit, 
        scale_8bit, 
        zero_point_8bit, 
        quant_type="linear"
    )
    
    deq_tensor_4bit = dequantize_4bit(
        q_tensor_4bit, 
        scale_4bit, 
        zero_point_4bit, 
        quant_type="linear"
    )
    
    # Calculate errors
    error_8bit = torch.mean(torch.abs(original_tensor - deq_tensor_8bit)).item()
    error_4bit = torch.mean(torch.abs(original_tensor - deq_tensor_4bit)).item()
    error_conversion = torch.mean(torch.abs(deq_tensor_8bit - deq_tensor_4bit)).item()
    
    print(f"Mean absolute error (8-bit): {error_8bit:.6f}")
    print(f"Mean absolute error (4-bit): {error_4bit:.6f}")
    print(f"Conversion error (8-bit to 4-bit): {error_conversion:.6f}")
    
    # Now convert back to 8-bit
    q_tensor_8bit_new, scale_8bit_new, zero_point_8bit_new, params_8bit_new = convert_4bit_to_8bit(
        q_tensor_4bit, params_4bit
    )
    
    # Dequantize the converted tensor
    deq_tensor_8bit_new = dequantize_8bit(
        q_tensor_8bit_new, 
        scale_8bit_new, 
        zero_point_8bit_new, 
        quant_type="linear"
    )
    
    # Calculate round-trip error
    roundtrip_error = torch.mean(torch.abs(deq_tensor_8bit - deq_tensor_8bit_new)).item()
    print(f"Round-trip conversion error (8-bit → 4-bit → 8-bit): {roundtrip_error:.6f}")
    
    # Hardware-specific optimization
    print("\nHardware-specific optimization:")
    
    # For different hardware targets
    for hardware in ["cpu", "mobile", "edge"]:
        q_tensor_hw, scale_hw, zero_point_hw, params_hw = optimize_for_target_hardware(
            q_tensor_8bit, params_8bit, hardware
        )
        
        # Dequantize and calculate error
        if params_hw['bits'] == 8:
            deq_tensor_hw = dequantize_8bit(
                q_tensor_hw, 
                scale_hw, 
                zero_point_hw, 
                quant_type=params_hw['type']
            )
        else:
            deq_tensor_hw = dequantize_4bit(
                q_tensor_hw, 
                scale_hw, 
                zero_point_hw, 
                quant_type=params_hw['type']
            )
        
        hw_error = torch.mean(torch.abs(original_tensor - deq_tensor_hw)).item()
        print(f"{hardware} optimization: {params_hw['bits']}-bit {params_hw['type']}")
        print(f"  Size: {q_tensor_hw.numel() * (params_hw['bits']/8)} bytes")
        print(f"  Error: {hw_error:.6f}")

def example_state_conversion():
    """
    Demonstrates precision conversion using the QuantizationState class.
    """
    print("\n=== State-Based Precision Conversion Example ===\n")
    
    # Create a state manager
    quant_state = QuantizationState()
    
    # Create a sample tensor
    original_tensor = torch.randn(128, 256)
    print(f"Original tensor shape: {original_tensor.shape}")
    
    # Quantize to 8-bit and store in state
    q_tensor_8bit, scale_8bit, zero_point_8bit = quantize_8bit(original_tensor, quant_type="linear")
    
    # Set parameters in state
    quant_state.set_tensor_params("model_weights", {
        "bits": 8,
        "scheme": QuantizationScheme.SYMMETRIC.value,
        "type": QuantizationType.LINEAR.value,
        "scale": scale_8bit,
        "zero_point": zero_point_8bit,
        "shape": original_tensor.shape
    })
    
    # Store the tensor in state
    if not hasattr(quant_state, "_quantized_tensors"):
        quant_state._quantized_tensors = {}
    quant_state._quantized_tensors["model_weights"] = q_tensor_8bit
    
    print(f"8-bit quantized tensor size: {q_tensor_8bit.numel() * 1} bytes")
    
    # Convert to 4-bit using the state
    q_tensor_4bit = quant_state.convert_tensor_precision(
        "model_weights", 
        target_bits=4, 
        target_type="linear"
    )
    
    print(f"4-bit quantized tensor size: {q_tensor_4bit.numel() * 0.5} bytes")
    
    # Get updated parameters
    params_4bit = quant_state.get_tensor_params("model_weights")
    print(f"Updated parameters: {params_4bit['bits']}-bit {params_4bit['type']}")
    
    # Dequantize and calculate error
    deq_tensor_4bit = quant_state.dequantize_tensor("model_weights", q_tensor_4bit)
    error_4bit = torch.mean(torch.abs(original_tensor - deq_tensor_4bit)).item()
    print(f"Mean absolute error (4-bit): {error_4bit:.6f}")
    
    # Try different quantization types
    for quant_type in ["linear", "nf4"]:
        # Convert to the specified type
        q_tensor_converted = quant_state.convert_tensor_precision(
            "model_weights", 
            target_bits=4, 
            target_type=quant_type
        )
        
        # Dequantize and calculate error
        deq_tensor = quant_state.dequantize_tensor("model_weights", q_tensor_converted)
        error = torch.mean(torch.abs(original_tensor - deq_tensor)).item()
        
        print(f"\n4-bit {quant_type} conversion:")
        print(f"  Error: {error:.6f}")
        
        # Convert back to 8-bit
        q_tensor_8bit_new = quant_state.convert_tensor_precision(
            "model_weights", 
            target_bits=8, 
            target_type="linear"
        )
        
        # Dequantize and calculate round-trip error
        deq_tensor_8bit_new = quant_state.dequantize_tensor("model_weights", q_tensor_8bit_new)
        roundtrip_error = torch.mean(torch.abs(original_tensor - deq_tensor_8bit_new)).item()
        print(f"  Round-trip error (8-bit → 4-bit {quant_type} → 8-bit): {roundtrip_error:.6f}")

if __name__ == "__main__":
    example_precision_conversion()
    example_state_conversion() 