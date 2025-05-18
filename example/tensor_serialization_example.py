"""
Example demonstrating the usage of tensor serialization and deserialization.
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytesandbits.functional.state import QuantizationState, QuantizationScheme, QuantizationType
from bytesandbits.functional.quantization import (
    quantize_8bit, 
    quantize_4bit, 
    dequantize_8bit, 
    dequantize_4bit
)
from bytesandbits.utils.tensor_utils import (
    save_quantized_tensor,
    load_quantized_tensor,
    save_quantized_tensor_torch,
    load_quantized_tensor_torch
)

def example_direct_serialization():
    """
    Demonstrates how to directly serialize and deserialize quantized tensors
    without using the QuantizationState class.
    """
    print("\n=== Direct Tensor Serialization Example ===\n")
    
    # Create a sample tensor (representing model weights)
    original_tensor = torch.randn(64, 128)
    print(f"Original tensor shape: {original_tensor.shape}")
    
    # Quantize to 8-bit
    q_tensor, scale, zero_point = quantize_8bit(original_tensor, quant_type="linear")
    print(f"Quantized tensor: {q_tensor.shape}, dtype: {q_tensor.dtype}")
    
    # Define parameters for serialization
    params = {
        "bits": 8,
        "scheme": QuantizationScheme.SYMMETRIC.value,
        "type": QuantizationType.LINEAR.value,
        "shape": original_tensor.shape
    }
    
    # Save using the custom binary format
    binary_file = "quantized_tensor.qtn"
    save_quantized_tensor(q_tensor, scale, zero_point, params, binary_file)
    print(f"Saved tensor to {binary_file}")
    
    # Save using PyTorch's format
    torch_file = "quantized_tensor.pt"
    save_quantized_tensor_torch(q_tensor, scale, zero_point, params, torch_file)
    print(f"Saved tensor to {torch_file}")
    
    # Load from binary format
    loaded_q_tensor, loaded_scale, loaded_zero_point, loaded_params = load_quantized_tensor(binary_file)
    print(f"Loaded tensor from {binary_file}")
    print(f"Loaded params: bits={loaded_params['bits']}, scheme={loaded_params['scheme']}")
    
    # Dequantize the loaded tensor
    deq_tensor = dequantize_8bit(
        loaded_q_tensor,
        loaded_scale,
        loaded_zero_point,
        quant_type="linear"
    )
    
    # Calculate the error
    error = torch.mean(torch.abs(original_tensor - deq_tensor)).item()
    print(f"Mean absolute error (binary format): {error:.6f}")
    
    # Load from PyTorch format
    loaded_q_tensor, loaded_scale, loaded_zero_point, loaded_params = load_quantized_tensor_torch(torch_file)
    print(f"Loaded tensor from {torch_file}")
    
    # Dequantize the loaded tensor
    deq_tensor = dequantize_8bit(
        loaded_q_tensor,
        loaded_scale,
        loaded_zero_point,
        quant_type="linear"
    )
    
    # Calculate the error
    error = torch.mean(torch.abs(original_tensor - deq_tensor)).item()
    print(f"Mean absolute error (PyTorch format): {error:.6f}")
    
    # Clean up
    for file in [binary_file, torch_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")

def example_state_serialization():
    """
    Demonstrates how to use QuantizationState to serialize and deserialize 
    quantized tensors with their quantization parameters.
    """
    print("\n=== State-Based Tensor Serialization Example ===\n")
    
    # Create a state manager
    quant_state = QuantizationState()
    
    # Create sample tensors (simulating different layers in a model)
    weights1 = torch.randn(256, 512)
    weights2 = torch.randn(512, 128)
    
    # Quantize tensors
    q_weights1, scale1, zero_point1 = quantize_8bit(weights1, quant_type="linear")
    q_weights2, scale2, zero_point2 = quantize_4bit(weights2, quant_type="linear")
    
    # Store parameters in state
    quant_state.set_tensor_params("layer1_weights", {
        "bits": 8,
        "scheme": QuantizationScheme.SYMMETRIC.value,
        "type": QuantizationType.LINEAR.value,
        "scale": scale1,
        "zero_point": zero_point1,
        "shape": weights1.shape
    })
    
    quant_state.set_tensor_params("layer2_weights", {
        "bits": 4,
        "scheme": QuantizationScheme.SYMMETRIC.value,
        "type": QuantizationType.LINEAR.value,
        "scale": scale2,
        "zero_point": zero_point2,
        "shape": weights2.shape
    })
    
    # Save quantized tensors using the state interface
    quant_state.save_quantized_tensor_with_state("layer1_weights", q_weights1, "layer1_weights.qtn")
    quant_state.save_quantized_tensor_with_state("layer2_weights", q_weights2, "layer2_weights.pt")
    
    print("Saved quantized tensors with state")
    
    # Create a new state and load tensors
    new_state = QuantizationState()
    
    # Load tensors and update the state
    loaded_q_weights1 = new_state.load_quantized_tensor_with_state("layer1_weights", "layer1_weights.qtn")
    loaded_q_weights2 = new_state.load_quantized_tensor_with_state("layer2_weights", "layer2_weights.pt")
    
    print("Loaded quantized tensors into new state")
    
    # Retrieve loaded parameters
    params1 = new_state.get_tensor_params("layer1_weights")
    params2 = new_state.get_tensor_params("layer2_weights")
    
    print(f"Layer1 parameters: {params1['bits']} bits, {params1['scheme']} scheme")
    print(f"Layer2 parameters: {params2['bits']} bits, {params2['scheme']} scheme")
    
    # Dequantize using the state interface
    deq_weights1 = new_state.dequantize_tensor("layer1_weights", loaded_q_weights1)
    deq_weights2 = new_state.dequantize_tensor("layer2_weights", loaded_q_weights2)
    
    # Calculate the error
    error1 = torch.mean(torch.abs(weights1 - deq_weights1)).item()
    error2 = torch.mean(torch.abs(weights2 - deq_weights2)).item()
    
    print(f"Mean absolute error (layer1): {error1:.6f}")
    print(f"Mean absolute error (layer2): {error2:.6f}")
    
    # Clean up
    for file in ["layer1_weights.qtn", "layer2_weights.pt"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")
    
    # Save full state with all parameters
    state_file = "quantization_state.json"
    new_state.save_state(state_file)
    print(f"Full state saved to {state_file}")
    
    # Clean up state file
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"Removed {state_file}")

if __name__ == "__main__":
    example_direct_serialization()
    example_state_serialization() 