"""
Example demonstrating the usage of QuantizationState for managing quantization parameters.
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

def example_state_management():
    """
    Demonstrates how to use QuantizationState to store and retrieve 
    quantization parameters for different tensors and layers.
    """
    print("\n=== QuantizationState Example ===\n")
    
    # Create a QuantizationState instance
    quant_state = QuantizationState()
    
    # 1. Managing tensor parameters
    print("1. Managing tensor parameters:")
    
    # Create sample tensors
    tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    tensor2 = torch.tensor([[-10.0, -5.0, 0.0], [5.0, 10.0, 15.0]], dtype=torch.float32)
    
    # Quantize tensors
    q_tensor1, scale1, zero_point1 = quantize_8bit(tensor1, symmetric=True)
    q_tensor2, scale2, zero_point2 = quantize_8bit(tensor2, symmetric=False)
    
    # Store the quantization parameters in state
    quant_state.set_tensor_params("tensor1", {
        "bits": 8,
        "scheme": QuantizationScheme.SYMMETRIC.value,
        "type": QuantizationType.LINEAR.value,
        "scale": scale1,
        "zero_point": zero_point1,
        "shape": tensor1.shape
    })
    
    quant_state.set_tensor_params("tensor2", {
        "bits": 8,
        "scheme": QuantizationScheme.ASYMMETRIC.value,
        "type": QuantizationType.LINEAR.value,
        "scale": scale2,
        "zero_point": zero_point2,
        "shape": tensor2.shape
    })
    
    # Retrieve and use the stored parameters
    params1 = quant_state.get_tensor_params("tensor1")
    params2 = quant_state.get_tensor_params("tensor2")
    
    print(f"Tensor1 parameters: {params1['bits']} bits, {params1['scheme']} scheme")
    print(f"Tensor2 parameters: {params2['bits']} bits, {params2['scheme']} scheme")
    
    # 2. Managing layer parameters
    print("\n2. Managing layer parameters:")
    
    # Store layer-specific quantization settings
    quant_state.set_layer_params("conv1", {
        "weight_bits": 4,
        "activation_bits": 8,
        "weight_scheme": QuantizationScheme.SYMMETRIC.value,
        "activation_scheme": QuantizationScheme.ASYMMETRIC.value,
        "weight_type": QuantizationType.NF4.value,
        "activation_type": QuantizationType.LINEAR.value
    })
    
    quant_state.set_layer_params("fc1", {
        "weight_bits": 8,
        "activation_bits": 8,
        "weight_scheme": QuantizationScheme.SYMMETRIC.value,
        "activation_scheme": QuantizationScheme.SYMMETRIC.value,
        "weight_type": QuantizationType.LINEAR.value,
        "activation_type": QuantizationType.LINEAR.value
    })
    
    # Retrieve layer parameters
    conv_params = quant_state.get_layer_params("conv1")
    fc_params = quant_state.get_layer_params("fc1")
    
    print(f"Conv1 parameters: weights={conv_params['weight_bits']}b {conv_params['weight_type']}, "
          f"activations={conv_params['activation_bits']}b {conv_params['activation_type']}")
    print(f"FC1 parameters: weights={fc_params['weight_bits']}b {fc_params['weight_type']}, "
          f"activations={fc_params['activation_bits']}b {fc_params['activation_type']}")
    
    # 3. Updating global configuration
    print("\n3. Updating global configuration:")
    
    print(f"Default config: {quant_state.global_config}")
    
    # Update global defaults
    quant_state.update_global_config({
        "default_bits": 4,
        "default_type": QuantizationType.NF4.value
    })
    
    print(f"Updated config: {quant_state.global_config}")
    
    # 4. Saving and loading state
    print("\n4. Saving and loading state:")
    
    # Save state to file
    state_file = "quantization_state.json"
    quant_state.save_state(state_file)
    print(f"State saved to {state_file}")
    
    # Create a new state and load from file
    new_state = QuantizationState()
    new_state.load_state(state_file)
    
    # Verify the loaded state
    print(f"Loaded global config: {new_state.global_config}")
    
    # Use the loaded parameters to dequantize a tensor
    loaded_params1 = new_state.get_tensor_params("tensor1")
    
    # Print the loaded parameters for debugging
    print(f"Loaded scale: {loaded_params1['scale']}")
    print(f"Loaded zero_point: {loaded_params1['zero_point']}")
    
    # Make sure we're using the original quantized tensor
    # Important: When we saved the state, we saved the parameters, not the quantized tensor itself
    # So we need to use the original q_tensor1 with the loaded parameters
    
    # Dequantize the first tensor using the loaded parameters
    deq_tensor1 = dequantize_8bit(
        q_tensor1, 
        loaded_params1["scale"], 
        loaded_params1["zero_point"],
        symmetric=(loaded_params1["scheme"] == QuantizationScheme.SYMMETRIC.value)
    )
    
    print(f"Original tensor1:\n{tensor1}")
    print(f"Dequantized tensor1 (using loaded params):\n{deq_tensor1}")
    print(f"Mean absolute error: {torch.mean(torch.abs(tensor1 - deq_tensor1)).item()}")
    
    # Clean up
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"Removed {state_file}")

def example_per_layer_quantization():
    """
    Demonstrates how to use QuantizationState for per-layer quantization settings.
    """
    print("\n=== Per-Layer Quantization Example ===\n")
    
    # Create a state manager
    quant_state = QuantizationState()
    
    # Define per-layer quantization settings
    layers = ["conv1", "conv2", "fc1", "fc2"]
    
    # Define different settings for different layers
    quant_config = {
        "conv1": {"bits": 8, "scheme": QuantizationScheme.SYMMETRIC.value, "type": QuantizationType.LINEAR.value},
        "conv2": {"bits": 4, "scheme": QuantizationScheme.SYMMETRIC.value, "type": QuantizationType.NF4.value},
        "fc1": {"bits": 8, "scheme": QuantizationScheme.ASYMMETRIC.value, "type": QuantizationType.LINEAR.value},
        "fc2": {"bits": 4, "scheme": QuantizationScheme.SYMMETRIC.value, "type": QuantizationType.FP4.value}
    }
    
    # Store configuration in state
    for layer_name, config in quant_config.items():
        quant_state.set_layer_params(layer_name, config)
    
    # Simulate model quantization by creating sample weights for each layer
    sample_weights = {
        "conv1": torch.randn(32, 3, 3, 3),  # 32 output channels, 3 input channels, 3x3 kernel
        "conv2": torch.randn(64, 32, 3, 3), # 64 output channels, 32 input channels, 3x3 kernel
        "fc1": torch.randn(256, 1024),      # 256 output features, 1024 input features
        "fc2": torch.randn(10, 256)         # 10 output features, 256 input features
    }
    
    # Quantize each layer according to its configuration
    quantized_weights = {}
    quantization_params = {}
    
    for layer_name, weights in sample_weights.items():
        config = quant_state.get_layer_params(layer_name)
        
        # Select quantization function based on bits
        if config["bits"] == 8:
            quantize_fn = quantize_8bit
            dequantize_fn = dequantize_8bit
        else:  # 4-bit
            quantize_fn = quantize_4bit
            dequantize_fn = dequantize_4bit
        
        # Quantize the weights
        q_weights, scale, zero_point = quantize_fn(
            weights, 
            symmetric=(config["scheme"] == QuantizationScheme.SYMMETRIC.value)
        )
        
        # Store quantized weights and parameters
        quantized_weights[layer_name] = q_weights
        quantization_params[layer_name] = {
            "scale": scale,
            "zero_point": zero_point,
            "dequantize_fn": dequantize_fn,
            "config": config
        }
        
        # Also store parameters in state
        quant_state.set_tensor_params(f"{layer_name}_weights", {
            "bits": config["bits"],
            "scheme": config["scheme"],
            "type": config["type"],
            "scale": scale,
            "zero_point": zero_point
        })
    
    # Print quantization info for each layer
    print("Layer quantization summary:")
    for layer_name in layers:
        params = quant_state.get_layer_params(layer_name)
        tensor_params = quant_state.get_tensor_params(f"{layer_name}_weights")
        
        original_size = sample_weights[layer_name].nelement() * 32  # 32 bits per float32
        quantized_size = sample_weights[layer_name].nelement() * params["bits"]
        compression_ratio = original_size / quantized_size
        
        print(f"\n{layer_name}:")
        print(f"  Config: {params['bits']} bits, {params['scheme']} scheme, {params['type']} type")
        print(f"  Shape: {sample_weights[layer_name].shape}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        # Estimate quantization error
        q_params = quantization_params[layer_name]
        deq_weights = q_params["dequantize_fn"](
            quantized_weights[layer_name],
            q_params["scale"],
            q_params["zero_point"],
            symmetric=(q_params["config"]["scheme"] == QuantizationScheme.SYMMETRIC.value)
        )
        
        error = torch.mean(torch.abs(sample_weights[layer_name] - deq_weights)).item()
        print(f"  Mean absolute error: {error:.6f}")
    
    # Save the state to a file
    state_file = "model_quantization_state.json"
    quant_state.save_state(state_file)
    print(f"\nQuantization state saved to {state_file}")
    
    # Illustrate loading the state in a different context
    print("\nLoading state in a different context:")
    loaded_state = QuantizationState()
    loaded_state.load_state(state_file)
    
    # Verify some of the loaded parameters
    print("\nVerifying loaded parameters for fc2 layer:")
    loaded_fc2_params = loaded_state.get_tensor_params("fc2_weights")
    print(f"Bits: {loaded_fc2_params['bits']}")
    print(f"Scheme: {loaded_fc2_params['scheme']}")
    print(f"Type: {loaded_fc2_params['type']}")
    
    # Clean up
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"Removed {state_file}")

if __name__ == "__main__":
    example_state_management()
    example_per_layer_quantization() 