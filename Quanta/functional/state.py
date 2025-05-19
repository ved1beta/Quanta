import torch
from enum import Enum
from typing import Dict, Optional, Tuple, Union
import json
import os

class QuantizationScheme(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"

class QuantizationType(Enum):
    LINEAR = "linear"
    NF4 = "nf4"
    NF8 = "nf8"
    FP4 = "fp4"
    FP8 = "fp8"

class QuantizationState:
    def __init__(self):
        self.tensor_params  = {}
        self.layers_params = {}
        self.global_config  = {
            "default_bits": 8,
            "default_scheme": QuantizationScheme.SYMMETRIC.value,
            "default_type": QuantizationType.LINEAR.value
        }

    def set_tensor_params(self, tensor_id: str, params: Dict):
        """
        Set quantization parameters for a specific tensor.
        
        Args:
            tensor_id: Unique identifier for the tensor
            params: Dictionary of quantization parameters
        """
        self.tensor_params[tensor_id] = params

    def get_tensor_params(self, tensor_id: str) -> Optional[Dict]:
        """
        Get quantization parameters for a specific tensor.
        
        Args:
            tensor_id: Unique identifier for the tensor
            
        Returns:
            Dictionary of parameters or None if not found
        """
        return self.tensor_params.get(tensor_id)

    def set_layer_params(self, layer_name: str, params: Dict):
        """
        Set quantization parameters for a specific layer.
        
        Args:
            layer_name: Name of the layer
            params: Dictionary of quantization parameters
        """
        self.layers_params[layer_name] = params

    def get_layer_params(self, layer_name: str) -> Optional[Dict]:
        """
        Get quantization parameters for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dictionary of parameters or None if not found
        """
        return self.layers_params.get(layer_name)

    def update_global_config(self, config_updates: Dict):
        """
        Update the global configuration with new values.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.global_config.update(config_updates)

    def save_state(self, filepath: str):
        """
        Save the current state to a JSON file.
        
        Args:
            filepath: Path to save the state
        """
        # Need to convert tensor values to serializable types
        serializable_tensor_params = {}
        for tensor_id, params in self.tensor_params.items():
            serializable_params = {}
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    serializable_params[key] = value.cpu().tolist()
                else:
                    serializable_params[key] = value
            serializable_tensor_params[tensor_id] = serializable_params

        state_dict = {
            "tensor_params": serializable_tensor_params,
            "layers_params": self.layers_params,
            "global_config": self.global_config
        }

        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def load_state(self, filepath: str):
        """
        Load state from a JSON file.
        
        Args:
            filepath: Path to the state file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found: {filepath}")

        with open(filepath, 'r') as f:
            state_dict = json.load(f)

        self.global_config = state_dict.get("global_config", self.global_config)
        self.layers_params = state_dict.get("layers_params", {})

        # Convert lists back to tensors for tensor parameters
        tensor_params = state_dict.get("tensor_params", {})
        for tensor_id, params in tensor_params.items():
            for key, value in params.items():
                if isinstance(value, list):
                    params[key] = torch.tensor(value)
            self.tensor_params[tensor_id] = params
            
    def save_quantized_tensor_with_state(self, tensor_name: str, q_tensor: torch.Tensor, file_path: str):
        """
        Save a quantized tensor along with its state information.
        
        This method saves both the tensor data and its quantization parameters to a single file.
        The state is used to retrieve the necessary quantization parameters.
        
        Args:
            tensor_name: Name of the tensor in the state
            q_tensor: The quantized tensor to save
            file_path: Path where to save the file
        """
        from ..utils.utils import save_quantized_tensor, save_quantized_tensor_torch
        
        params = self.get_tensor_params(tensor_name)
        if params is None:
            raise ValueError(f"No parameters found for tensor '{tensor_name}' in state")
        
        scale = params.get('scale')
        zero_point = params.get('zero_point')
        
        if scale is None or zero_point is None:
            raise ValueError(f"Missing scale or zero_point for tensor '{tensor_name}'")
        
        # Save the tensor with its parameters using the appropriate format
        if file_path.endswith('.pt'):
            save_quantized_tensor_torch(q_tensor, scale, zero_point, params, file_path)
        else:
            save_quantized_tensor(q_tensor, scale, zero_point, params, file_path)
            
    def load_quantized_tensor_with_state(self, tensor_name: str, file_path: str) -> torch.Tensor:
        """
        Load a quantized tensor and update its state information.
        
        This method loads both the tensor data and its quantization parameters from a file,
        then updates the state with the loaded parameters.
        
        Args:
            tensor_name: Name to use for the tensor in the state
            file_path: Path to the saved quantized tensor file
            
        Returns:
            The loaded quantized tensor
        """
        from ..utils.utils import load_quantized_tensor, load_quantized_tensor_torch
        
        if file_path.endswith('.pt'):
            q_tensor, scale, zero_point, params = load_quantized_tensor_torch(file_path)
        else:
            q_tensor, scale, zero_point, params = load_quantized_tensor(file_path)
        
        # Update the parameters with scale and zero_point if they aren't in params
        if 'scale' not in params:
            params['scale'] = scale
        if 'zero_point' not in params:
            params['zero_point'] = zero_point
        
        # Update state with loaded parameters
        self.set_tensor_params(tensor_name, params)
        
        # Store the tensor in the state for easy access later
        if not hasattr(self, "_quantized_tensors"):
            self._quantized_tensors = {}
        self._quantized_tensors[tensor_name] = q_tensor
        
        return q_tensor
        
    def convert_tensor_precision(self, tensor_name: str, target_bits: int, target_type: str = "linear", 
                                target_scheme: str = None) -> torch.Tensor:
        """
        Convert a tensor to a different precision format and update its state.
        
        Args:
            tensor_name: Name of the tensor in the state
            target_bits: Target bit depth (4 or 8)
            target_type: Target quantization type
            target_scheme: Target quantization scheme (if None, use source scheme)
            
        Returns:
            The converted quantized tensor
        """
        from ..utils.utils import convert_precision
        
        # Get original tensor and parameters
        source_params = self.get_tensor_params(tensor_name)
        if source_params is None:
            raise ValueError(f"No parameters found for tensor '{tensor_name}' in state")
        
        # Get the quantized tensor
        if hasattr(self, "_quantized_tensors") and tensor_name in self._quantized_tensors:
            q_tensor = self._quantized_tensors[tensor_name]
        else:
            raise ValueError(f"Quantized tensor '{tensor_name}' not found in state. "
                           f"Please load the tensor first using load_quantized_tensor_with_state.")
        
        # Convert to target precision
        new_q_tensor, new_scale, new_zero_point, new_params = convert_precision(
            q_tensor, 
            source_params, 
            target_bits, 
            target_type,
            target_scheme
        )
        
        # Update the state with new parameters
        self.set_tensor_params(tensor_name, new_params)
        
        # Store the new quantized tensor
        if not hasattr(self, "_quantized_tensors"):
            self._quantized_tensors = {}
        self._quantized_tensors[tensor_name] = new_q_tensor
        
        return new_q_tensor
        
    def dequantize_tensor(self, tensor_name: str, q_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dequantize a tensor using parameters stored in the state.
        
        Args:
            tensor_name: Name of the tensor in the state
            q_tensor: The quantized tensor to dequantize
            
        Returns:
            The dequantized tensor (full precision)
        """
        # Get parameters from state
        params = self.get_tensor_params(tensor_name)
        if params is None:
            raise ValueError(f"No parameters found for tensor '{tensor_name}' in state")
        
        bits = params.get('bits', 8)
        quant_type = params.get('type', QuantizationType.LINEAR.value)
        scale = params.get('scale')
        zero_point = params.get('zero_point')
        
        if scale is None or zero_point is None:
            raise ValueError(f"Missing scale or zero_point for tensor '{tensor_name}'")
        
        if bits == 8:
            from ..functional.quantization import dequantize_8bit
            return dequantize_8bit(
                q_tensor, 
                scale, 
                zero_point, 
                quant_type=quant_type
            )
        elif bits == 4:
            from ..functional.quantization import dequantize_4bit
            return dequantize_4bit(
                q_tensor, 
                scale, 
                zero_point, 
                quant_type=quant_type
            )
        else:
            raise ValueError(f"Unsupported bit depth: {bits}")