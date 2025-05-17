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