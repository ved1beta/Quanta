"""Layer/tensor quantization state registry for model-level workflows."""

import json
import os
from enum import Enum
from typing import Dict, Optional

import torch


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
    """Holds per-tensor and per-layer quantization metadata plus defaults.

    Serializes to JSON (small metadata) or to a ``.pt`` file (when tensors
    need to travel with the state).
    """

    def __init__(self):
        self.tensor_params: Dict[str, Dict] = {}
        self.layers_params: Dict[str, Dict] = {}
        self.global_config = {
            "default_bits": 8,
            "default_scheme": QuantizationScheme.SYMMETRIC.value,
            "default_type": QuantizationType.LINEAR.value,
        }

    def set_tensor_params(self, tensor_id: str, params: Dict):
        self.tensor_params[tensor_id] = params

    def get_tensor_params(self, tensor_id: str) -> Optional[Dict]:
        return self.tensor_params.get(tensor_id)

    def set_layer_params(self, layer_name: str, params: Dict):
        self.layers_params[layer_name] = params

    def get_layer_params(self, layer_name: str) -> Optional[Dict]:
        return self.layers_params.get(layer_name)

    def update_global_config(self, config_updates: Dict):
        self.global_config.update(config_updates)

    def save_state(self, filepath: str):
        """Save to JSON. Tensor values are converted to plain lists."""
        serializable = {}
        for tid, params in self.tensor_params.items():
            serializable[tid] = {
                k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in params.items()
            }

        with open(filepath, "w") as f:
            json.dump({
                "tensor_params": serializable,
                "layers_params": self.layers_params,
                "global_config": self.global_config,
            }, f, indent=2)

    def load_state(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found: {filepath}")

        with open(filepath) as f:
            state_dict = json.load(f)

        self.global_config = state_dict.get("global_config", self.global_config)
        self.layers_params = state_dict.get("layers_params", {})

        for tid, params in state_dict.get("tensor_params", {}).items():
            for key, value in params.items():
                if isinstance(value, list):
                    params[key] = torch.tensor(value)
            self.tensor_params[tid] = params

    def save_quantized_tensor_with_state(self, tensor_name: str, q_tensor: torch.Tensor, file_path: str):
        from ..utils.utils import save_quantized_tensor, save_quantized_tensor_torch

        params = self.get_tensor_params(tensor_name)
        if params is None:
            raise ValueError(f"No parameters found for tensor '{tensor_name}' in state")

        scale = params.get("scale")
        zero_point = params.get("zero_point")
        if scale is None or zero_point is None:
            raise ValueError(f"Missing scale or zero_point for tensor '{tensor_name}'")

        if file_path.endswith(".pt"):
            save_quantized_tensor_torch(q_tensor, scale, zero_point, params, file_path)
        else:
            save_quantized_tensor(q_tensor, scale, zero_point, params, file_path)

    def load_quantized_tensor_with_state(self, tensor_name: str, file_path: str) -> torch.Tensor:
        from ..utils.utils import load_quantized_tensor, load_quantized_tensor_torch

        if file_path.endswith(".pt"):
            q_tensor, scale, zero_point, params = load_quantized_tensor_torch(file_path)
        else:
            q_tensor, scale, zero_point, params = load_quantized_tensor(file_path)

        params.setdefault("scale", scale)
        params.setdefault("zero_point", zero_point)
        self.set_tensor_params(tensor_name, params)

        if not hasattr(self, "_quantized_tensors"):
            self._quantized_tensors = {}
        self._quantized_tensors[tensor_name] = q_tensor
        return q_tensor

    def convert_tensor_precision(
        self,
        tensor_name: str,
        target_bits: int,
        target_type: str = "linear",
        target_scheme: Optional[str] = None,
    ) -> torch.Tensor:
        from ..utils.utils import convert_precision

        source_params = self.get_tensor_params(tensor_name)
        if source_params is None:
            raise ValueError(f"No parameters found for tensor '{tensor_name}' in state")

        if not hasattr(self, "_quantized_tensors") or tensor_name not in self._quantized_tensors:
            raise ValueError(
                f"Quantized tensor '{tensor_name}' not found in state. "
                "Call load_quantized_tensor_with_state first."
            )
        q_tensor = self._quantized_tensors[tensor_name]

        new_q, _, _, new_params = convert_precision(
            q_tensor, source_params, target_bits, target_type, target_scheme,
        )
        self.set_tensor_params(tensor_name, new_params)
        self._quantized_tensors[tensor_name] = new_q
        return new_q

    def dequantize_tensor(self, tensor_name: str, q_tensor: torch.Tensor) -> torch.Tensor:
        params = self.get_tensor_params(tensor_name)
        if params is None:
            raise ValueError(f"No parameters found for tensor '{tensor_name}' in state")

        bits = params.get("bits", 8)
        quant_type = params.get("type", QuantizationType.LINEAR.value)
        scale = params.get("scale")
        zero_point = params.get("zero_point")
        if scale is None or zero_point is None:
            raise ValueError(f"Missing scale or zero_point for tensor '{tensor_name}'")

        if bits == 8:
            from ..functional.quantization import dequantize_8bit
            return dequantize_8bit(q_tensor, scale, zero_point, quant_type=quant_type)
        if bits == 4:
            from ..functional.quantization import dequantize_4bit
            return dequantize_4bit(q_tensor, scale, zero_point, quant_type=quant_type)
        raise ValueError(f"Unsupported bit depth: {bits}")
