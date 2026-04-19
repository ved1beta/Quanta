"""Model-level quantization with per-layer configuration and calibration."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..functional.quantization import quantize_4bit, quantize_8bit
from ..functional.state import QuantizationState


class ModelQuantize:
    """Quantize an ``nn.Module`` with per-layer configuration + calibration.

    Weights and (optionally) activations are quantized in-place on a copy of
    the source model; quantization parameters are stashed on the attached
    ``QuantizationState`` for save/load.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        scheme: str = "symmetric",
        quant_type: str = "linear",
    ):
        self.model = model
        self.bits = bits
        self.scheme = scheme
        self.quant_type = quant_type
        self.layer_config: Dict[str, Dict[str, Any]] = {}
        self.quantized_model: Optional[nn.Module] = None
        self.state = QuantizationState()
        self.activation_stats: Dict[str, Dict[str, Any]] = {}
        self.activation_hooks: Dict[str, Any] = {}

    def config_layer(
        self,
        layer_name: str,
        bits: int,
        scheme: Optional[str] = None,
        weights_only: bool = False,
        quant_type: Optional[str] = None,
        calibration_method: str = "minmax",
    ):
        self.layer_config[layer_name] = {
            "bits": bits,
            "scheme": scheme if scheme is not None else self.scheme,
            "weights_only": weights_only,
            "quant_type": quant_type if quant_type is not None else self.quant_type,
            "calibration_method": calibration_method,
        }

    def _get_layer_config(self, layer_name: str) -> Dict[str, Any]:
        return self.layer_config.get(layer_name, {
            "bits": self.bits,
            "scheme": self.scheme,
            "weights_only": False,
            "quant_type": self.quant_type,
            "calibration_method": "minmax",
        })

    def _quantize_tensor(self, tensor: torch.Tensor, config: Dict[str, Any]) -> tuple:
        bits = config["bits"]
        symmetric = config["scheme"] == "symmetric"
        if bits == 8:
            return quantize_8bit(tensor, quant_type=config["quant_type"], per_channel=True)
        if bits == 4:
            return quantize_4bit(tensor, quant_type=config["quant_type"], per_channel=True)
        raise ValueError(f"Unsupported bit depth: {bits}")

    def _pack_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Vectorized 4-bit packing: two nibbles per byte."""
        if bits != 4:
            return tensor
        flat = tensor.reshape(-1).to(torch.uint8)
        if flat.numel() % 2 == 1:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
        pairs = flat.view(-1, 2)
        return (pairs[:, 0] | (pairs[:, 1] << 4)).to(torch.uint8)

    def _unpack_tensor(self, packed: torch.Tensor, original_shape: Tuple[int, ...], bits: int) -> torch.Tensor:
        if bits != 4:
            return packed
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        out = torch.stack([low, high], dim=-1).reshape(-1)
        n = 1
        for s in original_shape:
            n *= s
        return out[:n].reshape(original_shape).to(torch.uint8)

    def _quantize_module(self, module: nn.Module, name: str) -> None:
        config = self._get_layer_config(name)
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            q_tensor, scale, zero_point = self._quantize_tensor(param.data, config)
            if config["bits"] == 4:
                q_tensor = self._pack_tensor(q_tensor, config["bits"])
            self.state.set_tensor_params(f"{name}.{param_name}", {
                "bits": config["bits"],
                "scheme": config["scheme"],
                "quant_type": config["quant_type"],
                "scale": scale,
                "zero_point": zero_point,
                "original_shape": tuple(param.data.shape),
            })
            param.data = q_tensor

    def _collect_activation_stats(self, module: nn.Module, name: str, input, output):
        stats = self.activation_stats.setdefault(name, {
            "min": float("inf"),
            "max": float("-inf"),
            "histogram": torch.zeros(256),
        })
        x = input[0] if isinstance(input, tuple) else input
        stats["min"] = min(stats["min"], x.min().item())
        stats["max"] = max(stats["max"], x.max().item())
        stats["histogram"] += torch.histc(x.float(), bins=256, min=stats["min"], max=stats["max"])

    def _calibrate_layer(self, layer_name: str, method: str = "minmax") -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_name not in self.activation_stats:
            raise ValueError(f"No activation statistics for layer {layer_name}")
        stats = self.activation_stats[layer_name]

        if method == "minmax":
            min_val, max_val = stats["min"], stats["max"]
        elif method in ("entropy", "percentile"):
            cdf = torch.cumsum(stats["histogram"], dim=0) / stats["histogram"].sum().clamp(min=1)
            low, high = (0.0005, 0.9995) if method == "entropy" else (0.01, 0.99)
            min_idx = torch.where(cdf > low)[0][0]
            max_idx = torch.where(cdf > high)[0][0]
            span = stats["max"] - stats["min"]
            min_val = stats["min"] + span * min_idx.item() / 255
            max_val = stats["min"] + span * max_idx.item() / 255
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        scale = (max_val - min_val) / (2 ** self.bits - 1)
        zero_point = min_val
        return scale, zero_point

    def _quantize_activations(self, module: nn.Module, name: str, inputs):
        config = self._get_layer_config(name)
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        if config["weights_only"]:
            return x

        scale, zero_point = self._calibrate_layer(name, config["calibration_method"])
        fn = quantize_8bit if config["bits"] == 8 else quantize_4bit
        q_tensor, _, _ = fn(x, quant_type=config["quant_type"], per_channel=False)

        self.state.set_tensor_params(f"{name}.activation", {
            "bits": config["bits"],
            "scheme": config["scheme"],
            "quant_type": config["quant_type"],
            "scale": scale,
            "zero_point": zero_point,
        })
        return q_tensor

    def _register_activation_hooks(self):
        for name, module in self.quantized_model.named_modules():
            if any(True for _ in module.parameters(recurse=False)):
                self.activation_hooks[name] = module.register_forward_pre_hook(
                    lambda m, i, name=name: self._quantize_activations(m, name, i)
                )

    def _remove_activation_hooks(self):
        for hook in self.activation_hooks.values():
            hook.remove()
        self.activation_hooks.clear()

    def quantize(self, calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        self.quantized_model = type(self.model)()
        self.quantized_model.load_state_dict(self.model.state_dict())

        if calibration_data is not None:
            hooks = []
            for name, module in self.quantized_model.named_modules():
                if any(True for _ in module.parameters(recurse=False)):
                    hooks.append(module.register_forward_hook(
                        lambda m, i, o, name=name: self._collect_activation_stats(m, name, i, o)
                    ))
            with torch.no_grad():
                self.quantized_model(calibration_data)
            for hook in hooks:
                hook.remove()

        for name, module in self.quantized_model.named_modules():
            if any(True for _ in module.parameters(recurse=False)):
                self._quantize_module(module, name)

        self._register_activation_hooks()
        return self.quantized_model

    def optimize_for_inference(self):
        if self.quantized_model is None:
            raise ValueError("Model must be quantized before optimization")
        self._remove_activation_hooks()
        self.quantized_model.eval()
        return self.quantized_model

    def save_quantized_model(self, path: str, export_format: str = "torch") -> None:
        if self.quantized_model is None:
            raise ValueError("Model must be quantized before saving")

        if export_format == "torch":
            torch.save(self.quantized_model.state_dict(), f"{path}_model.pt")
            self.state.save_state(f"{path}_state.json")
        elif export_format == "onnx":
            import onnx  # noqa: F401  (optional dependency; imported lazily)
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                self.quantized_model, dummy_input, f"{path}.onnx",
                opset_version=13, do_constant_folding=True,
            )
        elif export_format == "torchscript":
            scripted_model = torch.jit.script(self.quantized_model)
            scripted_model.save(f"{path}.pt")
        else:
            raise ValueError(f"Unknown export_format: {export_format}")

    def load_quantized_model(self, path: str, load_format: str = "torch") -> nn.Module:
        if load_format == "torch":
            state_dict = torch.load(f"{path}_model.pt")
            self.quantized_model = type(self.model)()
            self.quantized_model.load_state_dict(state_dict)
            self.state.load_state(f"{path}_state.json")
        elif load_format == "torchscript":
            self.quantized_model = torch.jit.load(f"{path}.pt")
        else:
            raise ValueError(f"Unknown load_format: {load_format}")
        return self.quantized_model

    def calibrate(self, calibration_data, method: str = "minmax"):
        self._register_activation_hooks()
        self._collect_stats_from_data(calibration_data)

        for name in self.activation_stats:
            scale, zero_point = self._calibrate_layer(name, method)
            self.state.set_tensor_params(f"{name}.activation", {
                "scale": scale,
                "zero_point": zero_point,
                "calibration_method": method,
            })

    def _collect_stats_from_data(self, calibration_data):
        with torch.no_grad():
            if hasattr(calibration_data, "__iter__") and not isinstance(calibration_data, torch.Tensor):
                for batch in calibration_data:
                    self.quantized_model(batch)
            else:
                self.quantized_model(calibration_data)
