import torch 
import torch.nn as nn
import onnx
from typing import Optional, Dict, Any, Union, List, Tuple
from ..functional.quantization import quantize_8bit, quantize_4bit
from ..functional.state import QuantizationState

class ModelQuantize:
    def __init__(self, 
                 model: nn.Module, 
                 bits: int = 8, 
                 scheme: str = "symmetric",
                 quant_type: str = "linear"
                 ):
        self.model = model 
        self.bits = bits
        self.scheme = scheme
        self.quant_type = quant_type
        self.layer_config = {}
        self.quantized_model = None
        self.state = QuantizationState()
        self.activation_stats = {}
        self.activation_hooks = {}
     
    def config_layer(
            self, 
            layer_name: str, 
            bits: int, 
            scheme: str = None, 
            weights_only: bool = False,
            quant_type: str = None,
            calibration_method: str = "minmax"):
        """Configure quantization parameters for a specific layer.
        
        Args:
            layer_name: Name of the layer to configure
            bits: Number of bits for quantization (4 or 8)
            scheme: Quantization scheme (symmetric/asymmetric)
            weights_only: If True, only quantize weights, not activations
            quant_type: Type of quantization (linear, nf4, fp4, etc.)
            calibration_method: Method for calibration (minmax, entropy, percentile)
        """
        self.layer_config[layer_name] = {
            'bits': bits,
            'scheme': scheme if scheme is not None else self.scheme,
            'weights_only': weights_only,
            'quant_type': quant_type if quant_type is not None else self.quant_type,
            'calibration_method': calibration_method
        }

    def _get_layer_config(self, layer_name: str) -> Dict[str, Any]:
        return self.layer_config.get(layer_name, {
            'bits': self.bits,
            'scheme': self.scheme,
            'weights_only': False,
            'quant_type': self.quant_type,
            'calibration_method': "minmax"
        })
    
    def _quantize_tensor(self, tensor: torch.Tensor, config: Dict[str, Any]) -> tuple:
        """Quantize a single tensor based on configuration."""
        bits = config['bits']
        scheme = config['scheme']
        quant_type = config['quant_type']
        
        if bits == 8:
            return quantize_8bit(tensor, quant_type=quant_type, per_channel=True, symmetric=(scheme == "symmetric"))
        elif bits == 4:
            return quantize_4bit(tensor, quant_type=quant_type, per_channel=True, symmetric=(scheme == "symmetric"))
        else:
            raise ValueError(f"Unsupported bit depth: {bits}")

    def _pack_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack a quantized tensor for efficient storage."""
        if bits == 4:
            packed = torch.zeros((tensor.numel() + 1) // 2, dtype=torch.uint8)
            for i in range(0, tensor.numel(), 2):
                val1 = tensor[i].item()
                val2 = tensor[i + 1].item() if i + 1 < tensor.numel() else 0
                packed[i // 2] = (val1 << 4) | val2
            return packed
        return tensor

    def _unpack_tensor(self, packed: torch.Tensor, original_shape: Tuple[int, ...], bits: int) -> torch.Tensor:
        """Unpack a packed tensor back to its original form."""
        if bits == 4:
            unpacked = torch.zeros(original_shape, dtype=torch.uint8)
            for i in range(0, packed.numel()):
                val = packed[i].item()
                unpacked[i * 2] = (val >> 4) & 0xF
                if i * 2 + 1 < unpacked.numel():
                    unpacked[i * 2 + 1] = val & 0xF
            return unpacked
        return packed

    def _quantize_module(self, module: nn.Module, name: str) -> None:
        """Quantize a single module's parameters."""
        config = self._get_layer_config(name)
        
        for param_name, param in module.named_parameters():
            if param.requires_grad:  # Only quantize trainable parameters
                q_tensor, scale, zero_point = self._quantize_tensor(param.data, config)
                
                # Pack tensor if using 4-bit quantization
                if config['bits'] == 4:
                    q_tensor = self._pack_tensor(q_tensor, config['bits'])
                
                state_key = f"{name}.{param_name}"
                self.state.set_tensor_params(state_key, {
                    'bits': config['bits'],
                    'scheme': config['scheme'],
                    'quant_type': config['quant_type'],
                    'scale': scale,
                    'zero_point': zero_point,
                    'original_shape': param.data.shape
                })
                
                param.data = q_tensor

    def _collect_activation_stats(self, module: nn.Module, name: str, input: torch.Tensor, output: torch.Tensor):
        """Collect activation statistics for calibration."""
        if name not in self.activation_stats:
            self.activation_stats[name] = {
                'min': float('inf'),
                'max': float('-inf'),
                'histogram': torch.zeros(256)
            }
        
        stats = self.activation_stats[name]
        stats['min'] = min(stats['min'], input.min().item())
        stats['max'] = max(stats['max'], input.max().item())
        
        # Update histogram
        hist = torch.histc(input.float(), bins=256, min=stats['min'], max=stats['max'])
        stats['histogram'] += hist

    def _calibrate_layer(self, layer_name: str, method: str = "minmax") -> Tuple[torch.Tensor, torch.Tensor]:
        """Calibrate quantization parameters for a layer using different methods.
        
        Args:
            layer_name: Name of the layer to calibrate
            method: Calibration method (minmax, entropy, percentile)
            
        Returns:
            Tuple of (scale, zero_point) for quantization
        """
        if layer_name not in self.activation_stats:
            raise ValueError(f"No activation statistics found for layer {layer_name}")
            
        stats = self.activation_stats[layer_name]
        
        if method == "minmax":
            # Simple min-max calibration
            min_val = stats['min']
            max_val = stats['max']
            scale = (max_val - min_val) / (2**self.bits - 1)
            zero_point = min_val
            
        elif method == "entropy":
            # Entropy-based calibration
            hist = stats['histogram']
            total = hist.sum()
            cdf = torch.cumsum(hist, dim=0) / total
            
            # Find the range that contains 99.9% of the data
            min_idx = torch.where(cdf > 0.0005)[0][0]
            max_idx = torch.where(cdf > 0.9995)[0][0]
            
            min_val = stats['min'] + (stats['max'] - stats['min']) * min_idx / 255
            max_val = stats['min'] + (stats['max'] - stats['min']) * max_idx / 255
            
            scale = (max_val - min_val) / (2**self.bits - 1)
            zero_point = min_val
            
        elif method == "percentile":
            # Percentile-based calibration
            hist = stats['histogram']
            total = hist.sum()
            cdf = torch.cumsum(hist, dim=0) / total
            
            # Use 1st and 99th percentiles
            min_idx = torch.where(cdf > 0.01)[0][0]
            max_idx = torch.where(cdf > 0.99)[0][0]
            
            min_val = stats['min'] + (stats['max'] - stats['min']) * min_idx / 255
            max_val = stats['min'] + (stats['max'] - stats['min']) * max_idx / 255
            
            scale = (max_val - min_val) / (2**self.bits - 1)
            zero_point = min_val
            
        else:
            raise ValueError(f"Unknown calibration method: {method}")
            
        return scale, zero_point

    def _quantize_activations(self, module: nn.Module, name: str, input: torch.Tensor) -> torch.Tensor:
        """Quantize activations for a module during forward pass.
        
        Args:
            module: The module being processed
            name: Name of the module
            input: Input tensor to quantize
            
        Returns:
            Quantized input tensor
        """
        config = self._get_layer_config(name)
        if config['weights_only']:
            return input
            
        scale, zero_point = self._calibrate_layer(name, config['calibration_method'])
        
        if config['bits'] == 8:
            q_tensor, _, _ = quantize_8bit(
                input, 
                quant_type=config['quant_type'],
                per_channel=False,  # Activations are usually quantized per-tensor
                symmetric=(config['scheme'] == "symmetric")
            )
        else: 
            q_tensor, _, _ = quantize_4bit(
                input,
                quant_type=config['quant_type'],
                per_channel=False,
                symmetric=(config['scheme'] == "symmetric")
            )
            
        state_key = f"{name}.activation"
        self.state.set_tensor_params(state_key, {
            'bits': config['bits'],
            'scheme': config['scheme'],
            'quant_type': config['quant_type'],
            'scale': scale,
            'zero_point': zero_point
        })
        
        return q_tensor

    def _register_activation_hooks(self):
        """Register forward hooks for activation quantization."""
        for name, module in self.quantized_model.named_modules():
            if len(list(module.parameters())) > 0:
                hook = module.register_forward_pre_hook(
                    lambda m, i, name=name: self._quantize_activations(m, name, i[0])
                )
                self.activation_hooks[name] = hook

    def _remove_activation_hooks(self):
        """Remove all activation quantization hooks."""
        for hook in self.activation_hooks.values():
            hook.remove()
        self.activation_hooks.clear()

    def quantize(self, calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """Quantize the entire model.
        
        Args:
            calibration_data: Optional tensor for calibration
            
        Returns:
            The quantized model
        """
        self.quantized_model = type(self.model)()
        self.quantized_model.load_state_dict(self.model.state_dict())
        
        if calibration_data is not None:
            hooks = []
            for name, module in self.quantized_model.named_modules():
                if len(list(module.parameters())) > 0:
                    hook = module.register_forward_hook(
                        lambda m, i, o, name=name: self._collect_activation_stats(m, name, i[0], o)
                    )
                    hooks.append(hook)
            
            with torch.no_grad():
                self.quantized_model(calibration_data)
            
            for hook in hooks:
                hook.remove()
        
        # Quantize weights
        for name, module in self.quantized_model.named_modules():
            if len(list(module.parameters())) > 0:
                self._quantize_module(module, name)
        
        # Register activation quantization hooks
        self._register_activation_hooks()
        
        return self.quantized_model

    def optimize_for_inference(self):
        """Optimize the quantized model for inference."""
        if self.quantized_model is None:
            raise ValueError("Model must be quantized before optimization")
            
        self._remove_activation_hooks()
        
        self.quantized_model.eval()
        torch.quantization.fuse_modules(
            self.quantized_model,
            [['conv', 'bn', 'relu']],
            inplace=True
        )
        
        return self.quantized_model

    def save_quantized_model(self, path: str, export_format: str = "torch") -> None:
        """Save the quantized model and its quantization state.
        
        Args:
            path: Path to save the model and state
            export_format: Format to export the model (torch, onnx, torchscript)
        """
        if self.quantized_model is None:
            raise ValueError("Model must be quantized before saving")
        
        if export_format == "torch":
            torch.save(self.quantized_model.state_dict(), f"{path}_model.pt")
            self.state.save(f"{path}_state.pt")
        elif export_format == "onnx":
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                self.quantized_model,
                dummy_input,
                f"{path}.onnx",
                opset_version=13,
                do_constant_folding=True
            )
        elif export_format == "torchscript":
            scripted_model = torch.jit.script(self.quantized_model)
            scripted_model.save(f"{path}.pt")

    def load_quantized_model(self, path: str, load_format: str = "torch") -> nn.Module:
        """Load a quantized model and its quantization state.
        
        Args:
            path: Path to load the model and state from
            load_format: Format of the saved model (torch, onnx, torchscript)
            
        Returns:
            The loaded quantized model
        """
        if load_format == "torch":
            state_dict = torch.load(f"{path}_model.pt")
            self.quantized_model = type(self.model)()
            self.quantized_model.load_state_dict(state_dict)
            self.state.load(f"{path}_state.pt")
        elif load_format == "onnx":
            onnx_model = onnx.load(f"{path}.onnx")
            self.quantized_model = self._onnx_to_pytorch(onnx_model)
        elif load_format == "torchscript":
            self.quantized_model = torch.jit.load(f"{path}.pt")
        
        return self.quantized_model
    
    def calibrate(self, calibration_data, method="minmax"):
        """
        Calibrate quantization parameters using a dataset.

        Args:
            calibration_data: PyTorch DataLoader or tensor batch
            method: Calibration method - "minmax", "entropy", or "percentile"
        """
        self._register_activation_hooks()
        
        self._collect_stats_from_data(calibration_data)
        
        if method == "minmax":
            self._calibrate_minmax()
        elif method == "entropy":
            self._calibrate_entropy()
        elif method == "percentile":
            self._calibrate_percentile(percentile=99.9)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
    def _calibrate_minmax(self):
        for name in self.activation_stats.keys():
            scale , zero_point = self._calibrate_layer(name , "minmax")
            self.state.set_tensor_params(f"{name}.activation", {
            'scale': scale,
            'zero_point': zero_point,
            'calibration_method': 'minmax'
        })
            
    def _calibrate_entropy(self):
        for name in self.activation_stats.keys():
            scale, zero_point = self._calibrate_layer(name, "entropy")
            self.state.set_tensor_params(f"{name}.activation", {
            'scale': scale,
            'zero_point': zero_point,
            'calibration_method': 'entropy'
        })

    def _calibrate_percentile(self, percentile=99.9):
        for name in self.activation_stats.keys():
            scale, zero_point = self._calibrate_layer(name, "percentile")
            self.state.set_tensor_params(f"{name}.activation", {
                'scale': scale,
                'zero_point': zero_point,
                'calibration_method': 'percentile',
                'percentile': percentile
            })







        
        


        
        
    

