import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Quanta.functional.model import ModelQuantize

class SimpleConvNet(nn.Module):
    """A simple convolutional neural network for demonstration."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def benchmark_model(model, input_data, num_runs=100, device='cuda'):
    """Benchmark model performance.
    
    Args:
        model: Model to benchmark
        input_data: Input tensor for benchmarking
        num_runs: Number of runs for timing
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing benchmark results
    """
    model = model.to(device)
    input_data = input_data.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(input_data)
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                model(input_data)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            else:
                start_time = time.time()
                model(input_data)
                torch.cuda.synchronize()
                times.append((time.time() - start_time) * 1000)  # Convert to ms
    
    # Memory usage
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(input_data)
        max_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    else:
        max_memory = 0  # CPU memory tracking not implemented
    
    return {
        'mean_latency_ms': sum(times) / len(times),
        'std_latency_ms': torch.tensor(times).std().item(),
        'max_memory_mb': max_memory
    }

def main():
    if torch.cuda.is_available():
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.zeros(1, device=device)
            del test_tensor
        except RuntimeError:
            print("CUDA device is busy or unavailable, falling back to CPU")
            device = torch.device('cpu')
    else:
        print("CUDA is not available, using CPU")
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleConvNet()
    model = model.to(device)
    
    # Create example input
    input_data = torch.randn(1, 3, 32, 32).to(device)
    
    # Benchmark original model
    print("\nBenchmarking original model...")
    original_results = benchmark_model(model, input_data)
    print(f"Original model - Mean latency: {original_results['mean_latency_ms']:.2f}ms")
    print(f"Original model - Memory usage: {original_results['max_memory_mb']:.2f}MB")
    
    # Example 1: Basic 8-bit quantization
    print("\nExample 1: Basic 8-bit quantization")
    quantizer = ModelQuantize(model, bits=8, scheme="symmetric")
    quantized_model = quantizer.quantize()
    results = benchmark_model(quantized_model, input_data)
    print(f"8-bit quantized - Mean latency: {results['mean_latency_ms']:.2f}ms")
    print(f"8-bit quantized - Memory usage: {results['max_memory_mb']:.2f}MB")
    
    # Example 2: Mixed precision quantization
    print("\nExample 2: Mixed precision quantization")
    quantizer = ModelQuantize(model, bits=8, scheme="symmetric")
    # Use 4-bit for conv layers, 8-bit for fc layers
    quantizer.config_layer("conv1", bits=4, calibration_method="entropy")
    quantizer.config_layer("conv2", bits=4, calibration_method="entropy")
    quantizer.config_layer("fc1", bits=8, calibration_method="percentile")
    quantizer.config_layer("fc2", bits=8, calibration_method="percentile")
    
    # Use calibration data
    calibration_data = torch.randn(32, 3, 32, 32).to(device)
    quantized_model = quantizer.quantize(calibration_data)
    results = benchmark_model(quantized_model, input_data)
    print(f"Mixed precision - Mean latency: {results['mean_latency_ms']:.2f}ms")
    print(f"Mixed precision - Memory usage: {results['max_memory_mb']:.2f}MB")
    
    # Example 3: Weight-only quantization
    print("\nExample 3: Weight-only quantization")
    quantizer = ModelQuantize(model, bits=8, scheme="symmetric")
    # Quantize only weights, not activations
    quantizer.config_layer("conv1", bits=8, weights_only=True)
    quantizer.config_layer("conv2", bits=8, weights_only=True)
    quantizer.config_layer("fc1", bits=8, weights_only=True)
    quantizer.config_layer("fc2", bits=8, weights_only=True)
    
    quantized_model = quantizer.quantize()
    results = benchmark_model(quantized_model, input_data)
    print(f"Weight-only - Mean latency: {results['mean_latency_ms']:.2f}ms")
    print(f"Weight-only - Memory usage: {results['max_memory_mb']:.2f}MB")
    
    # Example 4: Model optimization and export
    print("\nExample 4: Model optimization and export")
    quantizer = ModelQuantize(model, bits=8, scheme="symmetric")
    quantized_model = quantizer.quantize()
    
    # Optimize for inference
    optimized_model = quantizer.optimize_for_inference()
    results = benchmark_model(optimized_model, input_data)
    print(f"Optimized - Mean latency: {results['mean_latency_ms']:.2f}ms")
    print(f"Optimized - Memory usage: {results['max_memory_mb']:.2f}MB")
    
    # Save model in different formats
    print("\nSaving model in different formats...")
    quantizer.save_quantized_model("model_quantized", export_format="torch")
    quantizer.save_quantized_model("model_quantized", export_format="onnx")
    quantizer.save_quantized_model("model_quantized", export_format="torchscript")
    
    # Example 5: Loading quantized model
    print("\nExample 5: Loading quantized model")
    loaded_model = quantizer.load_quantized_model("model_quantized", load_format="torch")
    results = benchmark_model(loaded_model, input_data)
    print(f"Loaded model - Mean latency: {results['mean_latency_ms']:.2f}ms")
    print(f"Loaded model - Memory usage: {results['max_memory_mb']:.2f}MB")
    
    # Print summary
    print("\nSummary of quantization results:")
    print(f"{'Model Type':<20} {'Latency (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 50)
    print(f"{'Original':<20} {original_results['mean_latency_ms']:<15.2f} {original_results['max_memory_mb']:<15.2f}")
    print(f"{'8-bit':<20} {results['mean_latency_ms']:<15.2f} {results['max_memory_mb']:<15.2f}")
    print(f"{'Mixed precision':<20} {results['mean_latency_ms']:<15.2f} {results['max_memory_mb']:<15.2f}")
    print(f"{'Weight-only':<20} {results['mean_latency_ms']:<15.2f} {results['max_memory_mb']:<15.2f}")
    print(f"{'Optimized':<20} {results['mean_latency_ms']:<15.2f} {results['max_memory_mb']:<15.2f}")

if __name__ == "__main__":
    main() 