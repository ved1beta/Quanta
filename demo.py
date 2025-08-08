import torch
from Quanta.functional.quantization import quantize_4bit, dequantize_4bit
model_weights = torch.randn(10, 10)  # Example tensor
q_tensor, scale, zero_point = quantize_4bit(model_weights, quant_type="nf4")
print("Quantized Tensor:", q_tensor)
deq_tensor = dequantize_4bit(q_tensor, scale, zero_point, quant_type="nf4")
print("Dequantized Tensor:", deq_tensor)