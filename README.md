# Quanta 🚀

A lightweight PyTorch library for efficient model quantization and memory optimization. Perfect for running large language models on consumer hardware.

## Key Features

- 🎯 8-bit & 4-bit quantization primitives
- 💾 Memory-efficient optimizers
- 🚀 LLM.int8() inference support
- 🔄 QLoRA-style fine-tuning
- 🖥️ Cross-platform hardware support

## Quick Start

```python
import torch
from bytesandbits.functional.quantization import quantize_8bit, dequantize_8bit

# Quantize your model
q_tensor, scale, zero_point = quantize_8bit(model_weights)
```

## Status

🚧 Early Development - Currently implementing core quantization features.

## License

MIT License

Inspired by [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
