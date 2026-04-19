# Quanta

Lightweight quantization primitives for PyTorch — int8 / int4 / NF4 / FP4 / NF8 / FP8 in one consistent API, with a Triton NF4 kernel and `torch._int_mm` integration on CUDA.

## Install

```bash
pip install -e .
```

Requires `torch >= 2.2`. Triton is optional (used to accelerate blockwise NF4 on CUDA); falls back to eager PyTorch otherwise.

## Quickstart

```python
import torch
from Quanta import Linear4bit, dequantize, quantize

# Functional: any tensor, any scheme
x = torch.randn(2048, 2048, device="cuda")
state = quantize(x, scheme="nf4", block_size=64)
recovered = dequantize(state)              # MAE ~ 0.073 on a standard normal

# nn module: drop-in for nn.Linear
layer = Linear4bit(1024, 4096, quant_type="nf4").cuda()
layer.quantize_weight(block_size=64)
y = layer(torch.randn(8, 1024, device="cuda"))
```

Schemes: `int8_linear`, `int4_linear`, `nf4`, `nf8`, `fp4`, `fp8`. Pass `block_size=N` for blockwise scales (robust to outliers).

## Benchmarks

`example/benchmarks.py` measures Quanta against bitsandbytes for quantize/dequantize latency, round-trip MAE, and bytes per element. Below: RTX 3050 6GB at `block_size=64`, weight matrix 4096 × 11008 (~45 M elements).

| Scheme       | Impl   | Quantize (µs) | Dequantize (µs) | MAE       | Bytes/elem |
|--------------|--------|---------------|-----------------|-----------|------------|
| int8_linear  | Quanta | **2,470**     | **1,467**       | **0.0050**| 1.062      |
| int8_linear  | bnb    | 3,639         | 1,465           | 0.0075    | 1.062      |
| nf4          | Quanta | 5,547         | 1,973           | 0.0728    | 0.562      |
| nf4          | bnb    | **2,658**     | **1,327**       | 0.0728    | 0.562      |
| fp4          | Quanta | 20,474        | 8,594           | 0.0965    | 0.562      |
| fp4          | bnb    | **2,564**     | **1,328**       | 0.0965    | 0.562      |

- **int8_linear**: Quanta is ~1.5× faster than bnb on quantize and matches on dequant. Lower MAE because Quanta uses linear absmax encoding while bnb's default is dynamic exponent.
- **nf4 / fp4**: bit-identical numerics (same codebook). bnb's hand-written CUDA kernels are 2× faster on NF4 and 8× faster on FP4 — Quanta's NF4 has a Triton kernel; FP4 still rides the eager codebook path.

Run `python example/benchmarks.py` to regenerate `benchmarks/results.md` and four PNG plots on your hardware.

## API

```python
from Quanta import quantize, dequantize, QuantState, Linear8bitLt, Linear4bit

state: QuantState = quantize(tensor, scheme="nf4", block_size=64)
out = dequantize(state)
```

`QuantState` is a dataclass holding the packed qdata, per-block absmax, codebook (for NF/FP), shape, dtype, and packing flag — everything needed to dequantize.

`Linear8bitLt` and `Linear4bit` quantize their weight in-place; the quantized buffers move with `.to(device)` and round-trip through `state_dict` / `load_state_dict`.

## Status

`0.2.0` — quantization primitives + Linear modules are stable. CUDA NF4 path is Triton; full custom CUDA kernels and a real int8 GEMM Linear are on the roadmap.

## License

MIT. Inspired by [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).
