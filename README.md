# Quanta

Lightweight quantization primitives for PyTorch — int8 / int4 / NF4 / FP4 / NF8 / FP8 in one consistent API, with Triton kernels for NF4/FP4 encode-decode and fused 4-bit weight matmul, plus `torch._int_mm` integration for int8 Linear on CUDA.

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

`benchmarks/bench_vs_bnb.py` measures Quanta against bitsandbytes for quantize/dequantize latency, round-trip MAE, Linear forward latency, scaling with matrix size, and storage footprint. RTX 3050 6GB at `block_size=64`, weight matrix 4096 × 11008 (~45 M elements).

### Summary

| Comparison        | Quanta wins             | bitsandbytes wins         | Tie                              |
|-------------------|-------------------------|---------------------------|----------------------------------|
| Quantize latency  | int8 (2.80 vs 4.28 ms)  | nf4, fp4 (~2× slower)     | —                                |
| Dequantize        | —                       | nf4, fp4 (~3.3× slower)   | int8 (1.51 vs 1.48 ms)           |
| Round-trip MAE    | int8 (33% lower error)  | —                         | nf4, fp4 (codebooks identical)   |
| Linear forward    | —                       | all schemes (1.8–4× gap)  | —                                |
| Scaling vs size   | int8 at large dim       | nf4, fp4 across all sizes | —                                |
| Storage footprint | —                       | —                         | all schemes identical            |

- **int8 quantize/MAE**: Quanta wins on both speed (~1.5×) and accuracy (33% lower MAE). 
- **nf4 / fp4**: bit-identical numerics (same codebook, same MAE). 

## API

```python
from Quanta import quantize, dequantize, QuantState, Linear8bitLt, Linear4bit

state: QuantState = quantize(tensor, scheme="nf4", block_size=64)
out = dequantize(state)
```

`QuantState` is a dataclass holding the packed qdata, per-block absmax, codebook (for NF/FP), shape, dtype, and packing flag — everything needed to dequantize.

`Linear8bitLt` and `Linear4bit` quantize their weight in-place; the quantized buffers move with `.to(device)` and round-trip through `state_dict` / `load_state_dict`.

## Status

`0.2.0` — quantization primitives + Linear modules are stable. CUDA paths use Triton (NF4 / FP4 encode-decode, fused 4-bit weight matmul) and `torch._int_mm` for int8 Linear; full hand-written CUDA kernels are on the roadmap.

## License

MIT. Inspired by [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).
