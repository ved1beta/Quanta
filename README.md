# Quanta

Lightweight quantization primitives for PyTorch — int8 / int4 / NF4 / FP4 / NF8 / FP8 in one consistent API, with Triton kernels for NF4/FP4 encode-decode and fused 4-bit weight matmul, plus `torch._int_mm` integration for int8 Linear on CUDA.

## Install

```bash
pip install -e .
```

Requires `torch >= 2.2`. Triton is optional (used for NF4/FP4 encode-decode and the fused 4-bit matmul on CUDA); falls back to eager PyTorch otherwise.

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

### Whole-model quantization

Bring any `nn.Module` (plain torch or a Hugging Face model) and convert it to 4-bit or 8-bit in one call. Every `nn.Linear` is swapped for `Linear4bit` / `Linear8bitLt` in place; `lm_head` stays in full precision by default.

```python
from transformers import AutoModelForCausalLM
from Quanta import quantize_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16)
quantize_model(model, bits=4)                     # nf4, block_size=64
# quantize_model(model, bits=4, quant_type="fp4")
# quantize_model(model, bits=8, block_size=2048)  # per-row scales → torch._int_mm on CUDA

torch.save(model.state_dict(), "llama-1b-nf4.pt")

# Reload: rebuild the skeleton, quantize with the same args, then load.
model = quantize_model(AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B"), bits=4)
model.load_state_dict(torch.load("llama-1b-nf4.pt"))
```

`skip=("lm_head", "embed")` excludes modules by name substring; `compute_dtype` sets the 4-bit matmul dtype (defaults to the model's weight dtype).

## Benchmarks

`benchmarks/bench_vs_bnb.py` measures Quanta against bitsandbytes for quantize/dequantize latency, round-trip MAE, Linear forward latency, scaling with matrix size, and storage footprint (`--plot` regenerates `benchmarks/plots/`). Numbers below: NVIDIA A40, torch 2.14, bitsandbytes 0.50.2, `block_size=64`, weight matrix 11008 × 4096 (~45 M elements).

```bash
pip install -e ".[benchmark]"
python benchmarks/bench_vs_bnb.py --plot
```

### Summary

| Comparison        | Quanta wins                     | bitsandbytes wins                | Tie                            |
|-------------------|---------------------------------|----------------------------------|--------------------------------|
| Quantize latency  | int8 (0.38 vs 0.92 ms)          | nf4 (1.7×), fp4 (3.4×)           | —                              |
| Dequantize        | —                               | nf4, fp4 (~1.8×)                 | int8 (0.42 ms both)            |
| Round-trip MAE    | int8 (33% lower error)          | —                                | nf4, fp4 (codebooks identical) |
| Linear forward    | int8 at M=16                    | nf4/fp4 at all M (1.8–4.5×), int8 elsewhere | —                   |
| Storage footprint | —                               | —                                | all schemes identical          |

- **int8 quantize/MAE**: Quanta is ~2.4× faster to quantize and has 33% lower MAE.
- **nf4 / fp4**: bit-identical numerics (same codebook, same MAE).
- **4-bit Linear**: the fused Triton kernel is used for M ≤ 16 (decode); larger batches dequantize and call cuBLAS.

## API

```python
from Quanta import quantize, dequantize, QuantState, Linear8bitLt, Linear4bit, quantize_model

state: QuantState = quantize(tensor, scheme="nf4", block_size=64)
out = dequantize(state)
```

`QuantState` is a dataclass holding the packed qdata, per-block absmax, codebook (for NF/FP), shape, dtype, and packing flag — everything needed to dequantize.

`Linear8bitLt` and `Linear4bit` quantize their weight in-place; the quantized buffers move with `.to(device)` and round-trip through `state_dict` / `load_state_dict`.

## Status

`0.2.0` — quantization primitives + Linear modules are stable. CUDA paths use Triton (NF4 / FP4 encode-decode, fused 4-bit weight matmul) and `torch._int_mm` for int8 Linear; full hand-written CUDA kernels are on the roadmap.

## License

MIT. Inspired by [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).
