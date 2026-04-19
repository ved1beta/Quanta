# Quanta

Efficient quantization and memory-optimization primitives for PyTorch.
Compatible in spirit with [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes),
with a tighter Python-first API and a fused Triton path for NF4.

## Install

```bash
pip install quanta                       # core
pip install "quanta[triton]"             # enable the fused NF4 Triton kernel
pip install "quanta[bench]"              # for the vs-bitsandbytes benchmark suite
```

## Features

- **Quantization schemes** — `int8_linear`, `int4_linear`, `nf4`, `nf8`, `fp4`, `fp8`.
- **Blockwise scales** — outlier-robust absmax per block, ~2× lower MAE than per-tensor.
- **Packed 4-bit storage** — two nibbles per byte, ~8× fp32 compression.
- **Drop-in layers** — `Linear8bitLt`, `Linear4bit` with `.quantize_weight()` and `state_dict()` round-trip.
- **Fused NF4 kernel** — single-launch encode and decode on CUDA via Triton (bit-exact with the eager path).
- **Int8 GEMM** — `quantized_matmul_int8` routes through `torch._int_mm` for true int8 × int8 → int32.

## Quick start

### Functional API

```python
import torch
import Quanta

x = torch.randn(4096, 4096, device="cuda")

state = Quanta.quantize(x, scheme="nf4", block_size=64)
x_hat = Quanta.dequantize(state)          # ≈ x, MAE ~0.07 on N(0,1)

print(state.qdata.dtype, state.qdata.shape)       # torch.uint8, packed
print(state.qdata.numel() * 2, "nibbles")         # 16.7M, vs 67M fp32 bytes
```

### Quantized Linear layers

```python
import torch
from Quanta import Linear4bit

layer = Linear4bit(4096, 4096, quant_type="nf4", block_size=64).cuda()
layer.weight.data.copy_(fp32_weight)
layer.quantize_weight()                   # weight is now a packed 4-bit buffer

y = layer(x)                              # forward dequantizes in compute_dtype
```

`.state_dict()` persists the quantized tensors; loading into a freshly built
layer restores the full quantized state without materializing fp32 weights.

### Fused NF4 kernel

The Triton kernel is used automatically when:

- `scheme == "nf4"`
- `block_size` is set and even
- the input is on CUDA and Triton imports cleanly

Set `QUANTA_DISABLE_TRITON=1` to force the eager path, or
`QUANTA_DISABLE_COMPILE=1` to skip `torch.compile` on hot kernels.

## Benchmarks

Full reproducer in [example/bench_vs_bnb.py](example/bench_vs_bnb.py), plotting
in [example/plot_bench.py](example/plot_bench.py). Latest runs land in
[example/bench_plots/](example/bench_plots/):

| Plot | Shows |
|---|---|
| `01_speed_by_size.png` | round-trip time vs tensor side, per method |
| `02_speed_bars_largest.png` | horizontal bars at the largest tensor size |
| `03_accuracy_mae.png` | MAE per method (lower is better) |
| `04_accuracy_vs_speed.png` | pareto scatter — accuracy vs speed |
| `05_memory_compression.png` | compression ratio vs fp32 |
| `06_speed_heatmap.png` | method × size time heatmap |
| `07_dashboard.png` | single-page summary |

Run locally:

```bash
python example/bench_vs_bnb.py     # writes bench_results.json
python example/plot_bench.py       # writes bench_plots/
```

## Testing

```bash
pytest Quanta/tests -q
```

Parity tests cover every scheme on every available device and assert both
round-trip MAE ceilings and QuantState shape/device invariants.

## Project layout

```
Quanta/
├── functional/
│   ├── quantization.py   # canonical quantize / dequantize + QuantState
│   ├── tensor_ops.py     # quantized ops + int8 matmul
│   ├── base.py           # reference BaseQuantizer
│   ├── state.py          # layer/tensor param registry
│   └── model.py          # model-level quantizer with calibration
├── nn/linear.py          # Linear8bitLt, Linear4bit
├── optim/adam.py         # Adam8bit (stateful optimizer scaffold)
├── backends/triton_nf4.py  # fused NF4 kernel
└── utils/utils.py        # serialization + precision conversion
```

## License

MIT. Inspired by [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).
