"""Quanta vs bitsandbytes: quantize/dequantize latency, round-trip MAE, Linear
forward latency, scaling with matrix size, and storage footprint.

    pip install -e ".[benchmark]"
    python benchmarks/bench_vs_bnb.py            # print tables
    python benchmarks/bench_vs_bnb.py --plot     # also write benchmarks/plots/*.png

int8 on the bitsandbytes side is ``quantize_blockwise`` (dynamic 8-bit map),
the closest analogue to Quanta's blockwise ``int8_linear``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from triton.testing import do_bench

import bitsandbytes as bnb
import bitsandbytes.functional as bnbf

from Quanta import Linear4bit, Linear8bitLt, dequantize, quantize

SCHEMES = ("int8", "nf4", "fp4")
_QUANTA_SCHEME = {"int8": "int8_linear", "nf4": "nf4", "fp4": "fp4"}
PLOT_DIR = Path(__file__).resolve().parent / "plots"


def _bnb_quant(x, scheme, bs):
    if scheme == "int8":
        return bnbf.quantize_blockwise(x, blocksize=bs)
    return bnbf.quantize_4bit(x, blocksize=bs, quant_type=scheme)


def _bnb_dequant(q, scheme):
    if scheme == "int8":
        return bnbf.dequantize_blockwise(*q)
    return bnbf.dequantize_4bit(*q)


def _bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def bench_tensor_ops(w, bs):
    rows = []
    for scheme in SCHEMES:
        qs = quantize(w, _QUANTA_SCHEME[scheme], block_size=bs)
        bq = _bnb_quant(w, scheme, bs)
        bstate = bq[1]
        rows.append({
            "scheme": scheme,
            "quant_ms": (
                do_bench(lambda: quantize(w, _QUANTA_SCHEME[scheme], block_size=bs)),
                do_bench(lambda: _bnb_quant(w, scheme, bs)),
            ),
            "dequant_ms": (
                do_bench(lambda: dequantize(qs)),
                do_bench(lambda: _bnb_dequant(bq, scheme)),
            ),
            "mae": (
                (dequantize(qs).float() - w).abs().mean().item(),
                (_bnb_dequant(bq, scheme).float() - w).abs().mean().item(),
            ),
            "mb": (
                _bytes(qs.qdata, qs.absmax) / 2**20,
                _bytes(bq[0], bstate.absmax) / 2**20,
            ),
        })
    return rows


def _bnb_linear(w, scheme, bs):
    out_f, in_f = w.shape
    if scheme == "int8":
        layer = bnb.nn.Linear8bitLt(in_f, out_f, bias=False, has_fp16_weights=False, threshold=0.0)
        layer.weight = bnb.nn.Int8Params(w.half().cpu(), requires_grad=False, has_fp16_weights=False)
    else:
        layer = bnb.nn.Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.float16,
                                  quant_type=scheme)
        layer.weight = bnb.nn.Params4bit(w.half().cpu(), requires_grad=False,
                                         quant_type=scheme, blocksize=bs)
    return layer.cuda()


def _quanta_linear(w, scheme, bs):
    out_f, in_f = w.shape
    if scheme == "int8":
        # Per-row scales enable the torch._int_mm path.
        layer = Linear8bitLt(in_f, out_f, bias=False)
        layer.weight.data.copy_(w.cpu())
        return layer.cuda().quantize_weight(block_size=in_f)
    layer = Linear4bit(in_f, out_f, bias=False, quant_type=scheme, compute_dtype=torch.float16)
    layer.weight.data.copy_(w.cpu())
    return layer.cuda().quantize_weight(block_size=bs)


@torch.no_grad()
def bench_linear(w, bs, batch_sizes):
    rows = []
    fp16 = nn.Linear(w.shape[1], w.shape[0], bias=False).cuda().half()
    fp16.weight.copy_(w)
    for scheme in SCHEMES:
        q, b = _quanta_linear(w, scheme, bs), _bnb_linear(w, scheme, bs)
        for M in batch_sizes:
            x = torch.randn(M, w.shape[1], device="cuda", dtype=torch.float16)
            rows.append({
                "scheme": scheme, "M": M,
                "ms": (do_bench(lambda: q(x)), do_bench(lambda: b(x)), do_bench(lambda: fp16(x))),
            })
    return rows


def bench_scaling(dims, bs):
    rows = []
    for d in dims:
        w = torch.randn(d, d, device="cuda")
        for scheme in SCHEMES:
            rows.append({
                "scheme": scheme, "dim": d,
                "ms": (
                    do_bench(lambda: quantize(w, _QUANTA_SCHEME[scheme], block_size=bs)),
                    do_bench(lambda: _bnb_quant(w, scheme, bs)),
                ),
            })
    return rows


def _print_table(title, header, rows):
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def _plot(tensor_rows, linear_rows, scaling_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(exist_ok=True)

    def bars(key, fname, ylabel):
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = range(len(tensor_rows))
        ax.bar([i - 0.2 for i in xs], [r[key][0] for r in tensor_rows], 0.4, label="Quanta")
        ax.bar([i + 0.2 for i in xs], [r[key][1] for r in tensor_rows], 0.4, label="bitsandbytes")
        ax.set_xticks(list(xs), [r["scheme"] for r in tensor_rows])
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / fname, dpi=120)
        plt.close(fig)

    bars("quant_ms", "01_quantize_latency.png", "quantize latency (ms)")
    bars("dequant_ms", "02_dequantize_latency.png", "dequantize latency (ms)")
    bars("mae", "03_mae.png", "round-trip MAE")
    bars("mb", "06_memory.png", "storage (MiB)")

    for rows, xkey, fname, xlabel in (
        (linear_rows, "M", "04_linear_forward.png", "batch rows (M)"),
        (scaling_rows, "dim", "05_scaling.png", "matrix dim (d × d)"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for scheme in SCHEMES:
            sub = [r for r in rows if r["scheme"] == scheme]
            xs = [r[xkey] for r in sub]
            ax.plot(xs, [r["ms"][0] for r in sub], "-o", label=f"Quanta {scheme}")
            ax.plot(xs, [r["ms"][1] for r in sub], "--o", label=f"bnb {scheme}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("latency (ms)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / fname, dpi=120)
        plt.close(fig)
    print(f"\nPlots written to {PLOT_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-features", type=int, default=11008)
    ap.add_argument("--in-features", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 128, 1024])
    ap.add_argument("--scaling-dims", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required (bitsandbytes kernels are CUDA-only).")
    torch.manual_seed(0)
    bs = args.block_size
    w = torch.randn(args.out_features, args.in_features, device="cuda") * 0.02
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | bnb {bnb.__version__} | "
          f"W = {args.out_features} × {args.in_features}, block_size={bs}")

    tensor_rows = bench_tensor_ops(w, bs)
    _print_table(
        "Tensor ops (Quanta / bnb)",
        ["scheme", "quantize ms", "dequantize ms", "MAE", "storage MiB"],
        [[r["scheme"]] + [f"{r[k][0]:.3g} / {r[k][1]:.3g}" for k in ("quant_ms", "dequant_ms", "mae", "mb")]
         for r in tensor_rows],
    )

    linear_rows = bench_linear(w, bs, args.batch_sizes)
    _print_table(
        "Linear forward, fp16 input (ms)",
        ["scheme", "M", "Quanta", "bnb", "fp16 nn.Linear"],
        [[r["scheme"], str(r["M"])] + [f"{v:.3f}" for v in r["ms"]] for r in linear_rows],
    )

    scaling_rows = bench_scaling(args.scaling_dims, bs)
    _print_table(
        "Quantize latency vs size (Quanta / bnb, ms)",
        ["scheme", "dim", "ms"],
        [[r["scheme"], str(r["dim"]), f"{r['ms'][0]:.3f} / {r['ms'][1]:.3f}"] for r in scaling_rows],
    )

    if args.plot:
        _plot(tensor_rows, linear_rows, scaling_rows)


if __name__ == "__main__":
    main()
