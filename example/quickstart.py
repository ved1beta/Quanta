"""Quickstart for the Quanta API.

Two paths:

1. Functional: quantize a tensor with ``quantize`` and recover it with
   ``dequantize``. The returned ``QuantState`` carries every byte of metadata.
2. nn modules: drop-in replacements for ``nn.Linear`` (``Linear8bitLt``,
   ``Linear4bit``).
3. Whole model: ``quantize_model(model, bits=4)`` swaps every ``nn.Linear``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from Quanta import Linear4bit, Linear8bitLt, dequantize, quantize, quantize_model


def functional_demo() -> None:
    x = torch.randn(2048, 2048)
    state = quantize(x, scheme="nf4", block_size=64)
    recovered = dequantize(state)
    mae = (x - recovered).abs().mean().item()
    print(f"NF4 blockwise — MAE: {mae:.4f}")
    print(f"  qdata bytes: {state.qdata.numel()} (vs fp32 bytes: {x.numel() * 4})")


def linear_demo() -> None:
    layer = Linear4bit(1024, 4096, quant_type="nf4")
    layer.quantize_weight(block_size=64)
    y = layer(torch.randn(8, 1024))
    print(f"Linear4bit out: {y.shape} dtype={y.dtype}")


def int8_linear_demo() -> None:
    layer = Linear8bitLt(512, 1024)
    layer.quantize_weight(block_size=64)
    y = layer(torch.randn(4, 512))
    print(f"Linear8bitLt out: {y.shape}")


def model_demo() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 512)
    )
    x = torch.randn(4, 512)
    ref = model(x)
    quantize_model(model, bits=4)
    out = model(x)
    print(f"quantize_model(bits=4): {model[0].__class__.__name__}, "
          f"rel err {(out - ref).abs().mean() / ref.abs().mean():.3f}")


if __name__ == "__main__":
    functional_demo()
    linear_demo()
    int8_linear_demo()
    model_demo()
