"""Quickstart for the Quanta API.

Two paths:

1. Functional: quantize a tensor with ``quantize`` and recover it with
   ``dequantize``. The returned ``QuantState`` carries every byte of metadata.
2. nn modules: drop-in replacements for ``nn.Linear`` (``Linear8bitLt``,
   ``Linear4bit``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from Quanta import Linear4bit, Linear8bitLt, dequantize, quantize


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


if __name__ == "__main__":
    functional_demo()
    linear_demo()
    int8_linear_demo()
