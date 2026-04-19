"""
Element-wise and matmul ops over quantized tensors.

All ops route through the canonical `quantize_8bit_linear` / `dequantize_8bit`
pair so there's one code path, one set of conventions, and no silent
wrap-around from casting 0-255 values through `torch.int8`.

`quantized_matmul_int8` uses `torch._int_mm` on CUDA for true int8 GEMM
(int8 × int8 → int32) when inputs are 2-D; falls back to dequant-and-matmul
otherwise.
"""

import torch

from .base import BaseQuantizer
from .quantization import (
    quantize_8bit_linear,
    dequantize_8bit as _dequantize_8bit_legacy,
)


class Quantizer(BaseQuantizer):
    """BaseQuantizer façade with 8-bit and 4-bit convenience entry points."""

    def quantize_8bit(self, tensor, per_channel=False, symmetric=True):
        self.num_bits = 8
        self.symmetric = symmetric
        return self.quantize(tensor, per_channel)

    def quantize_4bit(self, tensor, per_channel=False, symmetric=True):
        self.num_bits = 4
        self.symmetric = symmetric
        return self.quantize(tensor, per_channel)


_quantizer = Quantizer()


def quantize_8bit(tensor, per_channel=False, symmetric=True):
    return _quantizer.quantize_8bit(tensor, per_channel, symmetric)


def quantize_4bit(tensor, per_channel=False, symmetric=True):
    return _quantizer.quantize_4bit(tensor, per_channel, symmetric)


def dequantize_8bit(q_tensor, scale, zero_point):
    return _quantizer.dequantize(q_tensor, scale, zero_point)


def dequantize_4bit(q_tensor, scale, zero_point):
    return _quantizer.dequantize(q_tensor, scale, zero_point)


# -----------------------------------------------------------------------------
# Single source of truth for linear 8-bit quant inside this module. Delegating
# to `quantization.quantize_8bit_linear` keeps one implementation.
# -----------------------------------------------------------------------------

def _quant(tensor):
    return quantize_8bit_linear(tensor)


def _dequant(q, scale, offset):
    # Uses the legacy functional dequantizer (which accepts quant_type).
    # We re-import it under an alias because this module shadows
    # `dequantize_8bit` with a BaseQuantizer-backed wrapper at line 48.
    return _dequantize_8bit_legacy(q, scale, offset, quant_type="linear")


def quantize_add(a, a_scale, a_zero_point, b, b_scale, b_zero_point):
    a_f = _dequant(a, a_scale, a_zero_point)
    b_f = _dequant(b, b_scale, b_zero_point)
    return _quant(a_f + b_f)


def quantized_matmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point):
    """Dequant-matmul-requant fallback. Always correct, but fp32 compute.

    Prefer `quantized_matmul_int8` for 2-D int8 × int8 on CUDA.
    """
    a_f = _dequant(a, a_scale, a_zero_point)
    b_f = _dequant(b, b_scale, b_zero_point)
    return _quant(torch.matmul(a_f, b_f))


def quantized_matmul_int8(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """True int8 GEMM using torch._int_mm where available.

    Args:
        a_int8, b_int8: 2-D int8 tensors in symmetric form (scale = absmax/127).
        a_scale, b_scale: absmax/127 scales so that `fp = q * scale`.

    Returns a float32 tensor of shape (M, N).
    """
    use_int_mm = (
        a_int8.is_cuda and b_int8.is_cuda
        and a_int8.dim() == 2 and b_int8.dim() == 2
        and a_int8.dtype == torch.int8 and b_int8.dtype == torch.int8
        and hasattr(torch, "_int_mm")
    )
    if use_int_mm:
        acc_i32 = torch._int_mm(a_int8.contiguous(), b_int8.contiguous())
        return acc_i32.to(torch.float32) * (a_scale * b_scale)
    # CPU / odd-dtype fallback
    return (a_int8.float() * a_scale) @ (b_int8.float() * b_scale)


def quantized_mul(a, a_scale, a_zero_point, b, b_scale, b_zero_point):
    a_f = _dequant(a, a_scale, a_zero_point)
    b_f = _dequant(b, b_scale, b_zero_point)
    return _quant(a_f * b_f)


def quantized_relu(x, scale, zero_point):
    return _quant(torch.relu(_dequant(x, scale, zero_point)))
