"""Element-wise and matmul ops over quantized tensors.

Routes through the canonical `quantize_8bit_linear` / `dequantize_8bit` pair so
there's one set of sign/offset conventions. ``quantized_matmul_int8`` uses
``torch._int_mm`` for true int8 × int8 → int32 GEMM on CUDA, falling back to
dequant-and-matmul elsewhere.
"""

import torch

from .base import BaseQuantizer
from .quantization import (
    dequantize_8bit as _dequantize_8bit_legacy,
    quantize_8bit_linear,
)


class Quantizer(BaseQuantizer):
    """BaseQuantizer façade with 8-bit / 4-bit convenience entry points."""

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


def _quant(tensor):
    return quantize_8bit_linear(tensor)


def _dequant(q, scale, offset):
    # Use the linear-quant legacy dequantizer (this module shadows the name above).
    return _dequantize_8bit_legacy(q, scale, offset, quant_type="linear")


def quantize_add(a, a_scale, a_zero_point, b, b_scale, b_zero_point):
    return _quant(_dequant(a, a_scale, a_zero_point) + _dequant(b, b_scale, b_zero_point))


def quantized_matmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point):
    """Dequant-matmul-requant fallback. Always correct; fp32 compute.

    Prefer ``quantized_matmul_int8`` for 2-D int8 × int8 on CUDA.
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
    """Int8 × int8 → fp32 GEMM via ``torch._int_mm`` where available.

    ``a_scale``/``b_scale`` are per-tensor absmax/127 so ``fp = q * scale``.
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
    return (a_int8.float() * a_scale) @ (b_int8.float() * b_scale)


def quantized_mul(a, a_scale, a_zero_point, b, b_scale, b_zero_point):
    return _quant(_dequant(a, a_scale, a_zero_point) * _dequant(b, b_scale, b_zero_point))


def quantized_relu(x, scale, zero_point):
    return _quant(torch.relu(_dequant(x, scale, zero_point)))
