"""Quantized matmul ops over the ``QuantState`` API.

``quantized_matmul_int8`` uses ``torch._int_mm`` on CUDA for true int8 GEMM
(int8 × int8 → int32) when inputs are 2-D; falls back to dequant-and-matmul
otherwise.
"""

from __future__ import annotations

from typing import Optional

import torch

from .quantization import QuantState, quantize, dequantize


def int_mm_supported(m: int, k: int, n: int) -> bool:
    """Shape constraints of ``torch._int_mm`` on CUDA: M > 16, K and N multiples of 8."""
    return hasattr(torch, "_int_mm") and m > 16 and k % 8 == 0 and n % 8 == 0


def quantized_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scheme: str = "int8_linear",
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Quantize ``a`` and ``b``, dequant, then matmul (simulated quantization).

    Always correct, but compute happens at fp32. Prefer
    :func:`quantized_matmul_int8` for 2-D int8 × int8 on CUDA.
    """
    sa = quantize(a, scheme=scheme, block_size=block_size)
    sb = quantize(b, scheme=scheme, block_size=block_size)
    out = dequantize(sa) @ dequantize(sb)
    return out


def quantized_matmul_int8(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """True int8 GEMM via ``torch._int_mm`` where available.

    Args:
        a_int8, b_int8: 2-D int8 tensors in symmetric form (``scale = absmax/127``).
        a_scale, b_scale: per-tensor scalar scales such that ``fp = q * scale``.

    Returns:
        Float32 tensor of shape ``(M, N)``.
    """
    use_int_mm = (
        a_int8.is_cuda and b_int8.is_cuda
        and a_int8.dim() == 2 and b_int8.dim() == 2
        and a_int8.dtype == torch.int8 and b_int8.dtype == torch.int8
        and int_mm_supported(a_int8.shape[0], a_int8.shape[1], b_int8.shape[1])
    )
    if use_int_mm:
        acc_i32 = torch._int_mm(a_int8.contiguous(), b_int8.contiguous())
        return acc_i32.to(torch.float32) * (a_scale * b_scale)
    return (a_int8.float() * a_scale) @ (b_int8.float() * b_scale)
