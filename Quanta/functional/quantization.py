"""
Quantization and dequantization built around a single ``QuantState`` API.

``quantize(x, scheme=..., block_size=...)`` returns a ``QuantState`` carrying
all metadata needed to reverse the operation. ``dequantize(state)`` dispatches
on ``state.scheme``.

Notes:
- nf4/nf8/fp4 use ``torch.bucketize`` over codebook midpoints, not an
  ``|x - levels|`` distance matrix — O(N log K) and avoids OOMs at scale.
- 4-bit data is stored packed (two nibbles per byte).
- fp4 is normalized by per-tensor (or per-block) absmax before encoding.
- Codebook tensors are always allocated on the input's device and dtype.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


_COMPILE_ENABLED = (
    hasattr(torch, "compile")
    and os.environ.get("QUANTA_DISABLE_COMPILE", "0") != "1"
)
_TRITON_ENABLED = os.environ.get("QUANTA_DISABLE_TRITON", "0") != "1"


def _maybe_compile(fn):
    if not _COMPILE_ENABLED:
        return fn
    try:
        return torch.compile(fn, dynamic=True, fullgraph=False)
    except Exception:
        return fn


_NF4_LEVELS: Tuple[float, ...] = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)

# bnb-compatible FP4 codebook: 16 signed values addressed by the 4-bit code.
_FP4_CODE: Tuple[float, ...] = (
    0.0, 0.00520833, 0.66666667, 1.0,
    0.33333333, 0.5, 0.16666667, 0.25,
    -0.0, -0.00520833, -0.66666667, -1.0,
    -0.33333333, -0.5, -0.16666667, -0.25,
)


def _nf4_levels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_NF4_LEVELS, device=device, dtype=dtype)


def _nf8_levels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lvl = torch.linspace(-1.0, 1.0, 256, device=device, dtype=dtype)
    return torch.tanh(lvl * 2)


def _fp4_code(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_FP4_CODE, device=device, dtype=dtype)


def _codebook_encode(normalized: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    sorted_levels, order = torch.sort(levels)
    edges = (sorted_levels[:-1] + sorted_levels[1:]) / 2
    sorted_idx = torch.bucketize(normalized.contiguous(), edges)
    return order[sorted_idx]


def pack_4bit(q: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values per byte. Flattens input; pads odd-length with 0."""
    flat = q.reshape(-1).to(torch.uint8)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    pairs = flat.view(-1, 2)
    return (pairs[:, 0] | (pairs[:, 1] << 4)).to(torch.uint8)


def unpack_4bit(packed: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reverse of ``pack_4bit``."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.stack([low, high], dim=-1).reshape(-1)
    n = 1
    for s in shape:
        n *= s
    return out[:n].reshape(shape)


@dataclass
class QuantState:
    """Everything ``dequantize`` needs to reverse a ``quantize`` call."""
    qdata: torch.Tensor
    scheme: str
    shape: Optional[torch.Size] = None
    dtype: torch.dtype = torch.float32
    absmax: Optional[torch.Tensor] = None
    offset: Optional[torch.Tensor] = None
    block_size: Optional[int] = None
    packed: bool = False
    code: Optional[torch.Tensor] = None


def _pad_for_blocks(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int]:
    n = x.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([x.reshape(-1), torch.zeros(pad, device=x.device, dtype=x.dtype)])
    else:
        flat = x.reshape(-1)
    return flat.view(-1, block_size), pad


def _linear_encode_blockwise_eager(blocks: torch.Tensor, qmax: int):
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.round(blocks / absmax * qmax).clamp(-qmax, qmax).to(torch.int8)
    return q, absmax.squeeze(-1)


def _linear_encode_pertensor_eager(x: torch.Tensor, qmax: int):
    absmax = x.abs().amax().clamp(min=1e-8)
    q = torch.round(x / absmax * qmax).clamp(-qmax, qmax).to(torch.int8)
    return q, absmax


def _linear_decode_blockwise_eager(q: torch.Tensor, absmax: torch.Tensor, qmax: int):
    return q.float() * (absmax.unsqueeze(-1) / qmax)


def _linear_decode_pertensor_eager(q: torch.Tensor, absmax: torch.Tensor, qmax: int):
    return q.float() * (absmax / qmax)


def _codebook_decode_eager(idx_long: torch.Tensor, code: torch.Tensor, absmax_mul: torch.Tensor):
    return code[idx_long] * absmax_mul


_linear_encode_blockwise = _maybe_compile(_linear_encode_blockwise_eager)
_linear_encode_pertensor = _maybe_compile(_linear_encode_pertensor_eager)
_linear_decode_blockwise = _maybe_compile(_linear_decode_blockwise_eager)
_linear_decode_pertensor = _maybe_compile(_linear_decode_pertensor_eager)
_codebook_decode = _maybe_compile(_codebook_decode_eager)


VALID_SCHEMES = ("int8_linear", "int4_linear", "nf4", "nf8", "fp4", "fp8")


def quantize(
    x: torch.Tensor,
    scheme: str = "int8_linear",
    block_size: Optional[int] = None,
) -> QuantState:
    """Quantize ``x`` into a ``QuantState``.

    Args:
        x: Input tensor of any shape and float dtype.
        scheme: One of ``int8_linear``, ``int4_linear``, ``nf4``, ``nf8``,
            ``fp4``, ``fp8``.
        block_size: If set, scales are computed per ``block_size`` consecutive
            elements (after flattening). Robust to outliers; use 32–256.

    Returns:
        ``QuantState`` carrying the packed/unpacked qdata and all metadata.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x32 = x.float().contiguous()

    if scheme == "int8_linear":
        return _quant_linear(x32, bits=8, orig_shape=orig_shape, orig_dtype=orig_dtype, block_size=block_size)
    if scheme == "int4_linear":
        return _quant_linear(x32, bits=4, orig_shape=orig_shape, orig_dtype=orig_dtype, block_size=block_size)
    if scheme == "nf4":
        if (
            _TRITON_ENABLED
            and block_size is not None
            and x.is_cuda
            and block_size % 2 == 0
        ):
            try:
                from Quanta.backends.triton_nf4 import (
                    nf4_encode_blockwise_triton,
                    triton_available,
                )
                if triton_available():
                    packed, absmax, _ = nf4_encode_blockwise_triton(x32, block_size)
                    return QuantState(
                        qdata=packed,
                        scheme="nf4",
                        shape=orig_shape,
                        dtype=orig_dtype,
                        absmax=absmax,
                        block_size=block_size,
                        packed=True,
                        code=_nf4_levels(x.device, torch.float32),
                    )
            except Exception:
                pass
        return _quant_codebook(x32, levels=_nf4_levels(x.device, torch.float32), bits=4, scheme="nf4",
                               orig_shape=orig_shape, orig_dtype=orig_dtype, block_size=block_size)
    if scheme == "nf8":
        return _quant_codebook(x32, levels=_nf8_levels(x.device, torch.float32), bits=8, scheme="nf8",
                               orig_shape=orig_shape, orig_dtype=orig_dtype, block_size=block_size)
    if scheme == "fp4":
        return _quant_codebook(x32, levels=_fp4_code(x.device, torch.float32), bits=4, scheme="fp4",
                               orig_shape=orig_shape, orig_dtype=orig_dtype, block_size=block_size)
    if scheme == "fp8":
        return _quant_fp8(x32, orig_shape=orig_shape, orig_dtype=orig_dtype)
    raise ValueError(f"Unknown scheme: {scheme!r}. Expected one of {VALID_SCHEMES}.")


def dequantize(state: QuantState) -> torch.Tensor:
    """Reverse a ``quantize`` call. Returns a tensor of ``state.dtype``."""
    if state.scheme in ("int8_linear", "int4_linear"):
        return _dequant_linear(state)
    if state.scheme in ("nf4", "nf8", "fp4"):
        if (
            _TRITON_ENABLED
            and state.scheme == "nf4"
            and state.packed
            and state.block_size is not None
            and state.qdata.is_cuda
        ):
            try:
                from Quanta.backends.triton_nf4 import (
                    nf4_decode_blockwise_triton,
                    triton_available,
                )
                if triton_available():
                    out = nf4_decode_blockwise_triton(
                        state.qdata, state.absmax, state.code,
                        state.shape, state.block_size,
                    )
                    return out.to(state.dtype)
            except Exception:
                pass
        return _dequant_codebook(state)
    if state.scheme == "fp8":
        return _dequant_fp8(state)
    raise ValueError(f"Unknown scheme: {state.scheme!r}. Expected one of {VALID_SCHEMES}.")


def _quant_linear(x, *, bits, orig_shape, orig_dtype, block_size):
    qmax = (1 << (bits - 1)) - 1
    if block_size:
        blocks, pad = _pad_for_blocks(x, block_size)
        q, absmax = _linear_encode_blockwise(blocks, qmax)
        q = q.reshape(-1)
        if pad:
            q = q[:-pad]
        qdata = q.reshape(orig_shape)
        return QuantState(
            qdata=_maybe_pack(qdata, bits),
            scheme="int8_linear" if bits == 8 else "int4_linear",
            shape=orig_shape,
            dtype=orig_dtype,
            absmax=absmax,
            block_size=block_size,
            packed=(bits == 4),
        )
    q, absmax = _linear_encode_pertensor(x, qmax)
    qdata = q.reshape(orig_shape)
    return QuantState(
        qdata=_maybe_pack(qdata, bits),
        scheme="int8_linear" if bits == 8 else "int4_linear",
        shape=orig_shape,
        dtype=orig_dtype,
        absmax=absmax,
        block_size=None,
        packed=(bits == 4),
    )


def _maybe_pack(q_int8: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 4:
        u = (q_int8 + 7).to(torch.uint8)
        return pack_4bit(u)
    return q_int8


def _maybe_unpack(state: QuantState) -> torch.Tensor:
    if state.packed:
        u = unpack_4bit(state.qdata, state.shape)
        return u.to(torch.int8) - 7
    return state.qdata


def _dequant_linear(state: QuantState) -> torch.Tensor:
    qmax = 127 if state.scheme == "int8_linear" else 7
    q = _maybe_unpack(state)
    if state.block_size:
        flat = q.reshape(-1)
        pad = (-flat.numel()) % state.block_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
        blocks = flat.view(-1, state.block_size)
        out = _linear_decode_blockwise(blocks, state.absmax, qmax)
        out = out.reshape(-1)
        if pad:
            out = out[:out.numel() - pad]
        return out.reshape(state.shape).to(state.dtype)
    return _linear_decode_pertensor(q, state.absmax, qmax).reshape(state.shape).to(state.dtype)


def _quant_codebook(x, *, levels, bits, scheme, orig_shape, orig_dtype, block_size):
    if block_size:
        blocks, pad = _pad_for_blocks(x, block_size)
        absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (blocks / absmax).clamp(levels.min(), levels.max())
        idx = _codebook_encode(normalized, levels).to(torch.uint8)
        idx = idx.reshape(-1)
        if pad:
            idx = idx[:-pad]
        qdata = idx.reshape(orig_shape)
        return QuantState(
            qdata=_pack_codebook(qdata, bits),
            scheme=scheme,
            shape=orig_shape,
            dtype=orig_dtype,
            absmax=absmax.squeeze(-1),
            block_size=block_size,
            packed=(bits == 4),
            code=levels,
        )
    absmax = x.abs().amax().clamp(min=1e-8)
    normalized = (x / absmax).clamp(levels.min(), levels.max())
    idx = _codebook_encode(normalized, levels).to(torch.uint8).reshape(orig_shape)
    return QuantState(
        qdata=_pack_codebook(idx, bits),
        scheme=scheme,
        shape=orig_shape,
        dtype=orig_dtype,
        absmax=absmax,
        block_size=None,
        packed=(bits == 4),
        code=levels,
    )


def _pack_codebook(idx_u8: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 4:
        return pack_4bit(idx_u8)
    return idx_u8


def _dequant_codebook(state: QuantState) -> torch.Tensor:
    if state.packed:
        idx = unpack_4bit(state.qdata, state.shape).to(torch.long)
    else:
        idx = state.qdata.to(torch.long)
    if state.block_size:
        flat_idx = idx.reshape(-1)
        pad = (-flat_idx.numel()) % state.block_size
        if pad:
            flat_idx = torch.cat([flat_idx, torch.zeros(pad, device=flat_idx.device, dtype=flat_idx.dtype)])
        blocks = flat_idx.view(-1, state.block_size)
        out = _codebook_decode(blocks, state.code, state.absmax.unsqueeze(-1))
        out = out.reshape(-1)
        if pad:
            out = out[:out.numel() - pad]
        return out.reshape(state.shape).to(state.dtype)
    return _codebook_decode(idx, state.code, state.absmax).reshape(state.shape).to(state.dtype)


def _quant_fp8(x, *, orig_shape, orig_dtype):
    # 1 sign / 4 exponent / 3 mantissa, bias 7. Analytical encode keeps the
    # storage at one byte per value without a 256-entry LUT.
    signs = torch.sign(x)
    abs_values = torch.abs(x)
    log_values = torch.log2(abs_values + (abs_values == 0).float())
    exp_bias = 7
    exp_values = torch.clamp(torch.round(log_values + exp_bias), 0, 15)
    mantissa_values = torch.round((abs_values / (2 ** (exp_values - exp_bias))) * 8 - 8)
    mantissa_values = torch.clamp(mantissa_values, 0, 7)
    exp_values = exp_values.to(torch.uint8)
    mantissa_values = mantissa_values.to(torch.uint8)
    q = ((exp_values << 3) | mantissa_values).to(torch.uint8)
    signs_u = (signs < 0).to(torch.bool)
    q = torch.where(signs_u, q | 0x80, q)
    return QuantState(
        qdata=q.reshape(orig_shape),
        scheme="fp8",
        shape=orig_shape,
        dtype=orig_dtype,
        absmax=torch.tensor(float(exp_bias)),
    )


def _dequant_fp8(state: QuantState) -> torch.Tensor:
    q = state.qdata.to(torch.uint8)
    bias = float(state.absmax.item()) if isinstance(state.absmax, torch.Tensor) else float(state.absmax)
    signs = torch.where(q & 0x80 > 0, -1.0, 1.0)
    exp_values = (q >> 3) & 0x0F
    mantissa_values = q & 0x07
    values = (1.0 + mantissa_values.float() / 8.0) * (2.0 ** (exp_values.float() - bias))
    return (values * signs).to(state.dtype)
