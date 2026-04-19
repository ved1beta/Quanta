"""Triton NF4 blockwise encode + decode.

Fuses per-block absmax reduction, normalization, nearest-level search, and
4-bit packing into one kernel. Bit-exact with the eager path in
``quantization._quant_codebook`` + ``pack_4bit``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


# Midpoints between adjacent NF4 levels — bucketize edges. 15 edges for 16 levels.
_NF4_EDGES = (
    -0.8480964004993439,
    -0.6106329262256622,
    -0.4599952697753906,
    -0.33967943489551544,
    -0.23460740596055984,
    -0.13791173323988914,
    -0.045525018125772476,
    0.03979014977812767,
    0.1202552504837513,
    0.2035212516784668,
    0.2920137718319893,
    0.3893125355243683,
    0.5016634166240692,
    0.6427869200706482,
    0.8614784181118011,
)


if _TRITON_OK:

    @triton.jit
    def _encode_nibbles(xn, E0, E1, E2, E3, E4, E5, E6, E7,
                        E8, E9, E10, E11, E12, E13, E14):
        # Strict > matches torch.bucketize(right=False): v == edge[i] → bucket i.
        idx = (xn > E0).to(tl.int32)
        idx += (xn > E1).to(tl.int32)
        idx += (xn > E2).to(tl.int32)
        idx += (xn > E3).to(tl.int32)
        idx += (xn > E4).to(tl.int32)
        idx += (xn > E5).to(tl.int32)
        idx += (xn > E6).to(tl.int32)
        idx += (xn > E7).to(tl.int32)
        idx += (xn > E8).to(tl.int32)
        idx += (xn > E9).to(tl.int32)
        idx += (xn > E10).to(tl.int32)
        idx += (xn > E11).to(tl.int32)
        idx += (xn > E12).to(tl.int32)
        idx += (xn > E13).to(tl.int32)
        idx += (xn > E14).to(tl.int32)
        return idx

    @triton.jit
    def _nf4_blockwise_kernel(
        x_ptr, absmax_ptr, packed_ptr, n_padded,
        BLOCK: tl.constexpr, HALF: tl.constexpr,
        E0: tl.constexpr, E1: tl.constexpr, E2: tl.constexpr, E3: tl.constexpr,
        E4: tl.constexpr, E5: tl.constexpr, E6: tl.constexpr, E7: tl.constexpr,
        E8: tl.constexpr, E9: tl.constexpr, E10: tl.constexpr, E11: tl.constexpr,
        E12: tl.constexpr, E13: tl.constexpr, E14: tl.constexpr,
    ):
        pid = tl.program_id(0)

        full_off = pid * BLOCK + tl.arange(0, BLOCK)
        full_mask = full_off < n_padded
        x_full = tl.load(x_ptr + full_off, mask=full_mask, other=0.0)
        absmax = tl.maximum(tl.max(tl.abs(x_full), axis=0), 1e-8)
        tl.store(absmax_ptr + pid, absmax)

        pair_arange = tl.arange(0, HALF)
        even_off = pid * BLOCK + pair_arange * 2
        odd_off = even_off + 1
        even_mask = even_off < n_padded
        odd_mask = odd_off < n_padded

        x_even = tl.load(x_ptr + even_off, mask=even_mask, other=0.0)
        x_odd = tl.load(x_ptr + odd_off, mask=odd_mask, other=0.0)

        xn_even = x_even / absmax
        xn_odd = x_odd / absmax

        low = _encode_nibbles(xn_even,
                              E0, E1, E2, E3, E4, E5, E6, E7,
                              E8, E9, E10, E11, E12, E13, E14)
        high = _encode_nibbles(xn_odd,
                               E0, E1, E2, E3, E4, E5, E6, E7,
                               E8, E9, E10, E11, E12, E13, E14)
        high = tl.where(odd_mask, high, 0)

        packed = (low & 0xF) | ((high & 0xF) << 4)
        out_off = pid * HALF + pair_arange
        out_mask = out_off < (n_padded // 2)
        tl.store(packed_ptr + out_off, packed.to(tl.uint8), mask=out_mask)


if _TRITON_OK:

    @triton.jit
    def _nf4_dequant_blockwise_kernel(
        packed_ptr, absmax_ptr, code_ptr, out_ptr, n_elements,
        BLOCK: tl.constexpr, HALF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        absmax = tl.load(absmax_ptr + pid)

        pair_arange = tl.arange(0, HALF)
        packed_off = pid * HALF + pair_arange
        even_off = pid * BLOCK + pair_arange * 2
        odd_off = even_off + 1

        even_mask = even_off < n_elements
        odd_mask = odd_off < n_elements

        packed = tl.load(packed_ptr + packed_off, mask=even_mask, other=0)
        low = (packed & 0xF).to(tl.int32)
        high = ((packed >> 4) & 0xF).to(tl.int32)

        low_val = tl.load(code_ptr + low) * absmax
        high_val = tl.load(code_ptr + high) * absmax

        tl.store(out_ptr + even_off, low_val, mask=even_mask)
        tl.store(out_ptr + odd_off, high_val, mask=odd_mask)


def triton_available() -> bool:
    return _TRITON_OK and torch.cuda.is_available()


def nf4_encode_blockwise_triton(x: torch.Tensor, block_size: int = 64):
    """Blockwise NF4 encode. Returns ``(packed_u8, absmax_fp32, n_elements)``.

    Requires CUDA and even ``block_size``.
    """
    if not triton_available():
        raise RuntimeError("Triton / CUDA not available")
    if not x.is_cuda:
        raise ValueError("Input must be on CUDA")
    if block_size % 2 != 0:
        raise ValueError("block_size must be even for 4-bit packing")

    x = x.contiguous().view(-1).to(torch.float32)
    n = x.numel()
    pad = (-n) % block_size
    if pad:
        x = torch.cat([x, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    n_padded = x.numel()
    num_blocks = n_padded // block_size

    absmax = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
    packed = torch.empty(n_padded // 2, device=x.device, dtype=torch.uint8)

    grid = (num_blocks,)
    _nf4_blockwise_kernel[grid](
        x, absmax, packed,
        n_padded,
        BLOCK=block_size,
        HALF=block_size // 2,
        E0=_NF4_EDGES[0], E1=_NF4_EDGES[1], E2=_NF4_EDGES[2], E3=_NF4_EDGES[3],
        E4=_NF4_EDGES[4], E5=_NF4_EDGES[5], E6=_NF4_EDGES[6], E7=_NF4_EDGES[7],
        E8=_NF4_EDGES[8], E9=_NF4_EDGES[9], E10=_NF4_EDGES[10], E11=_NF4_EDGES[11],
        E12=_NF4_EDGES[12], E13=_NF4_EDGES[13], E14=_NF4_EDGES[14],
    )
    return packed, absmax, n


def nf4_decode_blockwise_triton(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    shape: torch.Size,
    block_size: int = 64,
) -> torch.Tensor:
    """Blockwise NF4 dequant: unpack, codebook gather, scale — in one launch."""
    if not triton_available():
        raise RuntimeError("Triton / CUDA not available")
    if not packed.is_cuda:
        raise ValueError("Input must be on CUDA")
    if block_size % 2 != 0:
        raise ValueError("block_size must be even")

    code = code.to(torch.float32).contiguous()
    n_elements = 1
    for s in shape:
        n_elements *= s
    num_blocks = (n_elements + block_size - 1) // block_size

    out = torch.empty(n_elements, device=packed.device, dtype=torch.float32)
    grid = (num_blocks,)
    _nf4_dequant_blockwise_kernel[grid](
        packed, absmax, code, out,
        n_elements,
        BLOCK=block_size,
        HALF=block_size // 2,
    )
    return out.view(shape)
