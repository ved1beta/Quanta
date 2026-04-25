"""
Triton fp4 blockwise encoder/decoder.

Mirrors ``triton_nf4`` but the FP4 codebook is *unsorted*, so the encoder
bucket-search produces an index in *sorted* order which we then remap back
to the original codebook index via a small lookup.

Matches the numerics of ``quantization._quant_codebook`` + ``pack_4bit`` for
fp4 with blockwise scales bit-for-bit.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


# bnb-compatible FP4 codebook (16 signed values).
_FP4_CODE = (
    0.0, 0.00520833, 0.66666667, 1.0,
    0.33333333, 0.5, 0.16666667, 0.25,
    -0.0, -0.00520833, -0.66666667, -1.0,
    -0.33333333, -0.5, -0.16666667, -0.25,
)


def _compute_edges_and_order():
    """Sort the codebook on the host once; return (edges_tuple, order_tuple)."""
    levels = torch.tensor(_FP4_CODE, dtype=torch.float64)
    sorted_levels, order = torch.sort(levels)
    edges = (sorted_levels[:-1] + sorted_levels[1:]) / 2
    return tuple(float(e) for e in edges.tolist()), tuple(int(o) for o in order.tolist())


_FP4_EDGES, _FP4_ORDER = _compute_edges_and_order()


if _TRITON_OK:

    @triton.jit
    def _fp4_remap(sorted_idx,
                   O0, O1, O2, O3, O4, O5, O6, O7,
                   O8, O9, O10, O11, O12, O13, O14, O15):
        # 16-way select: order[sorted_idx]. Cheaper than an indirect load for
        # a 16-entry LUT and avoids a memory dependency.
        v = tl.where(sorted_idx == 0, O0, 0)
        v = tl.where(sorted_idx == 1, O1, v)
        v = tl.where(sorted_idx == 2, O2, v)
        v = tl.where(sorted_idx == 3, O3, v)
        v = tl.where(sorted_idx == 4, O4, v)
        v = tl.where(sorted_idx == 5, O5, v)
        v = tl.where(sorted_idx == 6, O6, v)
        v = tl.where(sorted_idx == 7, O7, v)
        v = tl.where(sorted_idx == 8, O8, v)
        v = tl.where(sorted_idx == 9, O9, v)
        v = tl.where(sorted_idx == 10, O10, v)
        v = tl.where(sorted_idx == 11, O11, v)
        v = tl.where(sorted_idx == 12, O12, v)
        v = tl.where(sorted_idx == 13, O13, v)
        v = tl.where(sorted_idx == 14, O14, v)
        v = tl.where(sorted_idx == 15, O15, v)
        return v

    @triton.jit
    def _fp4_encode_nibbles(xn, E0, E1, E2, E3, E4, E5, E6, E7,
                            E8, E9, E10, E11, E12, E13, E14):
        # Strict > to match torch.bucketize (right=False).
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
    def _fp4_blockwise_kernel(
        x_ptr,          # *fp32, [n_padded]
        absmax_ptr,     # *fp32, [num_blocks]
        packed_ptr,     # *uint8, [n_padded / 2]
        n_padded,
        BLOCK: tl.constexpr,
        HALF: tl.constexpr,
        E0: tl.constexpr, E1: tl.constexpr, E2: tl.constexpr, E3: tl.constexpr,
        E4: tl.constexpr, E5: tl.constexpr, E6: tl.constexpr, E7: tl.constexpr,
        E8: tl.constexpr, E9: tl.constexpr, E10: tl.constexpr, E11: tl.constexpr,
        E12: tl.constexpr, E13: tl.constexpr, E14: tl.constexpr,
        O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr, O3: tl.constexpr,
        O4: tl.constexpr, O5: tl.constexpr, O6: tl.constexpr, O7: tl.constexpr,
        O8: tl.constexpr, O9: tl.constexpr, O10: tl.constexpr, O11: tl.constexpr,
        O12: tl.constexpr, O13: tl.constexpr, O14: tl.constexpr, O15: tl.constexpr,
        # Sorted level closest to 0 from each side, used to clamp the input
        # the same way `_quant_codebook` does (clamp to [min, max] of levels).
        LMIN: tl.constexpr, LMAX: tl.constexpr,
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

        # Match the eager path's clamp(levels.min(), levels.max()).
        xn_even = tl.minimum(tl.maximum(x_even / absmax, LMIN), LMAX)
        xn_odd = tl.minimum(tl.maximum(x_odd / absmax, LMIN), LMAX)

        sorted_low = _fp4_encode_nibbles(xn_even,
                                         E0, E1, E2, E3, E4, E5, E6, E7,
                                         E8, E9, E10, E11, E12, E13, E14)
        sorted_high = _fp4_encode_nibbles(xn_odd,
                                          E0, E1, E2, E3, E4, E5, E6, E7,
                                          E8, E9, E10, E11, E12, E13, E14)
        low = _fp4_remap(sorted_low,
                         O0, O1, O2, O3, O4, O5, O6, O7,
                         O8, O9, O10, O11, O12, O13, O14, O15)
        high = _fp4_remap(sorted_high,
                          O0, O1, O2, O3, O4, O5, O6, O7,
                          O8, O9, O10, O11, O12, O13, O14, O15)
        high = tl.where(odd_mask, high, 0)

        packed = (low & 0xF) | ((high & 0xF) << 4)
        out_off = pid * HALF + pair_arange
        out_mask = out_off < (n_padded // 2)
        tl.store(packed_ptr + out_off, packed.to(tl.uint8), mask=out_mask)

    @triton.jit
    def _fp4_dequant_blockwise_kernel(
        packed_ptr,
        absmax_ptr,
        code_ptr,
        out_ptr,
        n_elements,
        BLOCK: tl.constexpr,
        HALF: tl.constexpr,
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


def fp4_encode_blockwise_triton(x: torch.Tensor, block_size: int = 64):
    """Blockwise FP4 encode using the Triton kernel.

    Returns (packed_u8, absmax_fp32, n_elements_original).
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
    lmin = min(_FP4_CODE)
    lmax = max(_FP4_CODE)
    _fp4_blockwise_kernel[grid](
        x, absmax, packed,
        n_padded,
        BLOCK=block_size,
        HALF=block_size // 2,
        E0=_FP4_EDGES[0], E1=_FP4_EDGES[1], E2=_FP4_EDGES[2], E3=_FP4_EDGES[3],
        E4=_FP4_EDGES[4], E5=_FP4_EDGES[5], E6=_FP4_EDGES[6], E7=_FP4_EDGES[7],
        E8=_FP4_EDGES[8], E9=_FP4_EDGES[9], E10=_FP4_EDGES[10], E11=_FP4_EDGES[11],
        E12=_FP4_EDGES[12], E13=_FP4_EDGES[13], E14=_FP4_EDGES[14],
        O0=_FP4_ORDER[0], O1=_FP4_ORDER[1], O2=_FP4_ORDER[2], O3=_FP4_ORDER[3],
        O4=_FP4_ORDER[4], O5=_FP4_ORDER[5], O6=_FP4_ORDER[6], O7=_FP4_ORDER[7],
        O8=_FP4_ORDER[8], O9=_FP4_ORDER[9], O10=_FP4_ORDER[10], O11=_FP4_ORDER[11],
        O12=_FP4_ORDER[12], O13=_FP4_ORDER[13], O14=_FP4_ORDER[14], O15=_FP4_ORDER[15],
        LMIN=lmin, LMAX=lmax,
    )
    return packed, absmax, n


def fp4_decode_blockwise_triton(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    shape: torch.Size,
    block_size: int = 64,
) -> torch.Tensor:
    """Blockwise FP4 dequant via the fused Triton kernel."""
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
    _fp4_dequant_blockwise_kernel[grid](
        packed, absmax, code, out,
        n_elements,
        BLOCK=block_size,
        HALF=block_size // 2,
    )
    return out.view(shape)
