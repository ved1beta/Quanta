"""
Fused 4-bit weight matmul (Triton).

Reads packed 4-bit weights, decodes inline, scales by per-block absmax, and
multiplies with an fp32/fp16 activation — all in a single kernel. Avoids
materializing the dequantized fp16 weight (90 MB for a 4096×11008 layer).

Works for both NF4 and FP4 because the on-disk format is identical; only the
codebook pointer changes.

Layout assumptions:
- weight is stored row-major (N, K) and packed as (N*K/2,) uint8 nibbles with
  pack_4bit semantics (low nibble = even flat index, high nibble = odd).
- absmax is per-flat-block of QBLK consecutive elements: shape (N*K/QBLK,).
  Requires K % QBLK == 0 so each absmax block stays within one weight row.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _4bit_matmul_kernel(
        x_ptr,          # *fp32, (M, K)
        w_packed_ptr,   # *u8, (N * K // 2,)
        absmax_ptr,     # *fp32, (N * K // QBLK,)
        code_ptr,       # *fp32, (16,)
        y_ptr,          # *fp32, (M, N)
        M, N, K,
        stride_xm, stride_xk,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        QBLK: tl.constexpr,
        HALF: tl.constexpr,
        USE_DOT: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

        m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_idx < M
        n_mask = n_idx < N

        pair_arange = tl.arange(0, HALF)
        K_BLOCKS = K // QBLK

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, K_BLOCKS):
            k_start = k_block * QBLK

            # x: (BLOCK_M, QBLK)
            even_k = k_start + pair_arange * 2
            odd_k = even_k + 1
            x_even_off = m_idx[:, None] * stride_xm + even_k[None, :] * stride_xk
            x_odd_off = m_idx[:, None] * stride_xm + odd_k[None, :] * stride_xk
            x_even = tl.load(x_ptr + x_even_off, mask=m_mask[:, None], other=0.0).to(tl.float32)
            x_odd = tl.load(x_ptr + x_odd_off, mask=m_mask[:, None], other=0.0).to(tl.float32)

            # packed: (BLOCK_N, HALF)
            w_off = n_idx[:, None] * (K // 2) + (k_start // 2 + pair_arange)[None, :]
            packed = tl.load(w_packed_ptr + w_off, mask=n_mask[:, None], other=0).to(tl.int32)
            low = packed & 0xF
            high = (packed >> 4) & 0xF

            # Codebook gather
            w_low = tl.load(code_ptr + low)
            w_high = tl.load(code_ptr + high)

            # Per-block absmax: one entry per (n, k_block)
            am_off = n_idx * K_BLOCKS + k_block
            am = tl.load(absmax_ptr + am_off, mask=n_mask, other=0.0)
            w_low = w_low * am[:, None]
            w_high = w_high * am[:, None]

            # Accumulate. For larger M we use tl.dot (tensor cores); for small
            # M (gemv) we fall through to elementwise — Triton requires
            # BLOCK_M >= 16 for tl.dot.
            if USE_DOT:
                # contraction over QBLK split into two HALF contractions
                acc += tl.dot(x_even.to(tl.float32), tl.trans(w_low.to(tl.float32)))
                acc += tl.dot(x_odd.to(tl.float32), tl.trans(w_high.to(tl.float32)))
            else:
                acc += tl.sum(x_even[:, None, :] * w_low[None, :, :], axis=2)
                acc += tl.sum(x_odd[:, None, :] * w_high[None, :, :], axis=2)

        y_off = m_idx[:, None] * stride_ym + n_idx[None, :] * stride_yn
        y_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(y_ptr + y_off, acc, mask=y_mask)


def triton_available() -> bool:
    return _TRITON_OK and torch.cuda.is_available()


def matmul_4bit_triton(
    x: torch.Tensor,           # (M, K) fp16/fp32 on CUDA
    w_packed: torch.Tensor,    # (N*K/2,) u8
    absmax: torch.Tensor,      # (N*K/QBLK,) fp
    code: torch.Tensor,        # (16,) fp
    out_features: int,
    in_features: int,
    block_size: int = 64,
) -> torch.Tensor:
    """Fused 4-bit weight matmul: y = x @ dequant(w).T.

    Requires:
      - CUDA and Triton.
      - in_features (K) is a multiple of block_size (QBLK).
      - x is contiguous 2-D.
    Returns (M, N) fp32 — caller casts as needed.
    """
    if not triton_available():
        raise RuntimeError("Triton / CUDA not available")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA")
    if x.dim() != 2:
        raise ValueError("x must be 2-D")
    if in_features % block_size != 0:
        raise ValueError("in_features must be a multiple of block_size")

    M, K = x.shape
    N = out_features
    assert K == in_features

    x_f32 = x.contiguous().to(torch.float32)
    code_f32 = code.contiguous().to(torch.float32)
    absmax_f32 = absmax.contiguous().to(torch.float32)
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = 16 if M >= 16 else max(1, 1 << (M - 1).bit_length())
    BLOCK_N = 64
    USE_DOT = M >= 16

    grid = ((N + BLOCK_N - 1) // BLOCK_N, (M + BLOCK_M - 1) // BLOCK_M)
    _4bit_matmul_kernel[grid](
        x_f32, w_packed, absmax_f32, code_f32, out,
        M, N, K,
        x_f32.stride(0), x_f32.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        QBLK=block_size, HALF=block_size // 2,
        USE_DOT=USE_DOT,
    )
    return out
