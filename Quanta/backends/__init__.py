"""Device-specific accelerated kernels.

Currently exposes a Triton NF4 blockwise encode/decode path; the main
``quantize`` / ``dequantize`` dispatch picks it up automatically on CUDA.
"""

from .triton_nf4 import (
    nf4_decode_blockwise_triton,
    nf4_encode_blockwise_triton,
    triton_available,
)

__all__ = [
    "nf4_encode_blockwise_triton",
    "nf4_decode_blockwise_triton",
    "triton_available",
]
