"""
Core quantization operations and functional interfaces.
"""

from .quantization import (
    # legacy tuple API (back-compat)
    quantize_8bit,
    quantize_4bit,
    dequantize_8bit,
    dequantize_4bit,
    quantize_8bit_linear,
    quantize_8bit_nf8,
    quantize_8bit_fp8,
    quantize_4bit_linear,
    quantize_4bit_nf4,
    quantize_4bit_fp4,
    # new unified API
    QuantState,
    quantize,
    dequantize,
    pack_4bit,
    unpack_4bit,
)
from .tensor_ops import quantized_matmul, quantized_matmul_int8

__all__ = [
    "quantize_8bit",
    "quantize_4bit",
    "dequantize_8bit",
    "dequantize_4bit",
    "quantize_8bit_linear",
    "quantize_8bit_nf8",
    "quantize_8bit_fp8",
    "quantize_4bit_linear",
    "quantize_4bit_nf4",
    "quantize_4bit_fp4",
    "QuantState",
    "quantize",
    "dequantize",
    "pack_4bit",
    "unpack_4bit",
    "quantized_matmul",
    "quantized_matmul_int8",
]
