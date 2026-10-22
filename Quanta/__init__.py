"""Quanta — efficient quantization primitives for PyTorch."""

import logging

from . import functional, nn
from .functional import (
    QuantState,
    VALID_SCHEMES,
    dequantize,
    quantize,
    quantized_matmul,
    quantized_matmul_int8,
)
from .nn import Linear4bit, Linear8bitLt, quantize_model

__version__ = "0.2.0"

__all__ = [
    "QuantState",
    "VALID_SCHEMES",
    "quantize",
    "dequantize",
    "quantized_matmul",
    "quantized_matmul_int8",
    "Linear8bitLt",
    "Linear4bit",
    "quantize_model",
    "functional",
    "nn",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
