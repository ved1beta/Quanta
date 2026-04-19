"""Quanta — efficient quantization and memory-optimized primitives for PyTorch."""

__version__ = "0.1.0"

from . import functional, nn, optim
from .functional import (
    QuantState,
    dequantize,
    pack_4bit,
    quantize,
    unpack_4bit,
)
from .nn import Linear4bit, Linear8bitLt
from .optim import Adam8bit

__all__ = [
    "QuantState",
    "quantize",
    "dequantize",
    "pack_4bit",
    "unpack_4bit",
    "Linear4bit",
    "Linear8bitLt",
    "Adam8bit",
    "functional",
    "nn",
    "optim",
]

import logging as _logging
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
