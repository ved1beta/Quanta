"""Quantized neural-network modules."""

from .linear import Linear4bit, Linear8bitLt
from .model import quantize_model

__all__ = ["Linear8bitLt", "Linear4bit", "quantize_model"]
