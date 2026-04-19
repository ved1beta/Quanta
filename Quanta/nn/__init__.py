"""Quantized neural-network modules."""

from .linear import Linear4bit, Linear8bitLt

__all__ = ["Linear8bitLt", "Linear4bit"]
