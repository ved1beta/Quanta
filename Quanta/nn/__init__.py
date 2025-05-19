"""
Neural network modules for quantized operations.
"""

from .linear import Linear8bitLt, Linear4bit

__all__ = [
    "Linear8bitLt",
    "Linear4bit",
] 
