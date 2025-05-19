"""
Bits-and-Byes: A library for efficient quantization and memory optimization in PyTorch.
"""

__version__ = "0.1.0"

from . import nn
from . import optim
from . import functional

# Import commonly used functions directly into the main namespace
from .nn import Linear8bitLt, Linear4bit
from .optim import Adam8bit

# Set up logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler()) 
