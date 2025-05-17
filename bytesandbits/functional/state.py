import torch
from enum import Enum
from typing import Dict, Optional, Tuple, Union
import json
import os

class QuantizationScheme(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"

class QuantizationType(Enum):
    LINEAR = "linear"
    NF4 = "nf4"
    NF8 = "nf8"
    FP4 = "fp4"
    FP8 = "fp8"

class QuantizationState:
    def __init__(self):
        self.tensor_params  = {}
        self.layers_params = {}
        self.global_config  = {
            "default_bits": 8,
            "default_scheme": QuantizationScheme.SYMMETRIC.value,
            "default_type": QuantizationType.LINEAR.value
        }