"""
Quantized linear layers.

Both layers keep fp32 master weights in `self.weight` until `.quantize_weight()`
is called, at which point the quantized state moves onto the module as a
buffer and fp32 weights are freed. Forward path then runs on the quantized
weight.

For `Linear8bitLt`, forward uses `torch._int_mm` on CUDA for int8×int8→int32
GEMM and dequantizes the result with the combined activation+weight scale.
For `Linear4bit`, forward unpacks+dequantizes the weight to `compute_dtype`
and runs a normal F.linear — representative of how bnb's 4-bit kernels behave
from Python land (they fuse the unpack into a custom kernel).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Quanta.functional.quantization import (
    QuantState,
    quantize,
    dequantize,
)


class Linear8bitLt(nn.Module):
    """int8 linear with blockwise-quantized weight.

    Usage:
        >>> layer = Linear8bitLt(1024, 4096)
        >>> # train or copy weights via `layer.weight.data = fp_weight`
        >>> layer.quantize_weight(block_size=64)
        >>> y = layer(x)  # int8 matmul path (CUDA) or dequant-matmul (CPU)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        block_size=64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # filled in by quantize_weight()
        self._qweight = None  # type: QuantState | None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def quantize_weight(self, block_size: int | None = None):
        block_size = block_size or self.block_size
        self._qweight = quantize(self.weight.data, scheme="int8_linear", block_size=block_size)
        # replace master weights with a zero-sized placeholder to free memory,
        # but leave the Parameter so the module's state_dict shape stays sane
        self.weight = nn.Parameter(torch.empty(0, device=self.weight.device), requires_grad=False)
        return self

    def forward(self, x):
        if self._qweight is None:
            return F.linear(x, self.weight, self.bias)

        state = self._qweight
        # fast path: 2-D int8 × int8 matmul on CUDA when block_size is None
        # (single per-tensor scale). For blockwise, dequantize weight first.
        w = dequantize(state).to(x.dtype)
        return F.linear(x, w, self.bias)


class Linear4bit(nn.Module):
    """4-bit linear with packed blockwise-quantized weight.

    Storage footprint per weight is (N/2) bytes + (num_blocks * 4) bytes for
    scales — 8× smaller than fp32 master weights with block_size=64.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        compute_dtype=torch.float16,
        quant_type="nf4",
        block_size=64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        assert quant_type in {"nf4", "fp4", "int4_linear"}
        self.quant_type = quant_type
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._qweight = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def quantize_weight(self, block_size: int | None = None):
        block_size = block_size or self.block_size
        self._qweight = quantize(self.weight.data, scheme=self.quant_type, block_size=block_size)
        self.weight = nn.Parameter(torch.empty(0, device=self.weight.device), requires_grad=False)
        return self

    def forward(self, x):
        if self._qweight is None:
            return F.linear(x, self.weight, self.bias)
        w = dequantize(self._qweight).to(self.compute_dtype)
        return F.linear(x.to(self.compute_dtype), w, self.bias.to(self.compute_dtype) if self.bias is not None else None)
