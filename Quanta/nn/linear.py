"""
Quantized linear layers.

Both layers keep fp32 master weights in `self.weight` until `.quantize_weight()`
is called. After that, the quantized tensors (`qdata`, `absmax`, optional
`code`) live on the module as **buffers** so they move with `.to(device)` and
round-trip through `state_dict()` / `load_state_dict()`. Scheme metadata
(block_size, scheme name, packed flag, original shape/dtype) is carried via
`get_extra_state` / `set_extra_state`.

Forward path:

- `Linear8bitLt`: currently dequantizes the weight and calls `F.linear`. A
  true int8 × int8 GEMM via `torch._int_mm` requires reshaping our 1-D
  blockwise scales into a layout that's compatible with the 2-D reduction
  (scale must factor across the reduction dim). That's a separate project;
  see `Quanta.functional.tensor_ops.quantized_matmul_int8` for the
  per-tensor version.
- `Linear4bit`: unpacks + dequantizes the weight to `compute_dtype` and runs
  a normal F.linear — representative of how bnb's 4-bit kernels look from
  Python land (they fuse the unpack into a custom kernel).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Quanta.functional.quantization import (
    QuantState,
    quantize,
    dequantize,
)


_VALID_4BIT_SCHEMES = {"nf4", "fp4", "int4_linear"}


_Q_TENSOR_NAMES = ("qdata", "absmax", "code")
_Q_META_KEYS = ("q_scheme", "q_block_size", "q_packed", "q_shape", "q_dtype")


class _QuantizedLinearMixin:
    """Shared state_dict persistence + qstate reconstruction helpers.

    Quantized tensors (`qdata`, `absmax`, optional `code`) are persisted via
    a custom `_save_to_state_dict` / `_load_from_state_dict` pair so they
    round-trip cleanly even when loading into a fresh module that hasn't yet
    registered them as buffers. Scheme metadata (block_size, scheme name,
    packed flag, shape, dtype) rides along via `get_extra_state`.
    """

    def _store_qstate(self, state: QuantState):
        """Attach `state`'s tensors + metadata to the module."""
        self.qdata = state.qdata
        self.absmax = state.absmax
        self.code = state.code  # may be None for linear schemes

        self._q_scheme = state.scheme
        self._q_block_size = state.block_size
        self._q_packed = state.packed
        self._q_shape = tuple(state.shape) if state.shape is not None else None
        self._q_dtype = state.dtype

        # Free the fp32 master weight. Leave an empty Parameter so `self.weight`
        # still exists (consumers may introspect it) but occupies no memory.
        self.weight = nn.Parameter(
            torch.empty(0, device=state.qdata.device, dtype=self._q_dtype),
            requires_grad=False,
        )

    def _qstate(self) -> Optional[QuantState]:
        """Rebuild a `QuantState` from the stored tensors, or None if the
        module hasn't been quantized."""
        if not getattr(self, "_q_scheme", None):
            return None
        return QuantState(
            qdata=self.qdata,
            scheme=self._q_scheme,
            shape=torch.Size(self._q_shape) if self._q_shape is not None else None,
            dtype=self._q_dtype,
            absmax=self.absmax,
            block_size=self._q_block_size,
            packed=self._q_packed,
            code=getattr(self, "code", None),
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if getattr(self, "_q_scheme", None) is None:
            return
        for name in _Q_TENSOR_NAMES:
            t = getattr(self, name, None)
            if t is None:
                continue
            destination[prefix + name] = t if keep_vars else t.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Pop quantized tensors before super() runs so it doesn't treat them as
        # unexpected. Assign them directly onto the module.
        for name in _Q_TENSOR_NAMES:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, state_dict.pop(key))

        # If the checkpoint holds a zero-sized `weight`, shrink our Parameter to
        # match so super's loader doesn't raise a size-mismatch error.
        wkey = prefix + "weight"
        if (
            wkey in state_dict
            and state_dict[wkey].numel() == 0
            and self.weight.numel() > 0
        ):
            self.weight = nn.Parameter(
                torch.empty(0, device=self.weight.device, dtype=self.weight.dtype),
                requires_grad=False,
            )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def get_extra_state(self):
        if not getattr(self, "_q_scheme", None):
            return {}
        return {k: getattr(self, "_" + k) for k in _Q_META_KEYS}

    def set_extra_state(self, state):
        if not state:
            self._q_scheme = None
            return
        self._q_scheme = state.get("q_scheme")
        self._q_block_size = state.get("q_block_size")
        self._q_packed = state.get("q_packed", False)
        self._q_shape = state.get("q_shape")
        self._q_dtype = state.get("q_dtype", torch.float32)


class Linear8bitLt(_QuantizedLinearMixin, nn.Module):
    """int8 linear with blockwise-quantized weight.

    Usage:
        >>> layer = Linear8bitLt(1024, 4096)
        >>> layer.weight.data = fp32_weight              # or train
        >>> layer.quantize_weight(block_size=64)
        >>> y = layer(x)                                 # forward dequantizes weight
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

        self._q_scheme = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def quantize_weight(self, block_size: Optional[int] = None):
        block_size = block_size or self.block_size
        state = quantize(self.weight.data, scheme="int8_linear", block_size=block_size)
        self._store_qstate(state)
        return self

    def forward(self, x):
        state = self._qstate()
        if state is None:
            return F.linear(x, self.weight, self.bias)
        w = dequantize(state).to(x.dtype)
        return F.linear(x, w, self.bias)


class Linear4bit(_QuantizedLinearMixin, nn.Module):
    """4-bit linear with packed blockwise-quantized weight.

    Storage footprint per weight is (N/2) bytes + (num_blocks * 4) bytes for
    scales — ~8× smaller than fp32 master weights with block_size=64.

    `compute_dtype` is the dtype used to materialize the dequantized weight
    and compute the matmul in. Cast of `bias` into `compute_dtype` is done
    once at `quantize_weight()` time and cached as a buffer so forward
    doesn't reallocate on every call.
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
        if quant_type not in _VALID_4BIT_SCHEMES:
            raise ValueError(
                f"Invalid quant_type={quant_type!r}. "
                f"Expected one of {sorted(_VALID_4BIT_SCHEMES)}."
            )
        self.quant_type = quant_type
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Populated by quantize_weight() when bias is present; avoids
        # per-forward bias.to(compute_dtype) allocations.
        self.register_buffer("_bias_compute", None, persistent=False)

        self._q_scheme = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def quantize_weight(self, block_size: Optional[int] = None):
        block_size = block_size or self.block_size
        state = quantize(self.weight.data, scheme=self.quant_type, block_size=block_size)
        self._store_qstate(state)
        if self.bias is not None:
            self._bias_compute = self.bias.detach().to(self.compute_dtype)
        return self

    def forward(self, x):
        state = self._qstate()
        if state is None:
            return F.linear(x, self.weight, self.bias)
        w = dequantize(state).to(self.compute_dtype)
        bias = self._bias_compute if self._bias_compute is not None else self.bias
        return F.linear(x.to(self.compute_dtype), w, bias)
