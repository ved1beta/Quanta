"""Quantized ``nn.Linear`` drop-ins.

Both layers hold an fp32 master weight in ``self.weight`` until
``.quantize_weight()`` is called. After that the quantized tensors
(``qdata``, ``absmax``, optional ``code``) live on the module as buffers, so
they move with ``.to(device)`` and round-trip through ``state_dict()`` /
``load_state_dict()``. Scheme metadata rides along via
``get_extra_state`` / ``set_extra_state``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Quanta.functional.quantization import (
    QuantState,
    dequantize,
    quantize,
)


_VALID_4BIT_SCHEMES = {"nf4", "fp4", "int4_linear"}

_Q_TENSOR_NAMES = ("qdata", "absmax", "code")
_Q_META_KEYS = ("q_scheme", "q_block_size", "q_packed", "q_shape", "q_dtype")


class _QuantizedLinearMixin:
    """Shared state_dict persistence + ``QuantState`` reconstruction helpers."""

    def _store_qstate(self, state: QuantState) -> None:
        self.qdata = state.qdata
        self.absmax = state.absmax
        self.code = state.code

        self._q_scheme = state.scheme
        self._q_block_size = state.block_size
        self._q_packed = state.packed
        self._q_shape = tuple(state.shape) if state.shape is not None else None
        self._q_dtype = state.dtype

        self.weight = nn.Parameter(
            torch.empty(0, device=state.qdata.device, dtype=self._q_dtype),
            requires_grad=False,
        )

    def _qstate(self) -> Optional[QuantState]:
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
        for name in _Q_TENSOR_NAMES:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, state_dict.pop(key))

        # Shrink the fp32 weight Parameter to match a checkpoint where it was
        # already freed; otherwise super's loader raises a size mismatch.
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

    Memory: ~4× smaller than fp32 ``nn.Linear`` (1 byte/weight + per-block
    scales) at ``block_size=64``.

    Example:
        >>> layer = Linear8bitLt(1024, 4096)
        >>> layer.weight.data.copy_(my_fp32_weight)
        >>> layer.quantize_weight(block_size=64)
        >>> y = layer(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 64,
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

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def quantize_weight(self, block_size: Optional[int] = None) -> "Linear8bitLt":
        block_size = block_size or self.block_size
        state = quantize(self.weight.data, scheme="int8_linear", block_size=block_size)
        self._store_qstate(state)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._qstate()
        if state is None:
            return F.linear(x, self.weight, self.bias)
        w = dequantize(state).to(x.dtype)
        return F.linear(x, w, self.bias)


class Linear4bit(_QuantizedLinearMixin, nn.Module):
    """4-bit linear with packed blockwise-quantized weight.

    Memory: ~8× smaller than fp32 ``nn.Linear`` (0.5 bytes/weight + per-block
    scales) at ``block_size=64``.

    ``compute_dtype`` is the dtype the dequantized weight is materialized in
    and the matmul runs in (default fp16). The bias is cast once at
    ``quantize_weight`` time and cached so forward doesn't reallocate.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4",
        block_size: int = 64,
    ):
        super().__init__()
        if quant_type not in _VALID_4BIT_SCHEMES:
            raise ValueError(
                f"Invalid quant_type={quant_type!r}. "
                f"Expected one of {sorted(_VALID_4BIT_SCHEMES)}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("_bias_compute", None, persistent=False)

        self._q_scheme = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def quantize_weight(self, block_size: Optional[int] = None) -> "Linear4bit":
        block_size = block_size or self.block_size
        state = quantize(self.weight.data, scheme=self.quant_type, block_size=block_size)
        self._store_qstate(state)
        if self.bias is not None:
            self._bias_compute = self.bias.detach().to(self.compute_dtype)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._qstate()
        if state is None:
            return F.linear(x, self.weight, self.bias)
        w = dequantize(state).to(self.compute_dtype)
        bias = self._bias_compute if self._bias_compute is not None else self.bias
        return F.linear(x.to(self.compute_dtype), w, bias)
