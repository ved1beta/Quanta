"""Quantized linear layers.

Master weights live in ``self.weight`` until ``.quantize_weight()`` is called;
after that, the quantized tensors (``qdata``, ``absmax``, optional ``code``)
are attached to the module and travel with ``state_dict()``. Scheme metadata
rides along via ``get_extra_state`` / ``set_extra_state``.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Quanta.functional.quantization import QuantState, dequantize, quantize

_VALID_4BIT_SCHEMES = {"nf4", "fp4", "int4_linear"}
_Q_TENSOR_NAMES = ("qdata", "absmax", "code")
_Q_META_KEYS = ("q_scheme", "q_block_size", "q_packed", "q_shape", "q_dtype")


class _QuantizedLinearMixin:
    """Shared state_dict persistence + qstate reconstruction."""

    def _store_qstate(self, state: QuantState):
        self.qdata = state.qdata
        self.absmax = state.absmax
        self.code = state.code

        self._q_scheme = state.scheme
        self._q_block_size = state.block_size
        self._q_packed = state.packed
        self._q_shape = tuple(state.shape) if state.shape is not None else None
        self._q_dtype = state.dtype

        # Free the fp32 master weight but keep `self.weight` as an empty
        # Parameter so external code introspecting it doesn't blow up.
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
        # Pop quantized tensors before super() so they aren't flagged unexpected.
        for name in _Q_TENSOR_NAMES:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, state_dict.pop(key))

        # If the checkpoint weight is empty, shrink our Parameter so the
        # default loader doesn't raise a size-mismatch error.
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

    Forward currently dequantizes the weight and calls ``F.linear``; true
    int8×int8 GEMM is exposed separately via
    ``Quanta.functional.quantized_matmul_int8``.
    """

    def __init__(self, in_features, out_features, bias=True, block_size=64):
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
    """4-bit linear with packed blockwise-quantized weight (~8× fp32).

    ``compute_dtype`` is the dtype the dequantized weight and matmul run in.
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

        # Cached at quantize_weight() time so forward doesn't reallocate.
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
