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
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Rebuild non-persistent tensors derived from the quantized state."""

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

    Fast int8 GEMM: when ``block_size == in_features`` the weight has one
    scale per output row, which is compatible with ``torch._int_mm`` — forward
    quantizes the activation per-row, runs an int8×int8→int32 matmul, and
    rescales. Otherwise forward falls back to dequant-and-fp-matmul.

    Example:
        >>> layer = Linear8bitLt(1024, 4096)
        >>> layer.weight.data.copy_(my_fp32_weight)
        >>> layer.quantize_weight(block_size=1024)  # per-row → fast int_mm
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

        self.register_buffer("_w_int8_t", None, persistent=False)

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
        self._refresh_cache()
        return self

    def _refresh_cache(self) -> None:
        # Cache transposed int8 weight (in,out) for the fast int_mm path.
        state = self._qstate()
        if (
            state is not None
            and state.block_size == self.in_features
            and state.qdata.is_cuda
            and hasattr(torch, "_int_mm")
        ):
            self._w_int8_t = state.qdata.t().contiguous()
        else:
            self._w_int8_t = None

    # torch._int_mm has a CUDA constraint that M > 16. For smaller M we pad.
    _INT_MM_MIN_M = 17

    def _int_mm_forward(self, x: torch.Tensor, state: QuantState) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float32)
        M = x_2d.shape[0]
        x_absmax = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        x_int8 = (x_2d / x_absmax * 127.0).round().clamp(-127, 127).to(torch.int8)

        if M < self._INT_MM_MIN_M:
            pad = self._INT_MM_MIN_M - M
            x_int8 = torch.cat(
                [x_int8, torch.zeros(pad, self.in_features, dtype=torch.int8, device=x.device)],
                dim=0,
            )
            acc_i32 = torch._int_mm(x_int8.contiguous(), self._w_int8_t)
            acc_i32 = acc_i32[:M]
        else:
            acc_i32 = torch._int_mm(x_int8.contiguous(), self._w_int8_t)
        # Rescale: out = acc * (x_absmax/127) * (w_absmax/127)
        out = acc_i32.to(torch.float32) * (x_absmax / 127.0) * (state.absmax.unsqueeze(0) / 127.0)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._qstate()
        if state is None:
            return F.linear(x, self.weight, self.bias)
        if (
            self._w_int8_t is not None
            and x.is_cuda
            and x.dim() >= 2
        ):
            return self._int_mm_forward(x, state)
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
        self._refresh_cache()
        return self

    def _refresh_cache(self) -> None:
        if self.bias is not None:
            self._bias_compute = self.bias.detach().to(self.compute_dtype)

    def _try_fused_matmul(self, x: torch.Tensor, state: QuantState):
        # Fused 4-bit weight matmul kernel: avoids materializing the fp16
        # weight on every call. Returns None if any precondition fails so the
        # caller can fall back.
        if not (state.packed and state.qdata.is_cuda and state.scheme in ("nf4", "fp4")):
            return None
        if state.block_size is None or self.in_features % state.block_size != 0:
            return None
        try:
            from Quanta.backends.triton_4bit_matmul import (
                matmul_4bit_triton,
                triton_available,
            )
        except ImportError:
            return None
        if not triton_available():
            return None
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        try:
            out = matmul_4bit_triton(
                x_2d, state.qdata, state.absmax, state.code,
                out_features=self.out_features,
                in_features=self.in_features,
                block_size=state.block_size,
            )
        except Exception:
            return None
        out = out.to(self.compute_dtype)
        bias = self._bias_compute if self._bias_compute is not None else self.bias
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._qstate()
        if state is None:
            return F.linear(x, self.weight, self.bias)
        fused = self._try_fused_matmul(x, state)
        if fused is not None:
            return fused
        w = dequantize(state).to(self.compute_dtype)
        bias = self._bias_compute if self._bias_compute is not None else self.bias
        return F.linear(x.to(self.compute_dtype), w, bias)
