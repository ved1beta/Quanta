"""Quantized ``nn.Linear`` drop-ins.

Both layers hold an fp32 master weight in ``self.weight`` until
``.quantize_weight()`` is called. After that the quantized tensors
(``qdata``, ``absmax``, optional ``code``) live on the module as registered
buffers, so they move with ``.to(device)`` and round-trip through
``state_dict()`` / ``load_state_dict()``. Scheme metadata rides along via
``get_extra_state`` / ``set_extra_state``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Quanta.functional.quantization import (
    _TRITON_ENABLED,
    QuantState,
    _is_pow2,
    _warn_triton_fallback,
    dequantize,
    quantize,
)
from Quanta.functional.tensor_ops import int_mm_supported


_VALID_4BIT_SCHEMES = {"nf4", "fp4", "int4_linear"}

_Q_TENSOR_NAMES = ("qdata", "absmax", "code")
_Q_META_KEYS = ("q_scheme", "q_block_size", "q_packed", "q_shape", "q_dtype")


class _QuantizedLinearMixin:
    """Shared state_dict persistence + ``QuantState`` reconstruction helpers."""

    _ALLOWED_SCHEMES: frozenset = frozenset()

    def _init_qbuffers(self) -> None:
        # Registered (as None) up front so later assignments land in _buffers
        # and follow .to()/.cuda() and state_dict like any other buffer.
        for name in _Q_TENSOR_NAMES:
            self.register_buffer(name, None)
        self._q_scheme = None

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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Quantized buffer shapes/dtypes come from the checkpoint, so size them
        # here (on this module's device) and let super() copy the values in.
        ckpt_quantized = bool(state_dict.get(prefix + "_extra_state"))
        for name in _Q_TENSOR_NAMES:
            key = prefix + name
            current = getattr(self, name, None)
            if key in state_dict:
                device = current.device if current is not None else self.weight.device
                setattr(self, name, torch.empty_like(state_dict[key], device=device))
            elif ckpt_quantized and current is not None:
                setattr(self, name, None)  # e.g. ``code`` absent for int4_linear

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

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # Keep quantization scales in fp32 through .half()/.bfloat16(), like
        # bitsandbytes; fp16 absmax costs accuracy for negligible memory.
        for name in ("absmax", "code"):
            t = getattr(self, name, None)
            if t is not None and t.dtype != torch.float32:
                setattr(self, name, t.float())
        self._refresh_cache()  # cached bias must stay in compute_dtype
        return self

    def get_extra_state(self):
        if not getattr(self, "_q_scheme", None):
            return {}
        return {k: getattr(self, "_" + k) for k in _Q_META_KEYS}

    def set_extra_state(self, state):
        if not state:
            self._q_scheme = None
            return
        scheme = state.get("q_scheme")
        if scheme not in self._ALLOWED_SCHEMES:
            raise ValueError(
                f"{type(self).__name__} cannot load a checkpoint quantized with "
                f"scheme={scheme!r}; expected one of {sorted(self._ALLOWED_SCHEMES)}."
            )
        self._q_scheme = scheme
        self._q_block_size = state.get("q_block_size")
        self._q_packed = state.get("q_packed", False)
        self._q_shape = state.get("q_shape")
        self._q_dtype = state.get("q_dtype", torch.float32)


class _Fused4bitMatmul(torch.autograd.Function):
    """Differentiable wrapper around the Triton 4-bit matmul.

    The weight is frozen, so backward only produces the input gradient:
    ``grad_x = grad_y @ dequant(W)``.
    """

    @staticmethod
    def forward(ctx, x_2d, state, out_features, in_features):
        from Quanta.backends.triton_4bit_matmul import matmul_4bit_triton

        ctx.state = state
        ctx.x_dtype = x_2d.dtype
        return matmul_4bit_triton(
            x_2d, state.qdata, state.absmax, state.code,
            out_features=out_features,
            in_features=in_features,
            block_size=state.block_size,
        )

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            w = dequantize(ctx.state).to(grad_out.dtype)
            grad_x = (grad_out @ w).to(ctx.x_dtype)
        return grad_x, None, None, None


class Linear8bitLt(_QuantizedLinearMixin, nn.Module):
    """int8 linear with blockwise-quantized weight.

    Memory: ~4× smaller than fp32 ``nn.Linear`` (1 byte/weight + per-block
    scales) at ``block_size=64``.

    Fast int8 GEMM: when ``block_size == in_features`` the weight has one
    scale per output row, which is compatible with ``torch._int_mm`` — forward
    quantizes the activation per-row, runs an int8×int8→int32 matmul, and
    rescales. Otherwise forward falls back to dequant-and-fp-matmul.
    ``torch._int_mm`` also needs ``in_features`` and ``out_features`` to be
    multiples of 8.

    Example:
        >>> layer = Linear8bitLt(1024, 4096)
        >>> layer.weight.data.copy_(my_fp32_weight)
        >>> layer.quantize_weight(block_size=1024)  # per-row → fast int_mm
        >>> y = layer(x)
    """

    _ALLOWED_SCHEMES = frozenset({"int8_linear"})

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

        self._init_qbuffers()
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

    def _can_int_mm(self, x: torch.Tensor, state: QuantState) -> bool:
        # Per-row weight scales (one block per row) are what int_mm rescaling
        # needs. M is padded up in _int_mm_forward, so only K and N matter.
        return (
            x.is_cuda
            and state.qdata.is_cuda
            and state.block_size == self.in_features
            and int_mm_supported(self._INT_MM_MIN_M, self.in_features, self.out_features)
        )

    # torch._int_mm has a CUDA constraint that M > 16. For smaller M we pad.
    _INT_MM_MIN_M = 17

    def _int_mm_forward(self, x: torch.Tensor, state: QuantState) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float32)
        M = x_2d.shape[0]
        x_absmax = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        x_int8 = (x_2d / x_absmax * 127.0).round().clamp(-127, 127).to(torch.int8)
        # (K, N) column-major view of the row-major (N, K) weight: no copy, and
        # the layout cuBLAS int8 GEMM prefers for the second operand.
        w_t = state.qdata.t()

        if M < self._INT_MM_MIN_M:
            pad = self._INT_MM_MIN_M - M
            x_int8 = torch.cat(
                [x_int8, torch.zeros(pad, self.in_features, dtype=torch.int8, device=x.device)],
                dim=0,
            )
            acc_i32 = torch._int_mm(x_int8.contiguous(), w_t)
            acc_i32 = acc_i32[:M]
        else:
            acc_i32 = torch._int_mm(x_int8.contiguous(), w_t)
        # Rescale: out = acc * (x_absmax/127) * (w_absmax/127)
        out = acc_i32.to(torch.float32) * (x_absmax / 127.0) * (state.absmax.unsqueeze(0) / 127.0)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._qstate()
        if state is None:
            return F.linear(x, self.weight, self.bias)
        if x.dim() >= 2 and self._can_int_mm(x, state):
            return self._int_mm_forward(x, state)
        w = dequantize(state).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Linear4bit(_QuantizedLinearMixin, nn.Module):
    """4-bit linear with packed blockwise-quantized weight.

    Memory: ~8× smaller than fp32 ``nn.Linear`` (0.5 bytes/weight + per-block
    scales) at ``block_size=64``.

    ``compute_dtype`` is the dtype the dequantized weight is materialized in
    and the matmul runs in (default fp16); the output is cast back to the
    input dtype. The bias is cast once at
    ``quantize_weight`` time and cached so forward doesn't reallocate.
    """

    _ALLOWED_SCHEMES = frozenset(_VALID_4BIT_SCHEMES)

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

        self._init_qbuffers()
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

    # The fused kernel is a gemv-style kernel: it beats dequant + cuBLAS only
    # for decode-sized batches and falls far behind above this.
    _FUSED_MAX_M = 16

    def _try_fused_matmul(self, x: torch.Tensor, state: QuantState):
        # Fused 4-bit weight matmul kernel: avoids materializing the fp16
        # weight on every call. Returns None if any precondition fails so the
        # caller can fall back.
        if not (
            _TRITON_ENABLED
            and state.packed
            and state.qdata.is_cuda
            and state.scheme in ("nf4", "fp4")
        ):
            return None
        if x.numel() // self.in_features > self._FUSED_MAX_M:
            return None
        if (
            state.block_size is None
            or not _is_pow2(state.block_size)
            or self.in_features % state.block_size != 0
        ):
            return None
        try:
            from Quanta.backends.triton_4bit_matmul import triton_available
        except ImportError:
            return None
        if not triton_available():
            return None
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        try:
            out = _Fused4bitMatmul.apply(x_2d, state, self.out_features, self.in_features)
        except Exception as e:
            _warn_triton_fallback("4-bit matmul", e)
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
        return F.linear(x.to(self.compute_dtype), w, bias).to(x.dtype)
