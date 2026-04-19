from typing import Tuple

import torch


class BaseQuantizer:
    """Reference quantizer used by ``tensor_ops.Quantizer``.

    Scale convention is inverse-scale (``scale = qmax / absmax`` symmetric;
    ``scale = (2^b - 1) / (max - min)`` asymmetric), so ``dequantize`` divides
    by scale. The functional API in ``quantization.py`` uses the bnb-style
    ``fp = q * absmax/qmax`` convention — don't mix them.
    """

    def __init__(self, num_bits: int = 8, symmetric: bool = True):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.max_val = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** num_bits - 1

    def _compute_scale_zero_point(
        self,
        tensor: torch.Tensor,
        per_channel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if per_channel:
            dim = 0 if tensor.dim() > 1 else None
            min_val = tensor.min(dim=dim, keepdim=True).values
            max_val = tensor.max(dim=dim, keepdim=True).values
        else:
            min_val = tensor.min()
            max_val = tensor.max()

        if torch.allclose(min_val, max_val):
            return torch.ones_like(min_val), min_val

        if self.symmetric:
            abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = self.max_val / abs_max.clamp(min=1e-8)
            zero_point = torch.zeros_like(min_val)
        else:
            scale = (2 ** self.num_bits - 1) / (max_val - min_val).clamp(min=1e-8)
            zero_point = min_val

        return scale, zero_point

    def quantize(
        self,
        tensor: torch.Tensor,
        per_channel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        scale, zero_point = self._compute_scale_zero_point(tensor, per_channel)

        if self.symmetric:
            q_tensor = torch.clamp(
                torch.round(tensor * scale),
                -self.max_val, self.max_val,
            ).to(torch.int8)
            q_tensor = (q_tensor + 2 ** (self.num_bits - 1)).to(torch.uint8)
        else:
            q_tensor = torch.clamp(
                torch.round((tensor - zero_point) * scale),
                0, 2 ** self.num_bits - 1,
            ).to(torch.uint8)

        return q_tensor, scale, zero_point

    def dequantize(
        self,
        q_tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        if not q_tensor.is_contiguous():
            q_tensor = q_tensor.contiguous()

        if self.symmetric:
            q_tensor = q_tensor.to(torch.int8) - 2 ** (self.num_bits - 1)
            return q_tensor.float() / scale
        return q_tensor.float() / scale + zero_point
