import torch 
from typing import Tuple, Optional 


class BaseQuantizer:
    def __init__(self, num_bits: int = 8, symmetric: bool = True):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.max_val = 2**(num_bits - 1) - 1 if symmetric else 2**num_bits - 1

    def _compute_scale_zero_point(
            self, 
            tensor: torch.Tensor, 
            per_channel: bool = False, 
    )-> Tuple[torch.Tensor, torch.Tensor]:
        
        if per_channel:
            dim = 0 if tensor.dim() > 1 else None
            min_val = tensor.min(dim=dim, keepdim=True).values
            max_val = tensor.max(dim=dim, keepdim=True).values
        else:
            min_val = tensor.min()
            max_val = tensor.max()
        
        # Handle edge case where min_val == max_val
        if torch.allclose(min_val, max_val):
            return torch.ones_like(min_val), min_val
        
        if self.symmetric:
            abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = self.max_val / abs_max
            zero_point = torch.zeros_like(min_val)
        else:
            scale = (2**self.num_bits - 1) / (max_val - min_val)
            zero_point = min_val
        
        return scale, zero_point
    
    def quantize(self, 
                tensor: torch.Tensor,
                per_channel: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        scale, zero_point = self._compute_scale_zero_point(tensor, per_channel)
        
        if self.symmetric:
            q_tensor = torch.clamp(
                torch.round(tensor * scale),
                -self.max_val, self.max_val
            ).to(torch.int8)
            q_tensor = (q_tensor + 2**(self.num_bits-1)).to(torch.uint8)
        else:
            q_tensor = torch.clamp(
                torch.round((tensor - zero_point) * scale),
                0, 2**self.num_bits - 1
            ).to(torch.uint8)

        return q_tensor, scale, zero_point
    
    def dequantize(self, 
                  q_tensor: torch.Tensor, 
                  scale: torch.Tensor, 
                  zero_point: torch.Tensor) -> torch.Tensor:
        if not q_tensor.is_contiguous():
            q_tensor = q_tensor.contiguous()

        if self.symmetric:
            q_tensor = q_tensor.to(torch.int8) - 2**(self.num_bits-1)
            return q_tensor.float() / scale
        else:
            return q_tensor.float() / scale + zero_point
