import torch 
from typing import Tuple , Optional 


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
            zero_point = torch.round(-min_val * scale)
        
        return scale, zero_point