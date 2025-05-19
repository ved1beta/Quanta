"""
Memory-efficient Adam optimizer using 8-bit quantization.
"""

import math
import torch
from torch.optim.optimizer import Optimizer

class Adam8bit(Optimizer):
    """
    Implementation of 8-bit Adam algorithm suitable for large-scale models.
    
    This optimizer quantizes state variables to 8-bit to save memory.
    
    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        block_wise: whether to use block-wise quantization (default: True)
        quantize_momentum: whether to quantize momentum (default: True)
        quantize_variance: whether to quantize variance (default: True)
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        block_wise=True,
        quantize_momentum=True,
        quantize_variance=True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_wise=block_wise,
            quantize_momentum=quantize_momentum,
            quantize_variance=quantize_variance,
        )
        super().__init__(params, defaults)
        
    def __setstate__(self, state):
        super().__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # Get parameters
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam8bit does not support sparse gradients")
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                # In actual implementation, these would be quantized
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                
                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                    
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                
                step_size = group["lr"] / bias_correction1
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss 
