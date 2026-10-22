"""Whole-model quantization: swap every ``nn.Linear`` for a Quanta layer."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn

from .linear import Linear4bit, Linear8bitLt

log = logging.getLogger(__name__)


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    bits: int = 4,
    *,
    quant_type: str = "nf4",
    block_size: int = 64,
    skip: Iterable[str] = ("lm_head",),
    compute_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Quantize ``model`` in place: every ``nn.Linear`` becomes ``Linear4bit``
    (``bits=4``) or ``Linear8bitLt`` (``bits=8``) with its weight quantized.

    Args:
        model: Any ``nn.Module`` (plain torch or HF ``PreTrainedModel``).
        bits: 4 or 8.
        quant_type: 4-bit scheme — ``nf4`` (default), ``fp4`` or ``int4_linear``.
            Ignored when ``bits=8``.
        block_size: Elements per scale block.
        skip: Module-name substrings left in full precision (``lm_head`` by default).
        compute_dtype: 4-bit matmul dtype; defaults to the original weight dtype.

    The result saves/loads through ``state_dict`` as usual. To reload:
    build the model, call ``quantize_model`` with the same args, then
    ``load_state_dict``.
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    skip = tuple(skip)
    n = 0
    # A Linear reused under several names is quantized once and swapped in at
    # every name, so the model never keeps a stale fp copy alongside it.
    replaced = {}
    for name, mod in list(model.named_modules(remove_duplicate=False)):
        if type(mod) is not nn.Linear or any(s in name for s in skip):
            continue
        if id(mod) in replaced:
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name), child, replaced[id(mod)])
            continue
        # Build on "meta" so the placeholder weight never allocates memory.
        with torch.device("meta"):
            if bits == 4:
                new = Linear4bit(
                    mod.in_features, mod.out_features, bias=mod.bias is not None,
                    compute_dtype=compute_dtype or mod.weight.dtype,
                    quant_type=quant_type, block_size=block_size,
                )
            else:
                new = Linear8bitLt(
                    mod.in_features, mod.out_features, bias=mod.bias is not None,
                    block_size=block_size,
                )
        new.weight = nn.Parameter(mod.weight.data, requires_grad=False)
        new.bias = mod.bias
        new.quantize_weight()
        replaced[id(mod)] = new
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child, new)
        n += 1
    log.info("quantized %d Linear layers to %d-bit", n, bits)
    return model
