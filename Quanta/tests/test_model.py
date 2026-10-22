"""Whole-model quantization: swap, accuracy, skip list, state_dict reload."""

import pytest
import torch
import torch.nn as nn

from Quanta import Linear4bit, Linear8bitLt, quantize_model


def _model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128, bias=False), nn.ReLU(),
        nn.Sequential(nn.Linear(128, 32)),
    )


# bf16 compute makes the 4-bit bias cache a real copy (not an alias), so a
# stale cache after load_state_dict shows up as a mismatch.
@pytest.mark.parametrize("bits,cls,tol,kw", [
    (4, Linear4bit, 0.15, {"compute_dtype": torch.bfloat16}),
    (8, Linear8bitLt, 0.02, {}),
])
def test_quantize_model_roundtrip(bits, cls, tol, kw):
    x = torch.randn(16, 64)
    ref_model = _model()
    ref = ref_model(x)

    qm = quantize_model(_model(), bits=bits, skip=(), **kw)
    assert all(isinstance(m, cls) for m in qm.modules() if isinstance(m, (nn.Linear, cls)))
    assert qm[2].bias is None
    out = qm(x)
    assert (out - ref).abs().mean() < tol * ref.abs().mean()

    # Reload into a fresh quantized skeleton; forward must match bit-for-bit.
    skeleton = _model()
    skeleton[0].bias.data.zero_()  # wrong until load; exposes stale caches
    fresh = quantize_model(skeleton, bits=bits, skip=(), **kw)
    fresh.load_state_dict(qm.state_dict())
    assert torch.equal(fresh(x), out)


def test_skip_keeps_linear():
    qm = quantize_model(_model(), bits=4, skip=("4.",))
    assert type(qm[4][0]) is nn.Linear
    assert isinstance(qm[0], Linear4bit)


def test_bad_bits():
    with pytest.raises(ValueError):
        quantize_model(_model(), bits=3)
