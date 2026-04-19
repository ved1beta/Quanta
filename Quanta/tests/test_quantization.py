"""Round-trip tests for the QuantState API."""

import pytest
import torch

from Quanta.functional import (
    QuantState,
    VALID_SCHEMES,
    dequantize,
    pack_4bit,
    quantize,
    unpack_4bit,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scheme", VALID_SCHEMES)
def test_roundtrip_small(device, scheme):
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)
    st = quantize(x, scheme=scheme)
    assert isinstance(st, QuantState)
    d = dequantize(st)
    assert d.shape == x.shape
    assert d.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_constant_tensor(device):
    x = torch.full((16,), 2.0, device=device)
    st = quantize(x, scheme="int8_linear")
    d = dequantize(st)
    assert torch.allclose(x, d, atol=1e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_tensor(device):
    x = torch.zeros(16, device=device)
    st = quantize(x, scheme="int8_linear")
    d = dequantize(st)
    assert torch.allclose(x, d, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_pack_unpack_roundtrip(device):
    raw = torch.randint(0, 16, (1024,), device=device, dtype=torch.uint8)
    packed = pack_4bit(raw)
    assert packed.numel() == raw.numel() // 2
    unpacked = unpack_4bit(packed, raw.shape)
    assert torch.equal(unpacked, raw)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown scheme"):
        quantize(torch.randn(4), scheme="not_a_scheme")
