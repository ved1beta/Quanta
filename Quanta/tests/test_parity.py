"""Device parity + numerical-quality tests for the QuantState API.

Every scheme is exercised on every available device. Tolerances are
round-trip MAE ceilings on a standard normal — exceeding them indicates a
regression.
"""

import pytest
import torch

from Quanta.functional import (
    QuantState,
    dequantize,
    quantize,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

SCHEMES_8BIT = ["int8_linear", "nf8", "fp8"]
SCHEMES_4BIT = ["int4_linear", "nf4", "fp4"]

TOL_PER_TENSOR = {
    "int8_linear": 0.02,
    "nf8":         0.03,
    "fp8":         0.15,
    "int4_linear": 0.25,
    "nf4":         0.20,
    "fp4":         0.25,
}

TOL_BLOCK = {
    "int8_linear": 0.01,
    "nf8":         0.02,
    "int4_linear": 0.15,
    "nf4":         0.10,
    "fp4":         0.15,
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scheme", SCHEMES_8BIT + SCHEMES_4BIT)
def test_quantstate_roundtrip(device, scheme):
    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device)
    st = quantize(x, scheme=scheme)
    assert isinstance(st, QuantState)
    assert st.qdata.device.type == device
    d = dequantize(st)
    assert d.device.type == device
    assert d.shape == x.shape
    mae = (x - d).abs().mean().item()
    assert mae < TOL_PER_TENSOR[scheme], f"{scheme}@{device} MAE {mae} > {TOL_PER_TENSOR[scheme]}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scheme", list(TOL_BLOCK))
def test_quantstate_blockwise(device, scheme):
    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device)
    st = quantize(x, scheme=scheme, block_size=64)
    d = dequantize(st)
    mae = (x - d).abs().mean().item()
    assert mae < TOL_BLOCK[scheme], f"{scheme}@{device} blockwise MAE {mae} > {TOL_BLOCK[scheme]}"


@pytest.mark.parametrize("device", DEVICES)
def test_packed_size_vs_unpacked(device):
    """int4_linear packs 2 nibbles per byte → ~2× smaller than int8_linear."""
    x = torch.randn(1024, 1024, device=device)
    s8 = quantize(x, scheme="int8_linear")
    s4 = quantize(x, scheme="int4_linear")
    assert s4.qdata.numel() == s8.qdata.numel() // 2


@pytest.mark.parametrize("device", DEVICES)
def test_blockwise_beats_per_tensor_under_outlier(device):
    torch.manual_seed(0)
    x = torch.randn(4096, device=device)
    x[0] = 100.0  # outlier dominates per-tensor absmax
    flat_st = quantize(x, scheme="int8_linear")
    block_st = quantize(x, scheme="int8_linear", block_size=64)
    mae_flat = (x - dequantize(flat_st)).abs().mean().item()
    mae_block = (x - dequantize(block_st)).abs().mean().item()
    assert mae_block < mae_flat * 0.5, (mae_flat, mae_block)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="int_mm requires CUDA")
def test_int8_matmul_via_int_mm():
    from Quanta.functional import quantized_matmul_int8

    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda")
    b = torch.randn(256, 64, device="cuda")
    a_scale = a.abs().amax() / 127
    b_scale = b.abs().amax() / 127
    a_q = torch.round(a / a_scale).clamp(-127, 127).to(torch.int8)
    b_q = torch.round(b / b_scale).clamp(-127, 127).to(torch.int8)
    out = quantized_matmul_int8(a_q, b_q, a_scale, b_scale)
    expected = a @ b
    rel = (out - expected).abs().mean().item() / expected.abs().mean().item()
    assert rel < 0.02, f"int8 matmul relative error {rel}"
