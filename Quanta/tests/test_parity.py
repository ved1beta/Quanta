"""
Device parity + QuantState round-trip tests.

Every quantization scheme is exercised on every available device to catch the
class of bug where a codebook or constant tensor lands on the wrong device.
"""

import pytest
import torch

from Quanta.functional.quantization import (
    QuantState,
    quantize,
    dequantize,
    quantize_8bit,
    dequantize_8bit,
    quantize_4bit,
    dequantize_4bit,
    pack_4bit,
    unpack_4bit,
)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

SCHEMES_8BIT = ["int8_linear", "nf8", "fp8"]
SCHEMES_4BIT = ["int4_linear", "nf4", "fp4"]

# Tolerance per scheme, per-tensor (no blockwise). These are round-trip MAE
# ceilings on a standard normal; exceeding these indicates a regression.
TOL_PER_TENSOR = {
    "int8_linear": 0.02,
    "nf8":         0.03,
    "fp8":         0.15,
    "int4_linear": 0.25,
    "nf4":         0.20,
    "fp4":         0.25,
}

# Blockwise tolerances are tighter
TOL_BLOCK = {
    "int8_linear": 0.01,
    "nf8":         0.02,
    "fp8":         0.15,   # fp8 legacy path, no blockwise variant
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
@pytest.mark.parametrize("scheme", ["int8_linear", "nf8", "int4_linear", "nf4", "fp4"])
def test_quantstate_blockwise(device, scheme):
    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device)
    st = quantize(x, scheme=scheme, block_size=64)
    d = dequantize(st)
    mae = (x - d).abs().mean().item()
    assert mae < TOL_BLOCK[scheme], f"{scheme}@{device} blockwise MAE {mae} > {TOL_BLOCK[scheme]}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("quant_type", ["linear", "nf8", "fp8"])
def test_legacy_8bit(device, quant_type):
    torch.manual_seed(0)
    x = torch.randn(256, 256, device=device)
    q, a, b = quantize_8bit(x, quant_type=quant_type)
    assert q.device.type == device
    d = dequantize_8bit(q, a, b, quant_type=quant_type)
    assert d.shape == x.shape
    # loose bound — just ensure it doesn't blow up
    assert (x - d).abs().mean().item() < 0.15


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("quant_type", ["linear", "nf4", "fp4"])
def test_legacy_4bit(device, quant_type):
    torch.manual_seed(0)
    x = torch.randn(256, 256, device=device)
    q, a, b = quantize_4bit(x, quant_type=quant_type)
    assert q.device.type == device
    d = dequantize_4bit(q, a, b, quant_type=quant_type)
    assert d.shape == x.shape
    assert (x - d).abs().mean().item() < 0.25


@pytest.mark.parametrize("device", DEVICES)
def test_4bit_packing_roundtrip(device):
    torch.manual_seed(0)
    # values in [0, 15]
    raw = torch.randint(0, 16, (1024,), device=device, dtype=torch.uint8)
    packed = pack_4bit(raw)
    assert packed.numel() == raw.numel() // 2
    assert packed.dtype == torch.uint8
    unpacked = unpack_4bit(packed, raw.shape)
    assert torch.equal(unpacked, raw)


@pytest.mark.parametrize("device", DEVICES)
def test_4bit_packing_odd_length(device):
    raw = torch.randint(0, 16, (1023,), device=device, dtype=torch.uint8)
    packed = pack_4bit(raw)
    unpacked = unpack_4bit(packed, raw.shape)
    assert torch.equal(unpacked, raw)


@pytest.mark.parametrize("device", DEVICES)
def test_packed_size_vs_unpacked(device):
    """int4_linear via QuantState should be ~2x smaller than int8_linear."""
    x = torch.randn(1024, 1024, device=device)
    s8 = quantize(x, scheme="int8_linear")
    s4 = quantize(x, scheme="int4_linear")
    assert s4.qdata.numel() == s8.qdata.numel() // 2, (s4.qdata.numel(), s8.qdata.numel())


@pytest.mark.parametrize("device", DEVICES)
def test_blockwise_beats_per_tensor_mae(device):
    """Blockwise should always match or beat per-tensor MAE — that's the point."""
    torch.manual_seed(0)
    x = torch.randn(4096, device=device)
    # inject an outlier so per-tensor absmax is dominated by it
    x[0] = 100.0
    flat_st = quantize(x, scheme="int8_linear")
    block_st = quantize(x, scheme="int8_linear", block_size=64)
    mae_flat = (x - dequantize(flat_st)).abs().mean().item()
    mae_block = (x - dequantize(block_st)).abs().mean().item()
    assert mae_block < mae_flat * 0.5, (mae_flat, mae_block)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="int_mm requires CUDA")
def test_int8_matmul_via_int_mm():
    from Quanta.functional.tensor_ops import quantized_matmul_int8
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda")
    b = torch.randn(256, 64, device="cuda")
    # symmetric quantize both to int8
    a_scale = a.abs().amax() / 127
    b_scale = b.abs().amax() / 127
    a_q = torch.round(a / a_scale).clamp(-127, 127).to(torch.int8)
    b_q = torch.round(b / b_scale).clamp(-127, 127).to(torch.int8)
    out = quantized_matmul_int8(a_q, b_q, a_scale, b_scale)
    expected = a @ b
    mae = (out - expected).abs().mean().item()
    rel = mae / expected.abs().mean().item()
    assert rel < 0.02, f"int8 matmul relative error {rel}"
