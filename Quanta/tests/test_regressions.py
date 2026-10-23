"""Regressions for bugs found by stress-testing: value round-trips, device
moves, checkpoint loading, autograd, and shared modules."""

import io
import os

import pytest
import torch
import torch.nn as nn

import Quanta.backends.triton_4bit_matmul as tmm
from Quanta import Linear4bit, Linear8bitLt, VALID_SCHEMES, dequantize, quantize, quantize_model
from Quanta.nn.linear import _Fused4bitMatmul

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ALT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else None
)
needs_alt_device = pytest.mark.skipif(ALT_DEVICE is None, reason="needs a non-CPU device")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block_size", [None, 64])
def test_fp8_values(device, block_size):
    x = torch.tensor([0.0, -0.5, 1.0, 1.45, 3.0, 448.0], device=device)
    d = dequantize(quantize(x, "fp8", block_size=block_size))
    assert d[0] == 0.0
    # E4M3 has 3 mantissa bits → relative error <= 2^-4.
    assert torch.allclose(d[1:], x[1:], rtol=2 ** -4)


def test_fp8_beats_int4_on_weights():
    torch.manual_seed(0)
    w = torch.randn(256, 256) * 0.02
    def mae(scheme):
        return (dequantize(quantize(w, scheme, block_size=64)) - w).abs().mean()

    assert mae("fp8") < mae("int4_linear") / 4


def test_nf8_codebook_endpoints():
    x = torch.tensor([1.0, -1.0, 0.0])
    assert torch.equal(dequantize(quantize(x, "nf8")), x)


@pytest.mark.parametrize("scheme", VALID_SCHEMES)
@pytest.mark.parametrize("block_size", [None, 64])
def test_empty_tensor(scheme, block_size):
    d = dequantize(quantize(torch.empty(0, 3), scheme, block_size=block_size))
    assert d.shape == (0, 3)


@pytest.mark.parametrize("block_size", [0, -4])
def test_bad_block_size(block_size):
    with pytest.raises(ValueError):
        quantize(torch.randn(8), "nf4", block_size=block_size)


@needs_alt_device
@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_then_move_device(bits):
    m = quantize_model(nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 8)), bits=bits)
    x = torch.randn(2, 64)
    ref = m(x)
    m.to(ALT_DEVICE)
    assert m[0].qdata.device.type == ALT_DEVICE
    assert torch.allclose(m(x.to(ALT_DEVICE)).cpu(), ref, atol=1e-2)


@needs_alt_device
def test_load_cpu_checkpoint_into_device_model():
    src = quantize_model(nn.Sequential(nn.Linear(64, 64)), bits=4)
    dst = quantize_model(nn.Sequential(nn.Linear(64, 64)), bits=4).to(ALT_DEVICE)
    dst.load_state_dict(src.state_dict())
    assert dst[0].qdata.device.type == ALT_DEVICE
    assert torch.equal(dst[0].qdata.cpu(), src[0].qdata)


def test_state_dict_loads_with_weights_only():
    m = quantize_model(nn.Sequential(nn.Linear(64, 64)), bits=4)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=True)
    fresh = nn.Sequential(Linear4bit(64, 64, compute_dtype=torch.float32))
    fresh.load_state_dict(sd)
    x = torch.randn(2, 64)
    assert torch.equal(fresh(x), m(x))


def test_mismatched_scheme_checkpoint_raises():
    m8 = quantize_model(nn.Sequential(nn.Linear(64, 64)), bits=8)
    m4 = quantize_model(nn.Sequential(nn.Linear(64, 64)), bits=4)
    with pytest.raises(ValueError, match="int8_linear"):
        m4.load_state_dict(m8.state_dict())


def test_int4_linear_checkpoint_into_nf4_skeleton_drops_code():
    src = quantize_model(nn.Sequential(nn.Linear(64, 64)), bits=4, quant_type="int4_linear")
    dst = quantize_model(nn.Sequential(nn.Linear(64, 64)), bits=4)  # has a ``code`` buffer
    dst.load_state_dict(src.state_dict())
    assert dst[0].code is None
    x = torch.randn(2, 64)
    assert torch.equal(dst(x), src(x))


def test_shared_linear_replaced_everywhere():
    shared = nn.Linear(64, 64)
    m = quantize_model(nn.Sequential(shared, nn.ReLU(), shared), bits=4)
    assert isinstance(m[0], Linear4bit) and m[2] is m[0]


def test_output_dtype_follows_input():
    l4 = Linear4bit(64, 32).quantize_weight()  # compute_dtype fp16
    assert l4(torch.randn(2, 64)).dtype == torch.float32
    l8 = Linear8bitLt(64, 32).quantize_weight()  # fp32 bias
    assert l8(torch.randn(2, 64, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_fused_matmul_backward(monkeypatch):
    # Stand in for the Triton kernel with its reference math so the autograd
    # wrapper is exercised without CUDA.
    def ref_kernel(x, qdata, absmax, code, out_features, in_features, block_size):
        return x.float() @ w_deq.T

    monkeypatch.setattr(tmm, "matmul_4bit_triton", ref_kernel)
    layer = Linear4bit(64, 32, compute_dtype=torch.float32).quantize_weight()
    state = layer._qstate()
    w_deq = dequantize(state).float()

    x = torch.randn(4, 64, requires_grad=True)
    _Fused4bitMatmul.apply(x, state, 32, 64).sum().backward()
    x_ref = x.detach().clone().requires_grad_()
    (x_ref @ w_deq.T).sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5)


needs_triton = pytest.mark.skipif(
    not tmm.triton_available() or os.environ.get("QUANTA_DISABLE_TRITON") == "1",
    reason="needs CUDA + Triton",
)


@needs_triton
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("M", [1, 16])
def test_fused_kernel_fp32_precision_and_backward(quant_type, M):
    torch.manual_seed(0)
    layer = Linear4bit(512, 384, quant_type=quant_type, compute_dtype=torch.float32).cuda().quantize_weight()
    w = dequantize(layer._qstate()).float()
    x = torch.randn(M, 512, device="cuda", requires_grad=True)
    calls = []
    orig = tmm.matmul_4bit_triton
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tmm, "matmul_4bit_triton", lambda *a, **k: calls.append(1) or orig(*a, **k))
        y = layer(x)
    assert calls
    # M=16 takes the tl.dot path; TF32 there gives ~1e-3 relative error.
    torch.testing.assert_close(y, torch.nn.functional.linear(x.detach(), w, layer.bias), rtol=1e-5, atol=1e-5)
    g = torch.randn_like(y)
    y.backward(g)
    torch.testing.assert_close(x.grad, g @ w)


@needs_triton
def test_fused_kernel_skipped_for_large_batch(monkeypatch):
    layer = Linear4bit(256, 128).cuda().quantize_weight()
    monkeypatch.setattr(tmm, "matmul_4bit_triton", lambda *a, **k: pytest.fail("fused kernel used"))
    layer(torch.randn(Linear4bit._FUSED_MAX_M + 1, 256, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="int_mm requires CUDA")
@pytest.mark.parametrize("M", [3, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int_mm_path_with_transposed_view(M, dtype, monkeypatch):
    layer = Linear8bitLt(256, 128).cuda().quantize_weight(block_size=256)
    seen = []
    orig = torch._int_mm
    monkeypatch.setattr(torch, "_int_mm", lambda a, b: seen.append(b.is_contiguous()) or orig(a, b))
    x = torch.randn(M, 256, device="cuda", dtype=dtype)
    y = layer(x)
    assert seen == [False]
    ref = torch.nn.functional.linear(x.float(), dequantize(layer._qstate()).float(), layer.bias)
    assert y.dtype == dtype
    assert ((y.float() - ref).abs().max() / ref.abs().max()) < 2e-2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scales_stay_fp32_after_half(device, dtype):
    model = quantize_model(nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64)), bits=4, skip=())
    model.to(device=device, dtype=dtype)
    for m in model:
        assert m.absmax.dtype == torch.float32 and m.code.dtype == torch.float32
        assert m.absmax.device.type == device
    assert model(torch.randn(2, 64, device=device, dtype=dtype)).dtype == dtype
