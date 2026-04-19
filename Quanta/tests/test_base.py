"""Tests for the BaseQuantizer reference implementation."""

import torch

from Quanta.functional.base import BaseQuantizer


def _mae(a, b):
    return torch.mean(torch.abs(a - b)).item()


def test_symmetric_8bit_roundtrip():
    q = BaseQuantizer(num_bits=8, symmetric=True)
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q_tensor, scale, zero_point = q.quantize(x)
    d = q.dequantize(q_tensor, scale, zero_point)
    assert q_tensor.dtype == torch.uint8
    assert _mae(x, d) < 0.02


def test_symmetric_4bit_roundtrip():
    q = BaseQuantizer(num_bits=4, symmetric=True)
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q_tensor, scale, zero_point = q.quantize(x)
    d = q.dequantize(q_tensor, scale, zero_point)
    assert q_tensor.dtype == torch.uint8
    assert _mae(x, d) < 0.15


def test_asymmetric_8bit_roundtrip():
    q = BaseQuantizer(num_bits=8, symmetric=False)
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    q_tensor, scale, zero_point = q.quantize(x)
    d = q.dequantize(q_tensor, scale, zero_point)
    assert _mae(x, d) < 0.02


def test_per_channel_shape_preserved():
    q = BaseQuantizer(num_bits=8, symmetric=True)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    q_tensor, scale, zero_point = q.quantize(x, per_channel=True)
    d = q.dequantize(q_tensor, scale, zero_point)
    assert q_tensor.shape == x.shape
    assert _mae(x, d) < 0.05


def test_constant_tensor_roundtrip():
    q = BaseQuantizer(num_bits=8, symmetric=True)
    x = torch.ones(3, 3) * 2.0
    q_tensor, scale, zero_point = q.quantize(x)
    d = q.dequantize(q_tensor, scale, zero_point)
    assert torch.allclose(x, d, atol=1e-5)


def test_tiny_values_dont_nan():
    q = BaseQuantizer(num_bits=8, symmetric=True)
    x = torch.tensor([1e-6, 2e-6, 3e-6])
    q_tensor, scale, zero_point = q.quantize(x)
    d = q.dequantize(q_tensor, scale, zero_point)
    assert torch.isfinite(d).all()
