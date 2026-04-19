"""Serialization + precision-conversion helpers for quantized tensors."""

import json
import os

import numpy as np
import torch

from ..functional.quantization import (
    dequantize_4bit,
    dequantize_8bit,
    quantize_4bit,
    quantize_8bit,
)


def pack_4bit_tensor(tensor):
    """Pack uint8 values (0..15) two-per-byte. Returns ``(packed, orig_shape)``."""
    if tensor.dtype != torch.uint8:
        raise ValueError("Input tensor must be uint8")

    origin_shape = tensor.shape
    flat = tensor.reshape(-1)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    pairs = flat.reshape(-1, 2)
    return (pairs[:, 0] | (pairs[:, 1] << 4)), origin_shape


def unpack_4bit_tensor(packed_tensor):
    """Inverse of :func:`pack_4bit_tensor` (returns a flat uint8 tensor)."""
    packed_flat = packed_tensor.reshape(-1)
    low = packed_flat & 0x0F
    high = (packed_flat >> 4) & 0x0F
    out = torch.empty(packed_flat.numel() * 2, dtype=torch.uint8)
    out[0::2] = low
    out[1::2] = high
    return out


def tensor_bits_to_bytes(tensor, bits):
    num_elements = torch.tensor(tensor.shape).prod().item()
    return (num_elements * bits + 7) // 8


def save_quantized_tensor(q_tensor, scale, zero_point, params, file_path):
    """Save to a compact ``.qtn`` file (JSON header + raw payload)."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if not file_path.endswith(".qtn"):
        file_path += ".qtn"

    q_np = q_tensor.cpu().numpy()
    scale_np = scale.cpu().numpy() if isinstance(scale, torch.Tensor) else np.asarray(scale)
    zp_np = zero_point.cpu().numpy() if isinstance(zero_point, torch.Tensor) else np.asarray(zero_point)

    dtype_str = str(q_tensor.dtype).removeprefix("torch.")
    metadata = {
        "bits": params.get("bits", 8),
        "scheme": params.get("scheme", "symmetric"),
        "type": params.get("type", "linear"),
        "shape": list(q_tensor.shape),
        "dtype": dtype_str,
    }

    with open(file_path, "wb") as f:
        header = json.dumps(metadata).encode("utf-8")
        f.write(len(header).to_bytes(8, byteorder="little"))
        f.write(header)
        f.write(q_np.tobytes())
        f.write(scale_np.tobytes())
        f.write(zp_np.tobytes())


def load_quantized_tensor(file_path):
    """Inverse of :func:`save_quantized_tensor`."""
    if not file_path.endswith(".qtn"):
        file_path += ".qtn"

    with open(file_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), byteorder="little")
        metadata = json.loads(f.read(header_len).decode("utf-8"))

        shape = tuple(metadata["shape"])
        q_bytes = f.read(int(np.prod(shape)))
        q_tensor = torch.from_numpy(
            np.frombuffer(q_bytes, dtype=np.uint8).copy().reshape(shape)
        )

        scale_np = np.frombuffer(f.read(4), dtype=np.float32).copy()
        zp_np = np.frombuffer(f.read(4), dtype=np.float32).copy()
        scale = torch.tensor(scale_np.item())
        zero_point = torch.tensor(zp_np.item())

        return q_tensor, scale, zero_point, metadata


def save_quantized_tensor_torch(q_tensor, scale, zero_point, params, file_path):
    if not file_path.endswith(".pt"):
        file_path += ".pt"
    torch.save({
        "q_tensor": q_tensor, "scale": scale,
        "zero_point": zero_point, "params": params,
    }, file_path)


def load_quantized_tensor_torch(file_path):
    if not file_path.endswith(".pt"):
        file_path += ".pt"
    d = torch.load(file_path)
    return d["q_tensor"], d["scale"], d["zero_point"], d["params"]


def convert_precision(q_tensor, source_params, target_bits, target_type="linear", target_scheme=None):
    """Dequantize → re-quantize to change bit depth or codebook.

    Returns ``(new_q_tensor, new_scale, new_zero_point, new_params)``.
    """
    source_bits = source_params.get("bits", 8)
    source_type = source_params.get("type", "linear")
    source_scale = source_params.get("scale")
    source_zero_point = source_params.get("zero_point")
    source_scheme = source_params.get("scheme", "symmetric")
    target_scheme = target_scheme or source_scheme

    if source_bits == 8:
        fp_tensor = dequantize_8bit(q_tensor, source_scale, source_zero_point, quant_type=source_type)
    elif source_bits == 4:
        fp_tensor = dequantize_4bit(q_tensor, source_scale, source_zero_point, quant_type=source_type)
    else:
        raise ValueError(f"Unsupported source bit depth: {source_bits}")

    if target_bits == 8:
        new_q, new_scale, new_zp = quantize_8bit(fp_tensor, quant_type=target_type)
    elif target_bits == 4:
        new_q, new_scale, new_zp = quantize_4bit(fp_tensor, quant_type=target_type)
    else:
        raise ValueError(f"Unsupported target bit depth: {target_bits}")

    new_params = {
        "bits": target_bits,
        "type": target_type,
        "scheme": target_scheme,
        "scale": new_scale,
        "zero_point": new_zp,
        "shape": tuple(new_q.shape),
    }
    return new_q, new_scale, new_zp, new_params


def convert_8bit_to_4bit(q_tensor, source_params, target_type="linear"):
    return convert_precision(q_tensor, source_params, 4, target_type)


def convert_4bit_to_8bit(q_tensor, source_params, target_type="linear"):
    return convert_precision(q_tensor, source_params, 8, target_type)


def optimize_for_target_hardware(q_tensor, source_params, target_hardware):
    """Re-quantize to a sensible default for the given hardware label."""
    hw_config = {
        "cpu": {"bits": 8, "type": "linear"},
        "gpu": {"bits": 8, "type": "linear"},
        "mobile": {"bits": 4, "type": "nf4"},
        "edge": {"bits": 4, "type": "linear"},
    }
    config = hw_config.get(target_hardware, {"bits": 8, "type": "linear"})
    return convert_precision(q_tensor, source_params, config["bits"], config["type"])
