#!/usr/bin/env python3
"""Export compact I8MM-native two-bit Decode weights from a GS32 GGUF."""
from __future__ import annotations
import argparse
import os
import re
import struct
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402

MAGIC = b"QG2I8M1\0"
VERSION = 1
HEADER = struct.Struct("<8sIIQ")
ENTRY = struct.Struct("<iIIIQQQ16sQ")
ALIGNMENT = 4096
PROJECTIONS = {
    "attn_q": (0, "self_attn.q_proj"),
    "attn_k": (1, "self_attn.k_proj"),
    "attn_v": (2, "self_attn.v_proj"),
    "attn_output": (3, "self_attn.o_proj"),
    "ffn_gate": (4, "mlp.gate_proj"),
    "ffn_up": (5, "mlp.up_proj"),
    "ffn_down": (6, "mlp.down_proj"),
}
TENSOR_RE = re.compile(
    r"^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$"
)

def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

def fnv1a_sample(raw: np.ndarray, columns: int, rows: int) -> int:
    flat = raw.reshape(-1)
    sample = np.concatenate((flat[:512], flat[-512:]))
    value = 0xCBF29CE484222325
    for byte in sample:
        value ^= int(byte)
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for item in (columns, rows):
        for shift in range(0, 64, 8):
            value ^= (item >> shift) & 0xFF
            value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value

def convert_tensor(raw: np.ndarray, columns: int, rows: int) -> np.ndarray:
    if columns % 32 or rows % 64:
        raise ValueError(f"unsupported GS32 geometry {columns}x{rows}")
    groups = columns // 32
    if raw.size != rows * groups * 12:
        raise ValueError(f"unexpected tensor bytes: {raw.size}")
    row_blocks = rows // 64
    qbytes = raw.reshape(row_blocks, groups, 768)[:, :, :512]
    canonical = (
        qbytes.reshape(row_blocks, groups, 2, 4, 4, 8, 2)
        .transpose(0, 2, 4, 5, 1, 3, 6)
        .reshape(rows, groups, 8)
    )
    shifts = np.arange(4, dtype=np.uint8) * np.uint8(2)
    codes = (
        (canonical[..., None] >> shifts.reshape(1, 1, 1, 4))
        & np.uint8(3)
    ).reshape(rows, groups, 32)
    octets = codes.reshape(rows, groups, 4, 8)
    native = np.bitwise_or.reduce(
        octets << shifts.reshape(1, 1, 4, 1), axis=2
    ).astype(np.uint8, copy=False)
    result = (
        native.reshape(row_blocks, 4, 8, 2, groups, 8)
        .transpose(0, 1, 4, 2, 3, 5)
        .reshape(rows // 16, groups, 128)
    )
    roundtrip_octets = (
        result.reshape(row_blocks, 4, groups, 8, 2, 8)
        .transpose(0, 1, 3, 4, 2, 5)
        .reshape(rows, groups, 8)
    )
    roundtrip_codes = (
        (roundtrip_octets[..., None] >> shifts.reshape(1, 1, 1, 4))
        & np.uint8(3)
    ).transpose(0, 1, 3, 2).reshape(rows, groups, 32)
    if not np.array_equal(roundtrip_codes, codes):
        raise AssertionError("native two-bit layout roundtrip mismatch")
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-layers", type=int, default=0)
    args = parser.parse_args()
    reader = GGUFReader(args.gguf)
    layout = reader.get_field("general.gptq2_32.layout")
    if layout is None or layout.contents() != "gs32_source_v1":
        raise ValueError("input is not a gs32_source_v1 GGUF")
    tensors = []
    for tensor in reader.tensors:
        match = TENSOR_RE.match(tensor.name)
        if match is None:
            continue
        layer = int(match.group(1))
        if args.max_layers and layer >= args.max_layers:
            continue
        projection_id, projection = PROJECTIONS[match.group(2)]
        if tensor.tensor_type != GGMLQuantizationType.GPTQ2_32:
            raise ValueError(f"{tensor.name} is not GPTQ2_32")
        columns, rows = map(int, tensor.shape)
        size = rows * (columns // 32) * 8
        tensors.append((
            layer, projection_id, projection, columns, rows, size,
            fnv1a_sample(tensor.data, columns, rows), tensor,
        ))
    tensors.sort(key=lambda item: (item[0], item[1]))
    if not tensors:
        raise ValueError("no decoder GPTQ2 tensors found")
    data_offset = align(HEADER.size + len(tensors) * ENTRY.size)
    offset = data_offset
    entries = []
    for layer, pid, projection, columns, rows, size, fingerprint, _ in tensors:
        entries.append((
            layer, pid, columns, rows, offset, size, fingerprint,
            projection.encode("ascii")[:16].ljust(16, b"\0"), 0,
        ))
        offset += size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(HEADER.pack(MAGIC, VERSION, len(entries), data_offset))
        for entry in entries:
            output.write(ENTRY.pack(*entry))
        output.write(bytes(data_offset - output.tell()))
        for index, item in enumerate(tensors):
            layer, _, projection, columns, rows, size, _, tensor = item
            native = convert_tensor(tensor.data, columns, rows)
            if native.nbytes != size:
                raise AssertionError(f"{tensor.name}: {native.nbytes} != {size}")
            output.write(native.tobytes(order="C"))
            print(
                f"[{index + 1}/{len(tensors)}] layer={layer} {projection} "
                f"{columns}x{rows} bytes={size}", flush=True,
            )
        output.flush()
        os.fsync(output.fileno())
    print(f"wrote {args.output} bytes={args.output.stat().st_size}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
