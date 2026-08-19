#!/usr/bin/env python3
"""Emit the tensor geometry needed by the Android in-place relayout tool."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402

MAGIC = b"G2RLV1\0\0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reader = GGUFReader(args.gguf)
    tensors = [
        tensor for tensor in reader.tensors
        if tensor.tensor_type == GGMLQuantizationType.GPTQ2_32
    ]
    with args.output.open("wb") as output:
        output.write(struct.pack("<8sII", MAGIC, 1, len(tensors)))
        for tensor in tensors:
            columns, rows = map(int, tensor.shape)
            output.write(struct.pack(
                "<QQII", int(tensor.data_offset), int(tensor.n_bytes),
                columns, rows,
            ))
    print(f"wrote {args.output}: tensors={len(tensors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
