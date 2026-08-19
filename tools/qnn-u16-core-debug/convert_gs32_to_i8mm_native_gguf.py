#!/usr/bin/env python3
"""Rewrite a gs32_source_v1 GPTQ2_32 GGUF into i8mm_native_v1.

The output has exactly the same file and tensor sizes as the input.  Only the
bytes inside each 64-row GPTQ2_32 source block are permuted:

  [group0(qbytes, metadata), ...]

becomes

  [tile16_0(all-group native qbytes), ..., tile16_3(...),
   group0(metadata), ...]

No expanded or duplicate weight representation is stored in the output.
"""
from __future__ import annotations

import argparse
import mmap
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402

OLD_LAYOUT = b"gs32_source_v1"
NEW_LAYOUT = b"i8mm_native_v1"
ROWS_PER_BLOCK = 64
SOURCE_GROUP_BYTES = 768
QBYTES_PER_GROUP = 512
METADATA_PER_GROUP = 256


def convert_qbytes(old_qbytes: np.ndarray, groups: int) -> np.ndarray:
    """Convert one [groups, 512] GS32 block to [4, groups, 128]."""
    canonical = (
        old_qbytes.reshape(groups, 2, 4, 4, 8, 2)
        .transpose(1, 3, 4, 0, 2, 5)
        .reshape(ROWS_PER_BLOCK, groups, 8)
    )
    shifts = np.arange(4, dtype=np.uint8) * np.uint8(2)
    codes = (
        (canonical[..., None] >> shifts.reshape(1, 1, 1, 4))
        & np.uint8(3)
    ).reshape(ROWS_PER_BLOCK, groups, 32)
    octets = codes.reshape(ROWS_PER_BLOCK, groups, 4, 8)
    native = np.bitwise_or.reduce(
        octets << shifts.reshape(1, 1, 4, 1), axis=2
    ).astype(np.uint8, copy=False)
    return native.reshape(4, 16, groups, 8).transpose(0, 2, 1, 3)


def verify_qbytes(native: np.ndarray, old_qbytes: np.ndarray, groups: int) -> None:
    shifts = np.arange(4, dtype=np.uint8) * np.uint8(2)
    native_rows = native.transpose(0, 2, 1, 3).reshape(
        ROWS_PER_BLOCK, groups, 8
    )
    codes = (
        (native_rows[..., None] >> shifts.reshape(1, 1, 1, 4))
        & np.uint8(3)
    ).transpose(0, 1, 3, 2).reshape(ROWS_PER_BLOCK, groups, 32)
    canonical = (
        (codes[:, :, 0::4])
        | (codes[:, :, 1::4] << np.uint8(2))
        | (codes[:, :, 2::4] << np.uint8(4))
        | (codes[:, :, 3::4] << np.uint8(6))
    )
    restored = (
        canonical.reshape(2, 4, 8, groups, 4, 2)
        .transpose(3, 0, 4, 1, 2, 5)
        .reshape(groups, QBYTES_PER_GROUP)
    )
    if not np.array_equal(restored, old_qbytes):
        raise AssertionError("native GPTQ2 qbyte roundtrip mismatch")


def rewrite_layout_marker(path: Path) -> None:
    if len(OLD_LAYOUT) != len(NEW_LAYOUT):
        raise AssertionError("layout markers must have equal length")
    with path.open("r+b") as output:
        mapped = mmap.mmap(output.fileno(), 0)
        try:
            offset = mapped.find(OLD_LAYOUT)
            if offset < 0 or mapped.find(OLD_LAYOUT, offset + 1) >= 0:
                raise ValueError("expected exactly one gs32_source_v1 marker")
            mapped[offset : offset + len(NEW_LAYOUT)] = NEW_LAYOUT
            mapped.flush()
        finally:
            mapped.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")

    reader = GGUFReader(args.input)
    layout = reader.get_field("general.gptq2_32.layout")
    if layout is None or layout.contents() != OLD_LAYOUT.decode("ascii"):
        raise ValueError("input is not a gs32_source_v1 GGUF")
    tensors = [
        tensor for tensor in reader.tensors
        if tensor.tensor_type == GGMLQuantizationType.GPTQ2_32
    ]
    if not tensors:
        raise ValueError("input contains no GPTQ2_32 tensors")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output)
    rewrite_layout_marker(args.output)

    for tensor_index, tensor in enumerate(tensors):
        columns, rows = map(int, tensor.shape)
        if columns % 32 or rows % ROWS_PER_BLOCK:
            raise ValueError(f"unsupported GPTQ2 geometry {tensor.name}: {columns}x{rows}")
        groups = columns // 32
        block_bytes = groups * SOURCE_GROUP_BYTES
        if tensor.n_bytes != rows // ROWS_PER_BLOCK * block_bytes:
            raise ValueError(f"unexpected GPTQ2 size for {tensor.name}")
        payload = np.memmap(
            args.output,
            dtype=np.uint8,
            mode="r+",
            offset=tensor.data_offset,
            shape=(tensor.n_bytes,),
        )
        for row_block in range(rows // ROWS_PER_BLOCK):
            begin = row_block * block_bytes
            old = payload[begin : begin + block_bytes].copy().reshape(
                groups, SOURCE_GROUP_BYTES
            )
            old_qbytes = old[:, :QBYTES_PER_GROUP]
            native = convert_qbytes(old_qbytes, groups)
            if args.verify:
                verify_qbytes(native, old_qbytes, groups)
            payload[begin : begin + groups * QBYTES_PER_GROUP] = native.reshape(-1)
            payload[
                begin + groups * QBYTES_PER_GROUP : begin + block_bytes
            ] = old[:, QBYTES_PER_GROUP:].reshape(-1)
        payload.flush()
        del payload
        print(
            f"[{tensor_index + 1}/{len(tensors)}] {tensor.name} "
            f"{columns}x{rows} bytes={tensor.n_bytes}",
            flush=True,
        )

    if args.output.stat().st_size != args.input.stat().st_size:
        raise AssertionError("output GGUF size changed")
    print(f"wrote {args.output} bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
