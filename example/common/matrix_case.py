#!/usr/bin/env python3
#
# Copyright (c) 2026 BOSC & ICT, CAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""Prepare external matrix inputs and references for the example runner.

The interchange format is deliberately small and dependency free.  An input
file is JSON with ``dtype``, ``shape`` (``[M, K, N]``), and row-major ``a``,
``b`` and optional ``c`` arrays.  A reference file is either JSON containing
``values`` (or ``reference``) or an already generated Ztt report.

The generated C header is consumed by both the host and RISC-V builds.  This
keeps the matrix exactly the same in the two execution environments without
requiring a guest file system.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


REPORT_VERSION = 1


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON {path}: {exc}") from exc


def as_f32(item: Any, label: str) -> float:
    try:
        value = float(item)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} contains a non-numeric value") from exc
    if not math.isfinite(value):
        raise SystemExit(f"{label} contains a non-finite value")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise SystemExit(f"{label} contains a value outside FP32") from exc


def input_spec(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise SystemExit(f"matrix input must be a JSON object: {path}")
    shape = value.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise SystemExit("matrix input shape must be [M, K, N]")
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise SystemExit("matrix input dimensions must be positive integers")
    m, k, n = shape
    if any(dim % 4 != 0 for dim in shape):
        raise SystemExit(
            "the gemm_fp32 example needs M, K and N to be multiples of 4"
        )
    dtype = value.get("dtype", "f32")
    if dtype != "f32":
        raise SystemExit("the C example currently supports only dtype=f32")

    def matrix(
        name: str, rows: int, cols: int, default: float = 0.0
    ) -> list[float]:
        data = value.get(name)
        if data is None:
            return [default] * (rows * cols)
        if not isinstance(data, list) or len(data) != rows * cols:
            count = rows * cols
            raise SystemExit(
                f"{name} must contain {count} row-major values"
            )
        return [as_f32(item, name) for item in data]

    name = str(value.get("name", path.stem))
    if not name or "\n" in name or "\r" in name:
        raise SystemExit("matrix input name must be a non-empty single line")

    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "a": matrix("a", m, k),
        "b": matrix("b", k, n),
        "c": matrix("c", m, n),
        "embedded_reference": value.get("reference"),
    }


def reference_values(path: Path | None, spec: dict[str, Any]) -> list[float]:
    value: Any
    if path is None:
        value = spec.get("embedded_reference")
        if value is None:
            raise SystemExit(
                "reference mode needs --reference or input.reference"
            )
    else:
        text = path.read_text(encoding="utf-8")
        if text.startswith("Ztt GEMM result report"):
            # A report is copied by make-reference; no JSON conversion needed.
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            message = f"reference must be JSON or a Ztt report: {path}"
            raise SystemExit(message) from exc

    if isinstance(value, dict):
        m, _, n = spec["shape"]
        if value.get("dtype", "f32") != "f32":
            raise SystemExit("reference dtype must be f32")
        if "shape" in value and value["shape"] != [m, n]:
            raise SystemExit(f"reference shape must be [{m}, {n}]")
        value = value.get(
            "values", value.get("reference", value.get("output"))
        )
    if not isinstance(value, list):
        raise SystemExit(
            "reference JSON must contain a values/reference/output array"
        )
    m, _, n = spec["shape"]
    if len(value) != m * n:
        raise SystemExit(f"reference must contain {m * n} row-major values")
    return [as_f32(item, "reference") for item in value]


def c_float(value: float) -> str:
    # Hexadecimal literals are exact and accepted by both clang and GCC.
    return f"{value.hex()}f"


def write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def write_header(args: argparse.Namespace) -> None:
    spec = input_spec(args.input)
    m, k, n = spec["shape"]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    def array(name: str, values: list[float]) -> str:
        rows = []
        for offset in range(0, len(values), n if name != "a" else k):
            cols = n if name != "a" else k
            row = values[offset : offset + cols]
            rows.append("    " + ", ".join(map(c_float, row)))
        declaration = (
            f"static const float ztt_matrix_input_{name}"
            f"[{len(values)}] __attribute__((unused)) = {{\n"
        )
        return declaration + ",\n".join(rows) + "\n};\n"

    text = "/* Generated by common/matrix_case.py; do not edit. */\n"
    text += "#ifndef ZTT_MATRIX_INPUT_H\n#define ZTT_MATRIX_INPUT_H\n\n"
    text += f"#define ZTT_MATRIX_INPUT_M {m}\n"
    text += f"#define ZTT_MATRIX_INPUT_K {k}\n"
    text += f"#define ZTT_MATRIX_INPUT_N {n}\n"
    text += "#define ZTT_LLM_M ZTT_MATRIX_INPUT_M\n"
    text += "#define ZTT_LLM_K ZTT_MATRIX_INPUT_K\n"
    text += "#define ZTT_LLM_N ZTT_MATRIX_INPUT_N\n\n"
    text += f"#define ZTT_MATRIX_CASE_NAME {json.dumps(spec['name'])}\n\n"
    text += "#if !defined(__ASSEMBLER__)\n"
    text += array("a", spec["a"])
    text += array("b", spec["b"])
    text += array("c", spec["c"])
    text += "#endif\n"
    text += "\n#endif /* ZTT_MATRIX_INPUT_H */\n"
    write_if_changed(output, text)


def fnv1a(values: list[float]) -> int:
    # Match ztt_llm_hash() exactly, including its historical offset basis.
    result = 1469598103934665603
    for value in values:
        for byte in struct.pack("<f", value):
            result ^= byte
            result = (result * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result


def write_report(args: argparse.Namespace) -> None:
    spec = input_spec(args.input)
    values = reference_values(args.reference, spec)
    if not values and args.reference is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = args.reference.read_text(encoding="utf-8")
        args.output.write_text(report, encoding="utf-8")
        return
    if len(values) <= 32:
        samples = values
    else:
        samples = [values[(i * len(values)) // 32] for i in range(32)]
    bits = [
        struct.unpack("<I", struct.pack("<f", value))[0]
        for value in samples
    ]
    lines = [
        "Ztt GEMM result report (UTF-8)",
        f"format_version: {REPORT_VERSION}",
        "suite: 1",
        "case_count: 1",
        "",
        f"case[0]: {args.case_name or spec['name']}",
        "  [000] 0x0000000000000000",
        f"  [001] 0x{len(values):016x}",
        f"  [002] 0x{fnv1a(values):016x}",
    ]
    lines.extend(
        f"  [{index + 3:03d}] 0x{item:016x}"
        for index, item in enumerate(bits)
    )
    lines.extend(["", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    header = subparsers.add_parser("header")
    header.add_argument("--input", required=True, type=Path)
    header.add_argument("--output", required=True, type=Path)
    header.set_defaults(function=write_header)

    report = subparsers.add_parser("report")
    report.add_argument("--input", required=True, type=Path)
    report.add_argument("--reference", type=Path)
    report.add_argument("--case-name")
    report.add_argument("--output", required=True, type=Path)
    report.set_defaults(function=write_report)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
