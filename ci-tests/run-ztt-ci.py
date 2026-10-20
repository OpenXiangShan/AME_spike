#!/usr/bin/env python3
"""Run and independently compare the checked-in AMEspike Ztt asset set.

The asset directory is an input-only directory.  In particular, this script
never opens historical actual.bin/actual.hex files and never treats a
historical report status as the result of the current run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any


TRACE_RE = re.compile(
    r"core\s+\d+:\s+0x([0-9a-fA-F]+)\s+\(0x([0-9a-fA-F]+)\)"
)
HEX_WORD_RE = re.compile(r"[0-9a-fA-F]{16}")
SIGNATURE_HEADER_SIZE = 8 * 8
FLOAT_ELEMENT_WIDTH = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2}


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"invalid JSONL {path}:{line_number}: {exc}")
                if not isinstance(value, dict):
                    fail(f"JSONL record is not an object: {path}:{line_number}")
                rows.append(value)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    return rows


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        fail(f"path escaped asset root: {path}")


def asset_path(root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        fail(f"{field} is not a non-empty path")
    raw_path = Path(raw)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        try:
            candidates.append(raw_path.resolve().relative_to(root.resolve()))
        except ValueError:
            pass
        parts = raw_path.parts
        if "verify-all-fixed" in parts:
            marker = parts.index("verify-all-fixed")
            candidates.append(Path(*parts[marker + 1 :]))
    else:
        candidates.append(raw_path)

    for candidate in candidates:
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return resolved
    fail(f"{field} cannot be safely mapped into asset root: {raw}")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        fail(f"missing {description}: {path}")


def parse_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{field} is not an integer: {value!r}")
    return value


def parse_ulp_bound(value: Any, field: str) -> Fraction:
    try:
        bound = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        fail(f"{field} is not a valid ULP bound: {value!r}")
    if bound <= 0:
        fail(f"{field} must be positive")
    return bound


def expected_trace_counts(log: str) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for match in TRACE_RE.finditer(log):
        key = (int(match.group(1), 16), int(match.group(2), 16))
        counts[key] = counts.get(key, 0) + 1
    return counts


def decode_signature(path: Path) -> bytes:
    words: list[bytes] = []
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        fail(f"invalid Spike signature file {path}: {exc}")
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        if not HEX_WORD_RE.fullmatch(line):
            fail(f"signature line is not a 64-bit hex word: {path}:{line_number}")
        words.append(bytes.fromhex(line)[::-1])
    if not words:
        fail(f"Spike produced an empty signature: {path}")
    return b"".join(words)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def diff_bytes(expected: bytes, actual: bytes, limit: int = 64) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for offset, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            differences.append(
                {"offset": offset, "expected": f"{want:02x}", "actual": f"{got:02x}"}
            )
            if len(differences) == limit:
                break
    if len(expected) != len(actual):
        differences.append(
            {
                "length_expected": len(expected),
                "length_actual": len(actual),
            }
        )
    return differences


def check_expected_contract(row: dict[str, Any], path: Path) -> None:
    """Validate the metadata needed to interpret one expected signature.

    This consumes only the expected side of report.jsonl.  The historical
    actual/status fields are intentionally not consulted.
    """
    for field in ("case_id", "case_index", "target_pc", "target_encoding", "target_executions"):
        if field not in row:
            fail(f"{path}: missing expected contract field {field}")
    expected = row.get("expected")
    if not isinstance(expected, dict) or not isinstance(expected.get("header"), dict):
        fail(f"{path}: missing expected signature header contract")
    observations = row.get("expected_observations")
    if not isinstance(observations, dict):
        fail(f"{path}: missing expected observation contract")
    if row.get("observation_state") != "COMPLETE":
        fail(f"{path}: expected observation state is not COMPLETE")
    if row.get("target_instruction") is not True:
        fail(f"{path}: target_instruction is not true")
    executions = parse_int(row["target_executions"], f"{path}:target_executions")
    if executions < 1:
        fail(f"{path}: target_executions must be positive")
    for field in ("target_pc", "target_encoding"):
        if parse_int(row[field], f"{path}:{field}") < 0:
            fail(f"{path}: {field} must be non-negative")
    approximation = row.get("approximation_comparison")
    if approximation is not None:
        if not isinstance(approximation, dict):
            fail(f"{path}: approximation comparison is not an object")
        datatype = row.get("datatype")
        if datatype not in FLOAT_ELEMENT_WIDTH:
            fail(f"{path}: approximation comparison has unsupported datatype")
        elements = approximation.get("elements")
        expected_elements = 32 if datatype in ("fp16", "bf16") else 16
        if not isinstance(elements, list) or len(elements) != expected_elements:
            fail(
                f"{path}: expected {expected_elements} approximation elements, "
                f"got {len(elements) if isinstance(elements, list) else 'invalid'}"
            )
        for index, element in enumerate(elements):
            if not isinstance(element, dict) or "bound" not in element:
                fail(f"{path}: approximation element {index} has no bound")
            parse_ulp_bound(element["bound"], f"{path}:approximation[{index}].bound")


def ordered_float_bits(bits: int, width: int) -> int:
    sign = 1 << (width - 1)
    magnitude = sign - 1
    if bits & sign:
        return sign - (bits & magnitude)
    return sign + bits


def float_ulp_distance(actual: int, expected: int, width: int) -> int | None:
    exponent_bits = 11 if width == 64 else 8 if width == 32 else 5 if width == 16 else 8
    fraction_bits = width - exponent_bits - 1
    exponent_mask = (1 << exponent_bits) - 1
    fraction_mask = (1 << fraction_bits) - 1

    def is_nan(value: int) -> bool:
        return (
            ((value >> fraction_bits) & exponent_mask) == exponent_mask
            and (value & fraction_mask) != 0
        )
    if is_nan(actual) or is_nan(expected):
        return 0 if actual == expected else None
    return abs(
        ordered_float_bits(actual, width) - ordered_float_bits(expected, width)
    )


def compare_signature(
    expected: bytes, actual: bytes, case: dict[str, Any]
) -> list[dict[str, Any]]:
    approximation = case.get("approximation_bounds")
    if not approximation:
        return diff_bytes(expected, actual)

    width = int(case["approximation_width"])
    element_count = len(approximation)
    value_bytes = width // 8 * element_count
    differences = diff_bytes(
        expected[:SIGNATURE_HEADER_SIZE], actual[:SIGNATURE_HEADER_SIZE]
    )
    expected_values = expected[
        SIGNATURE_HEADER_SIZE : SIGNATURE_HEADER_SIZE + value_bytes
    ]
    actual_values = actual[
        SIGNATURE_HEADER_SIZE : SIGNATURE_HEADER_SIZE + value_bytes
    ]
    for index, bound in enumerate(approximation):
        start = index * width // 8
        end = start + width // 8
        if end > len(expected_values) or end > len(actual_values):
            differences.append(
                {
                    "offset": SIGNATURE_HEADER_SIZE + start,
                    "reason": "floating-point value is outside signature",
                }
            )
            continue
        expected_bits = int.from_bytes(expected_values[start:end], "little")
        actual_bits = int.from_bytes(actual_values[start:end], "little")
        distance = float_ulp_distance(actual_bits, expected_bits, width)
        if distance is None or Fraction(distance) > bound:
            differences.append(
                {
                    "offset": SIGNATURE_HEADER_SIZE + start,
                    "expected": f"0x{expected_bits:0{width // 4}x}",
                    "actual": f"0x{actual_bits:0{width // 4}x}",
                    "ulp_distance": distance,
                    "ulp_bound": str(bound),
                }
            )
    differences.extend(
        diff_bytes(
            expected[SIGNATURE_HEADER_SIZE + value_bytes :],
            actual[SIGNATURE_HEADER_SIZE + value_bytes :],
        )
    )
    return differences


def discover_programs(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    checkpoint_path = root / "checkpoint.json"
    require_file(checkpoint_path, "checkpoint")
    checkpoint = read_json(checkpoint_path)
    if not isinstance(checkpoint, dict):
        fail("checkpoint.json is not an object")
    partitions = checkpoint.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != 66:
        fail(f"checkpoint must contain 66 partitions, got {len(partitions or [])}")

    require_file(root / "targets.jsonl", "global target catalog")
    catalog = checkpoint.get("catalog")
    if not isinstance(catalog, dict) or not catalog.get("sha256"):
        fail("checkpoint has no global catalog checksum")
    if sha256(root / "targets.jsonl") != catalog["sha256"]:
        fail("global targets.jsonl SHA256 does not match checkpoint")

    expected_partitions = {f"partition_{index:04d}" for index in range(66)}
    actual_partitions = {path.name for path in root.glob("partition_*") if path.is_dir()}
    if actual_partitions != expected_partitions:
        fail(
            "partition directory set mismatch: "
            f"missing={sorted(expected_partitions - actual_partitions)} "
            f"extra={sorted(actual_partitions - expected_partitions)}"
        )

    for entry in partitions:
        if not isinstance(entry, dict):
            fail("checkpoint partition entry is not an object")
        directory = asset_path(root, entry.get("directory"), "checkpoint partition")
        catalog_info = entry.get("catalog")
        if not isinstance(catalog_info, dict) or not catalog_info.get("sha256"):
            fail(f"checkpoint has no catalog checksum for {directory}")
        catalog_path = asset_path(root, catalog_info.get("path"), "partition catalog")
        require_file(catalog_path, "partition target catalog")
        if sha256(catalog_path) != catalog_info["sha256"]:
            fail(f"partition catalog SHA256 mismatch: {catalog_path}")

    manifest_paths = [root / "base_000" / "manifest.json"]
    manifest_paths.extend(
        root / name / "executed_000" / "manifest.json"
        for name in sorted(expected_partitions)
    )
    for path in manifest_paths:
        require_file(path, "execution manifest")
    if len(manifest_paths) != 67:
        fail("expected exactly 67 execution manifests")

    programs: list[dict[str, Any]] = []
    expected_elfs: set[str] = set()
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("programs"), list):
            fail(f"invalid program list in {manifest_path}")
        scope = "base_000" if manifest_path.parts[-2] == "base_000" else manifest_path.parts[-3]
        for entry in manifest["programs"]:
            if not isinstance(entry, dict):
                fail(f"program entry is not an object: {manifest_path}")
            elf = asset_path(root, entry.get("elf"), f"{manifest_path}:elf")
            require_file(elf, "manifest ELF")
            rel = relative_path(elf, root)
            expected_elfs.add(rel)
            identity = entry.get("elf_identity")
            expected_sha = identity.get("sha256") if isinstance(identity, dict) else None
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                fail(f"manifest ELF has no SHA256: {elf}")
            actual_sha = sha256(elf)
            if actual_sha != expected_sha:
                fail(f"ELF SHA256 mismatch for {rel}: {actual_sha} != {expected_sha}")

            report = elf.parent / "report.jsonl"
            require_file(report, "case report")
            rows = jsonl(report)
            if not rows:
                fail(f"empty case report: {report}")
            case_count = parse_int(
                entry.get("generated_case_count"), f"{manifest_path}:generated_case_count"
            )
            if len(rows) != case_count:
                fail(f"case count mismatch in {report}: {len(rows)} != {case_count}")
            indices = [parse_int(row.get("case_index"), f"{report}:case_index") for row in rows]
            if sorted(indices) != list(range(case_count)):
                fail(f"case indices are not contiguous in {report}")
            signature_size = parse_int(
                entry.get("signature_size"), f"{manifest_path}:signature_size"
            )
            if signature_size <= 0:
                fail(f"invalid program signature size in {manifest_path}")

            intervals: list[tuple[int, int]] = []
            cases: list[dict[str, Any]] = []
            for row in rows:
                check_expected_contract(row, report)
                offset = parse_int(row.get("signature_offset"), f"{report}:signature_offset")
                size = parse_int(row.get("signature_size"), f"{report}:signature_size")
                if offset < 0 or size <= 0 or offset + size > signature_size:
                    fail(f"signature range is outside program output: {report}")
                expected_path = asset_path(root, row.get("expected_bin"), f"{report}:expected_bin")
                require_file(expected_path, "Expected signature")
                if expected_path.stat().st_size != size:
                    fail(
                        f"Expected length mismatch for {expected_path}: "
                        f"{expected_path.stat().st_size} != {size}"
                    )
                expected_header = row["expected"]["header"]
                header_fields = (
                    "status",
                    "mcause",
                    "mepc",
                    "mtval",
                    "amexsat",
                    "amefflags",
                    "amestatus",
                    "reserved",
                )
                try:
                    expected_header_bytes = struct.pack(
                        "<8Q",
                        *(parse_int(expected_header[field], f"{report}:expected.header.{field}")
                          for field in header_fields),
                    )
                except KeyError as exc:
                    fail(f"{report}: expected signature header is missing {exc.args[0]}")
                expected_bytes = expected_path.read_bytes()
                if expected_bytes[: len(expected_header_bytes)] != expected_header_bytes:
                    fail(f"{report}: Expected signature header does not match report metadata")
                approximation = row.get("approximation_comparison")
                approximation_bounds: list[Fraction] = []
                approximation_width = 0
                if approximation is not None:
                    datatype = row["datatype"]
                    approximation_width = FLOAT_ELEMENT_WIDTH[datatype] * 8
                    approximation_bounds = [
                        parse_ulp_bound(
                            element["bound"],
                            f"{report}:approximation[{index}].bound",
                        )
                        for index, element in enumerate(approximation["elements"])
                    ]
                    value_bytes = approximation_width // 8 * len(approximation_bounds)
                    if len(expected_bytes) < SIGNATURE_HEADER_SIZE + value_bytes:
                        fail(
                            f"{report}: Expected signature is too short for "
                            "floating-point comparison"
                        )
                intervals.append((offset, offset + size))
                cases.append(
                    {
                        "case_id": row["case_id"],
                        "case_index": row["case_index"],
                        "target_pc": row["target_pc"],
                        "target_encoding": row["target_encoding"],
                        "target_executions": row["target_executions"],
                        "signature_offset": offset,
                        "signature_size": size,
                        "expected_path": expected_path,
                        "expected_header": row["expected"]["header"],
                        "expected_observations": row["expected_observations"],
                        "approximation_bounds": approximation_bounds,
                        "approximation_width": approximation_width,
                    }
                )
            for previous, current in zip(sorted(intervals), sorted(intervals)[1:]):
                if current[0] < previous[1]:
                    fail(f"overlapping Expected signature ranges in {report}")
            programs.append(
                {
                    "scope": scope,
                    "program_id": entry.get("program_id", elf.parent.name),
                    "rel": rel,
                    "elf": elf,
                    "report": report,
                    "signature_size": signature_size,
                    "cases": cases,
                }
            )

    scanned_elfs = {
        relative_path(path, root)
        for path in root.rglob("*.elf")
        if path.is_file()
    }
    if scanned_elfs != expected_elfs:
        fail(
            "recursive ELF scan does not match manifests: "
            f"missing={len(expected_elfs - scanned_elfs)} "
            f"extra={len(scanned_elfs - expected_elfs)}"
        )
    if len(programs) != 1049:
        fail(f"manifests describe {len(programs)} ELF files, expected 1049")
    return checkpoint, programs, expected_elfs


def selected_programs(
    programs: list[dict[str, Any]], group_index: int, group_count: int, only: set[str]
) -> list[dict[str, Any]]:
    if group_count < 1 or not 0 <= group_index < group_count:
        fail("invalid group index/count")
    scopes = ["base_000"] + [f"partition_{index:04d}" for index in range(66)]
    selected_scopes = {
        scope
        for index, scope in enumerate(scopes)
        if index % group_count == group_index
    }
    selected = [program for program in programs if program["scope"] in selected_scopes]
    if only:
        valid_names = {
            name
            for program in programs
            for name in (program["rel"], program["rel"].removesuffix("/program.elf"))
        }
        unknown = only - valid_names
        if unknown:
            fail(f"--only path is not a manifest ELF: {sorted(unknown)}")
        selected = [
            program
            for program in selected
            if program["rel"] in only
            or program["rel"].removesuffix("/program.elf") in only
        ]
    if not selected:
        fail("selection contains no ELF files")
    return selected


def run_program(
    program: dict[str, Any],
    spike: Path,
    output: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    program_out = output / Path(program["rel"]).parent
    program_out.mkdir(parents=True, exist_ok=True)
    actual_hex = program_out / "actual.hex"
    actual_bin = program_out / "actual.bin"
    run_log = program_out / "run.log"
    trace_log = program_out / "trace.log"
    comparison_path = program_out / "comparison.json"
    command = [
        str(spike),
        "-l",
        "--isa=rv64imac_ztt",
        f"+signature={actual_hex}",
        "+signature-granularity=8",
        str(program["elf"]),
    ]
    start = time.monotonic()
    returncode: int | None = None
    timed_out = False
    with run_log.open("w", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    elapsed = time.monotonic() - start
    text = run_log.read_text(encoding="utf-8", errors="replace")
    trace_log.write_text(
        "\n".join(line for line in text.splitlines() if TRACE_RE.search(line)) + "\n",
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "scope": program["scope"],
        "program_id": program["program_id"],
        "elf": program["rel"],
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "case_count": len(program["cases"]),
        "passed_cases": 0,
        "failed_cases": [],
        "status": "FAIL",
    }
    if timed_out or returncode != 0 or not actual_hex.is_file():
        result["failed_cases"] = [
            {
                "case_id": case["case_id"],
                "case_index": case["case_index"],
                "reason": "timeout, non-zero Spike exit, or missing signature",
            }
            for case in program["cases"]
        ]
        evidence = output / "failure-evidence" / Path(program["rel"]).parent
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_log, evidence / "run.log")
        shutil.copy2(trace_log, evidence / "trace.log")
        if actual_hex.is_file():
            shutil.copy2(actual_hex, evidence / "actual.hex")
        write_json(comparison_path, result)
        return result

    try:
        actual = decode_signature(actual_hex)
    except ValidationError as exc:
        result["failed_cases"] = [
            {"case_id": case["case_id"], "case_index": case["case_index"], "reason": str(exc)}
            for case in program["cases"]
        ]
        evidence = output / "failure-evidence" / Path(program["rel"]).parent
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_log, evidence / "run.log")
        shutil.copy2(trace_log, evidence / "trace.log")
        if actual_hex.is_file():
            shutil.copy2(actual_hex, evidence / "actual.hex")
        write_json(comparison_path, result)
        return result
    actual_bin.write_bytes(actual)
    result["actual_size"] = len(actual)
    result["actual_sha256"] = hashlib.sha256(actual).hexdigest()
    if len(actual) != program["signature_size"]:
        result["failed_cases"].append(
            {
                "reason": "program signature length mismatch",
                "expected": program["signature_size"],
                "actual": len(actual),
            }
        )

    trace_counts = expected_trace_counts(text)
    result["trace_instruction_lines"] = sum(trace_counts.values())
    for case in program["cases"]:
        offset = case["signature_offset"]
        size = case["signature_size"]
        expected = case["expected_path"].read_bytes()
        actual_slice = actual[offset : offset + size]
        mismatches = compare_signature(expected, actual_slice, case)
        trace_key = (int(case["target_pc"]), int(case["target_encoding"]))
        actual_executions = trace_counts.get(trace_key, 0)
        reasons: list[str] = []
        if mismatches:
            reasons.append("Expected signature mismatch")
        if actual_executions != int(case["target_executions"]):
            reasons.append(
                f"target trace count {actual_executions} != {case['target_executions']}"
            )
        case_result = {
            "case_id": case["case_id"],
            "case_index": case["case_index"],
            "target_pc": case["target_pc"],
            "target_encoding": case["target_encoding"],
            "expected_executions": case["target_executions"],
            "actual_executions": actual_executions,
            "signature_offset": offset,
            "signature_size": size,
            "expected_sha256": sha256(case["expected_path"]),
            "actual_sha256": hashlib.sha256(actual_slice).hexdigest(),
            "status": "PASS" if not reasons else "FAIL",
        }
        if mismatches:
            case_result["diff"] = mismatches
        if reasons:
            case_result["reasons"] = reasons
            result["failed_cases"].append(case_result)
        else:
            result["passed_cases"] += 1

    result["status"] = "PASS" if not result["failed_cases"] else "FAIL"
    write_json(comparison_path, result)
    if result["failed_cases"]:
        evidence = output / "failure-evidence" / Path(program["rel"]).parent
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_log, evidence / "run.log")
        shutil.copy2(trace_log, evidence / "trace.log")
        shutil.copy2(actual_hex, evidence / "actual.hex")
        shutil.copy2(actual_bin, evidence / "actual.bin")
        write_json(evidence / "comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", default=os.environ.get("ZTT_CI_ASSET_ROOT"))
    parser.add_argument("--spike", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--group-count", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--limit-programs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.asset_root:
        parser.error("--asset-root or ZTT_CI_ASSET_ROOT is required")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    root = Path(args.asset_root).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    summary: dict[str, Any] = {
        "status": "FAIL",
        "asset_root": str(root),
        "asset_version": {},
        "group_index": args.group_index,
        "group_count": args.group_count,
        "commit": os.environ.get("GITHUB_SHA", "unknown"),
        "spike": str(args.spike) if args.spike else None,
        "selected_programs": 0,
        "selected_cases": 0,
        "passed_programs": 0,
        "failed_programs": [],
        "coverage_obligation": {
            "selected_scope_complete": False,
            "all_asset_scope_complete": False,
            "note": "A matrix group is a selected regression scope, not the full 67-manifest obligation.",
        },
        "errors": [],
    }
    try:
        if not root.is_dir():
            fail(f"asset root is not a directory: {root}")
        checkpoint, programs, _ = discover_programs(root)
        summary["asset_version"] = {
            "checkpoint_sha256": sha256(root / "checkpoint.json"),
            "catalog_sha256": sha256(root / "targets.jsonl"),
            "schema": checkpoint.get("schema"),
            "status": checkpoint.get("status"),
            "capability_sha256": (
                checkpoint.get("capability_version", {}).get("capability_sha256")
                if isinstance(checkpoint.get("capability_version"), dict)
                else None
            ),
        }
        only = set(args.only)
        selected = selected_programs(
            programs, args.group_index, args.group_count, only
        )
        if args.limit_programs is not None:
            if args.limit_programs < 1:
                parser.error("--limit-programs must be positive")
            selected = selected[: args.limit_programs]
        summary["selected_programs"] = len(selected)
        summary["selected_cases"] = sum(len(program["cases"]) for program in selected)
        summary["manifest_programs"] = len(programs)
        summary["manifest_partitions"] = 67
        summary["coverage_obligation"]["selected_scope_complete"] = True
        if args.dry_run:
            summary["status"] = "DRY_RUN"
            write_json(output / "summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if not args.spike:
            parser.error("--spike is required unless --dry-run is used")
        spike = args.spike.resolve()
        require_file(spike, "current Spike")
        summary["spike_sha256"] = sha256(spike)
        for program in selected:
            result = run_program(program, spike, output, args.timeout_seconds)
            if result["status"] == "PASS":
                summary["passed_programs"] += 1
            else:
                summary["failed_programs"].append(result)
        summary["status"] = (
            "PASS" if summary["passed_programs"] == summary["selected_programs"] else "FAIL"
        )
        summary["coverage_obligation"]["all_asset_scope_complete"] = (
            args.group_count == 1
            and summary["selected_programs"] == len(programs)
        )
    except (ValidationError, OSError, subprocess.SubprocessError) as exc:
        summary["errors"].append(str(exc))
        summary["status"] = "FAIL"
    summary["elapsed_seconds"] = round(time.time() - run_started, 3)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in ("PASS", "DRY_RUN") else 1


if __name__ == "__main__":
    sys.exit(main())
