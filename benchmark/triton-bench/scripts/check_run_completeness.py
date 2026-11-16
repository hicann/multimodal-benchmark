#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check result coverage independently of numerical correctness and NPU availability."""
from __future__ import annotations

from collections import namedtuple
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
import logging
from triton_eval.console import emit


TERMINAL_STATUSES = frozenset({
    "SPEC_ERROR", "COMPILE_FAIL", "RUNTIME_FAIL", "PASS", "WRONG_RESULT",
    "SHAPE_MISMATCH", "ORACLE_FAIL", "ENV_ABORT", "SUBMISSION_INVALID",
})
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "corpus/v0.6-ascend-ops/manifest.json"
)


def audit_results(manifest: dict, lines: Iterable[str]) -> dict:
    expected = {}
    errors = []
    for case in manifest["cases"]:
        code = case["code"]
        if code in expected:
            errors.append(f"duplicate manifest case: {code}")
        expected[code] = case["operator"]
    if not expected:
        errors.append("manifest contains no cases")

    counts, by_operator, run_ids, unexpected, mismatches, result_count = _audit_result_rows(lines, expected, errors)

    if len(run_ids) > 1:
        errors.append("results contain multiple run IDs")
    missing = sorted(set(expected) - set(counts))
    duplicates = {code: count for code, count in sorted(counts.items()) if count > 1}
    statuses = Counter()
    for counts_for_operator in by_operator.values():
        statuses.update(counts_for_operator)
    complete = not (errors or missing or duplicates or unexpected or mismatches)
    return {
        "corpus_version": manifest.get("version"),
        "manifest_hash": manifest.get("manifest_hash"),
        "expected_cases": len(expected),
        "result_rows": result_count,
        "unique_cases": len(counts),
        "run_ids": sorted(run_ids),
        "complete": complete,
        "all_pass": complete and statuses["PASS"] == len(expected),
        "by_status": dict(sorted(statuses.items())),
        "by_operator": {name: dict(sorted(counts_for_operator.items()))
                        for name, counts_for_operator in sorted(by_operator.items())},
        "missing": missing,
        "duplicates": duplicates,
        "unexpected": sorted(unexpected),
        "operator_mismatches": mismatches,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", required=True, help="results.jsonl path, or - for stdin")
    parser.add_argument("--require-pass", action="store_true",
                        help="also fail if any case did not PASS")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.results == "-":
            summary = audit_results(manifest, sys.stdin)
        else:
            with Path(args.results).open(encoding="utf-8") as result_file:
                summary = audit_results(manifest, result_file)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        emit(f"audit failed: {exc}", file=sys.stderr)
        return 2
    emit(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_pass" if args.require_pass else "complete"] else 1




AuditedRows = namedtuple('AuditedRows', 'counts by_operator run_ids unexpected mismatches result_count')


def _audit_result_rows(lines, expected, errors):
    counts = Counter()
    by_operator = defaultdict(Counter)
    run_ids = set()
    unexpected = set()
    mismatches = []
    result_count = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        result_count += 1
        try:
            record = json.loads(line)
        except ValueError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {line_number}: result must be an object")
            continue
        code = record.get("case_code")
        operator = record.get("operator")
        status = record.get("status")
        run_id = record.get("run_id")
        if not all(isinstance(value, str) and value for value in (code, operator, status)):
            errors.append(f"line {line_number}: missing or invalid case/operator/status")
            continue
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id < 1:
            errors.append(f"line {line_number}: invalid run_id")
        else:
            run_ids.add(run_id)
        if status not in TERMINAL_STATUSES:
            errors.append(f"line {line_number}: unknown terminal status {status}")
        counts[code] += 1
        by_operator[operator][status] += 1
        if code not in expected:
            unexpected.add(code)
        elif expected[code] != operator:
            mismatches.append({"case_code": code, "expected": expected[code], "actual": operator})
    return AuditedRows(counts, by_operator, run_ids, unexpected, mismatches, result_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
