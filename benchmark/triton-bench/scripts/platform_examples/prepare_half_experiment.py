#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Prepare a deterministic half-operator standard experiment; never start hardware."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from triton_eval.console import emit
from triton_eval.fixture_sources import fixture_path
from triton_eval.corpusmod import load_manifest, verify_corpus_files
from triton_eval.operator_registry import by_name
from triton_eval.submission_preflight import require_submission_ready
from triton_eval.workload_generation import TARGETS, VERSION


def prepare(
    destination,
    *,
    source=ROOT / "examples/coverage-v19-ascend-submission",
    corpus=ROOT / "corpus" / VERSION,
    submit_id="v19-half-standard-repaired",
):
    source, corpus, destination = fixture_path(Path(source)), Path(corpus), Path(destination)
    manifest = load_manifest(corpus / "manifest.json")
    verify_corpus_files(corpus, manifest)
    require_submission_ready(source, {c.operator for c in manifest.cases})
    chosen, names = _choose_operators(manifest)
    operators = sorted(chosen)
    payload = yaml.safe_load((source / "manifest.yaml").read_text())
    payload["submit_id"] = submit_id
    payload["operators"] = [o for o in payload["operators"] if o["name"] in chosen]
    entries = {"manifest.yaml": yaml.safe_dump(payload, sort_keys=False).encode()}
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("submission source escapes root")
        entries[path.relative_to(source).as_posix()] = path.read_bytes()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    plan = dict(
        corpus=manifest.version,
        manifest_hash=manifest.manifest_hash,
        operators=operators,
        operator_count=len(operators),
        total_operators=len(names),
        cases=sum(c.operator in chosen for c in manifest.cases),
        categories=dict(sorted(Counter(by_name(n).category.value for n in operators).items())),
        profile="standard",
        sampling_policy=("six extensions plus Abs/Matmul/ReduceMax; "
                         "deterministic category round robin; all frozen cases"),
        package_sha256=hashlib.sha256(stream.getvalue()).hexdigest(),
        source_files={
            name: hashlib.sha256(data).hexdigest() for name, data in sorted(entries.items()) if name.endswith(".py")
        },
    )
    _write_preparation(destination, stream.getvalue(), plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plan = prepare(args.out)
    emit(
        json.dumps(
            {k: plan[k] for k in ("corpus", "manifest_hash", "operator_count", "cases", "package_sha256")}, indent=2
        )
    )


def _choose_operators(manifest):
    chosen = set(TARGETS) | {"Abs", "Matmul", "ReduceMax"}
    names = {o.name for o in manifest.operators}
    target = math.ceil(len(names) / 2)
    if not chosen <= names or len(chosen) > target:
        raise ValueError("corpus cannot satisfy half-experiment seed policy")
    groups = defaultdict(list)
    for name in sorted(names - chosen):
        groups[by_name(name).category.value].append(name)
    while len(chosen) < target:
        for category in sorted(groups):
            if groups[category] and len(chosen) < target:
                chosen.add(groups[category].pop(0))
    return chosen, names


def _write_preparation(destination, archive, plan):
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "submission.zip": archive,
        "plan.json": (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode(),
    }
    for name, data in outputs.items():
        path = destination / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError("refuse to overwrite a different prepared experiment")
    for name, data in outputs.items():
        path = destination / name
        if not path.exists():
            path.write_bytes(data)


if __name__ == "__main__":
    main()
