#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Build downloadable source packages from the current v19 reference submission.

This performs static checks and deterministic packing, never device execution.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile
import yaml
from triton_eval.console import emit

from prepare_half_experiment import ROOT, prepare
from triton_eval.fixture_sources import fixture_path


def _archive_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def build(destination=ROOT / "frontend/public/examples"):
    destination = Path(destination)
    source = fixture_path(ROOT / "examples/coverage-v19-ascend-submission")
    with tempfile.TemporaryDirectory(prefix="platform-examples-") as temporary:
        prepared = Path(temporary) / "half"
        plan = prepare(prepared)
        half = (prepared / "submission.zip").read_bytes()
    with zipfile.ZipFile(io.BytesIO(half)) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    full_manifest = yaml.safe_load((source / "manifest.yaml").read_text())
    full_manifest["submit_id"] = "v19-full-74-source"
    entries["manifest.yaml"] = yaml.safe_dump(full_manifest, sort_keys=False).encode()
    full = _archive_bytes(entries)
    from triton_eval.corpusmod import load_manifest

    manifest = load_manifest(ROOT / "corpus" / plan["corpus"] / "manifest.json")
    packages = [
        ("v19-half-37.zip", half, plan["operators"], plan["cases"], "static-preflight-only"),
        (
            "v19-full-74.zip",
            full,
            sorted(o["name"] for o in full_manifest["operators"]),
            len(manifest.cases),
            "static-preflight-only",
        ),
    ]
    abs_entry = next(item for item in full_manifest["operators"] if item["name"] == "Abs")
    abs_manifest = dict(submit_id="v19-abs-structure", operators=[abs_entry])
    minimal = _archive_bytes(
        {
            "manifest.yaml": yaml.safe_dump(abs_manifest, sort_keys=False).encode(),
            abs_entry["path"]: (source / abs_entry["path"]).read_bytes(),
        }
    )
    packages.append(
        (
            "abs-smoke.zip",
            minimal,
            ["Abs"],
            sum(case.operator == "Abs" for case in manifest.cases),
            "static-preflight-only",
        )
    )
    return _write_packages(destination, packages, plan)


def _write_packages(destination, packages, plan):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, data, operators, cases, evidence in packages:
        (destination / name).write_bytes(data)
        records.append(
            dict(
                filename=name,
                operators=operators,
                operator_count=len(operators),
                case_count=cases,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                corpus=plan["corpus"],
                manifest_hash=plan["manifest_hash"],
                validation=evidence,
            )
        )
    (destination / "packages.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    return records


if __name__ == "__main__":
    emit(json.dumps(build(), ensure_ascii=False, indent=2))
