#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Package new implementations for the existing frozen v19 cases; never generate cases."""

from __future__ import annotations


import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from triton_eval.console import emit
from triton_eval.corpusmod import load_manifest, verify_corpus_files
from triton_eval.submission_preflight import require_submission_ready

CORPUS = ROOT / "corpus/v0.19-workload-completion-candidates"
SOURCE = ROOT / "examples/independent-20-submission"


def package(source, output, *, submit_id):
    source, output = Path(source), Path(output)
    declarations = yaml.safe_load((source / "manifest.yaml").read_text())
    declarations["submit_id"] = submit_id
    operators = {item["name"] for item in declarations["operators"]}
    if len(operators) != 20 or len(declarations["operators"]) != 20:
        raise ValueError("exactly twenty distinct operator declarations required")
    readiness = require_submission_ready(source, operators)
    files = {"manifest.yaml": yaml.safe_dump(declarations, sort_keys=False).encode()}
    for name in sorted({item["path"] for item in declarations["operators"]}):
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("source path escapes submission")
        files[name] = path.read_bytes()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 14, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != stream.getvalue():
        raise ValueError("refuse to overwrite a previous revision")
    output.write_bytes(stream.getvalue())
    return dict(
        sha256=hashlib.sha256(stream.getvalue()).hexdigest(),
        bytes=len(stream.getvalue()),
        source_files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        readiness=readiness,
        operators=sorted(operators),
    )


def prepare(destination, *, source=SOURCE, corpus=CORPUS, revision="r1"):
    destination, corpus = Path(destination), Path(corpus).resolve()
    manifest = load_manifest(corpus / "manifest.json")
    verify_corpus_files(corpus, manifest)
    before = (corpus / "manifest.json").read_bytes()
    destination.mkdir(parents=True)
    info = package(source, destination / f"submission-{revision}.zip", submit_id=f"independent-20-v19-{revision}")
    selected = [case for case in manifest.cases if case.operator in info["operators"]]
    expected = dict(
        corpus=str(corpus.relative_to(ROOT)),
        corpus_version=manifest.version,
        manifest_hash=manifest.manifest_hash,
        assets_hash=json.loads((corpus / "evaluation-assets.json").read_text())["assets_hash"],
        package=info,
        operators=info["operators"],
        cases=len(selected),
        cases_by_operator=dict(Counter(c.operator for c in selected)),
        case_codes=[case.code for case in selected],
        expected_status_counts={"PASS": len(selected)},
        profile="standard",
        case_policy="all existing frozen cases of declared operators; no new cases",
    )
    (destination / "expectations.json").write_text(json.dumps(expected, indent=2) + "\n")
    if (corpus / "manifest.json").read_bytes() != before:
        raise ValueError("frozen manifest changed while preparing submission")
    return expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--revision", default="r1")
    args = parser.parse_args()
    value = prepare(args.out, revision=args.revision)
    emit(json.dumps({k: value[k] for k in ("corpus_version", "cases", "manifest_hash", "assets_hash")}))
