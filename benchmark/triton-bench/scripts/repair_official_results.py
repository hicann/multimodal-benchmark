#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Recover a published frozen catalogue only when it matches its original SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import logging
from triton_eval.console import emit


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def repair(candidate, evidence, destination):
    """Reconstruct public provenance without changing scores or pinned hashes."""
    if destination.exists():
        raise ValueError("恢复目录已存在，拒绝覆盖")
    assets = json.loads((evidence / "evaluation-assets.json").read_text(encoding="utf-8"))
    raw_catalogue = (candidate / "official-catalog.json").read_bytes()
    if digest(raw_catalogue) != assets["files"]["@corpus/official-catalog.json"]:
        raise ValueError("候选来源目录与原运行指纹不同")
    qualification = json.loads((evidence / "qualification.json").read_text(encoding="utf-8"))
    if assets["manifest_hash"] != qualification["candidate_manifest_hash"]:
        raise ValueError("候选资格摘要与资产版本不同")
    frozen = evidence / "frozen-official50"
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    selected = set(qualification["selected_operators"])
    if selected != {operator["name"] for operator in manifest["operators"]}:
        raise ValueError("冻结算子与资格名单不同")
    catalogue = json.loads(raw_catalogue)
    catalogue["operators"] = [operator for operator in catalogue["operators"] if operator["name"] in selected]
    catalogue.update(version=manifest["version"], status="NPU_QUALIFIED_FOR_RECORDED_ENVIRONMENT")
    recovered = (json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    pinned = json.loads((frozen / "evaluation-assets.json").read_text(encoding="utf-8"))
    if digest(recovered) != pinned["files"]["@corpus/official-catalog.json"]:
        raise ValueError("重建目录与原始冻结 SHA256 不同，拒绝恢复")
    # Validate every portable asset before creating any output. Framework hashes
    # still require checking out the original execution code revision.
    for name, expected in pinned["files"].items():
        if not name.startswith("@corpus/"):
            continue
        path = (frozen / name.removeprefix("@corpus/")).resolve()
        if not path.is_relative_to(frozen.resolve()):
            raise ValueError(f"冻结路径越界：{name}")
        raw = recovered if name == "@corpus/official-catalog.json" else path.read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"其他冻结资产损坏：{name}")
    shutil.copytree(frozen, destination)
    (destination / "official-catalog.json").write_bytes(recovered)
    receipt = {
        "format": 1,
        "candidate_manifest_hash": assets["manifest_hash"],
        "frozen_manifest_hash": manifest["manifest_hash"],
        "repaired_file": "official-catalog.json",
        "recovered_sha256": digest(recovered),
        "published_sha256": digest((frozen / "official-catalog.json").read_bytes()),
        "method": "reconstructed from pinned candidate catalogue and selected operators; original SHA256 matched",
        "framework_validation": "requires original execution code revision",
    }
    (destination / "recovery.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-corpus", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    receipt = repair(args.candidate_corpus.resolve(), args.evidence.resolve(), args.out.resolve())
    emit(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
