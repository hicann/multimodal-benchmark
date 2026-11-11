# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A published catalogue may be repaired only to its recorded original bytes."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "repair_official_results", Path(__file__).resolve().parents[2] / "scripts/repair_official_results.py")
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


def evidence(tmp_path):
    candidate, evidence_root = tmp_path / "candidate", tmp_path / "evidence"
    candidate.mkdir()
    frozen = evidence_root / "frozen-official50"
    frozen.mkdir(parents=True)
    catalogue = {"version": "v0.8-official-candidates", "status": "CANDIDATE_NOT_NPU_QUALIFIED",
                 "sources": {"repository": "https://github.com/triton-lang/triton-ascend"},
                 "operators": [{"name": "Abs"}, {"name": "Add"}]}
    raw = (json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n").encode()
    (candidate / "official-catalog.json").write_bytes(raw)
    (evidence_root / "evaluation-assets.json").write_text(json.dumps({"manifest_hash": "candidate-hash", "files": {
        "@corpus/official-catalog.json": hashlib.sha256(raw).hexdigest()}}))
    (evidence_root / "qualification.json").write_text(json.dumps({"candidate_manifest_hash": "candidate-hash",
                                                            "selected_operators": ["Abs"]}))
    version = "v0.8-official50-frozen"
    (frozen / "manifest.json").write_text(json.dumps({"version": version, "manifest_hash": "frozen-hash",
                                                     "operators": [{"name": "Abs"}]}))
    catalogue.update(version=version, status="NPU_QUALIFIED_FOR_RECORDED_ENVIRONMENT", operators=[{"name": "Abs"}])
    raw = (json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n").encode()
    (frozen / "evaluation-assets.json").write_text(json.dumps({"files": {
        "@corpus/official-catalog.json": hashlib.sha256(raw).hexdigest(),
        "@corpus/ref.py": hashlib.sha256(b"original-reference").hexdigest()}}))
    (frozen / "ref.py").write_bytes(b"original-reference")
    (frozen / "official-catalog.json").write_text('{"repository": "<remote-url-redacted>\n')
    return candidate, evidence_root, raw


def test_repair_restores_original_hash_without_changing_published_evidence(tmp_path):
    candidate, published, raw = evidence(tmp_path)
    destination = tmp_path / "recovered"
    receipt = REPAIR.repair(candidate, published, destination)
    assert (destination / "official-catalog.json").read_bytes() == raw
    assert receipt["recovered_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (published / "frozen-official50/official-catalog.json").read_bytes() != raw
    with pytest.raises(ValueError, match="拒绝覆盖"):
        REPAIR.repair(candidate, published, destination)


@pytest.mark.parametrize("changed_file", ["candidate", "reference", "selection"])
def test_repair_rejects_unverifiable_reconstruction(tmp_path, changed_file):
    candidate, published, raw = evidence(tmp_path)
    if changed_file == "candidate":
        (candidate / "official-catalog.json").write_text("{}")
    elif changed_file == "reference":
        (published / "frozen-official50/ref.py").write_text("changed-reference")
    else:
        (published / "qualification.json").write_text(json.dumps({"candidate_manifest_hash": "candidate-hash",
                                                                  "selected_operators": ["Add"]}))
    destination = tmp_path / "recovered"
    with pytest.raises(ValueError):
        REPAIR.repair(candidate, published, destination)
    assert not destination.exists()
