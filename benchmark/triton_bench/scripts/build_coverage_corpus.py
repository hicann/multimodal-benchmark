#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build v0.16 with unchanged cases and strict physical-memory pattern evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import logging
from triton_eval.console import emit

from triton_eval.buildlog import GenerationLog
from triton_eval.corpus_assets import copy_reference_tree, freeze_assets
from triton_eval.corpusmod import (
    build_manifest,
    compute_case_content_hash,
    load_manifest,
    verify_manifest,
    write_manifest,
)
from triton_eval.coverage_gate import BASELINE_HASH, static_readiness
from triton_eval.coverage_generation import (
    EXTENSIONS,
    VERSION,
    extension_pool,
    select_extensions,
)
from triton_eval.models import CaseSource, OperatorSpec
from triton_eval.operator_registry import by_name
from triton_eval.specload import load_feature_atoms, load_spec

from triton_eval.fixture_sources import fixture_path


def build(corpus, submission):
    corpus, submission = Path(corpus), Path(submission)
    from triton_eval.submission_preflight import require_submission_ready

    if (corpus / "manifest.json").exists():
        report = static_readiness(corpus)
        require_submission_ready(submission, {c.operator for c in load_manifest(corpus / "manifest.json").cases})
        return report
    parent_dir = ROOT / "corpus/v0.9-official-candidates"
    parent = load_manifest(parent_dir / "manifest.json")
    verify_manifest(parent)  # Historical framework assets are deliberately not reused.
    if parent.manifest_hash != BASELINE_HASH or any(
        c.content_hash != compute_case_content_hash(parent_dir, c) for c in parent.cases
    ):
        raise ValueError("baseline identity/content mismatch")
    active = sorted(a["id"] for a in load_feature_atoms()["atoms"] if a.get("status", "active") == "active")
    added, selection = select_extensions(parent.cases, extension_pool(), active)
    corpus.mkdir(parents=True, exist_ok=True)
    copy_reference_tree(parent_dir, corpus)
    _prepare_extension_references(corpus)

    _write_extension_plan(corpus, selection, active, parent)

    _prepare_extension_submission(submission)

    cases = [*parent.cases, *added]
    entry_report = require_submission_ready(submission, {c.operator for c in cases})
    # CPU validation is mandatory before publishing any manifest. No failure is
    # filtered out to improve coverage or to keep the builder green.
    from triton_eval.refcheck import check_all

    log = GenerationLog(corpus / "generation-log.yaml")
    log.step("build_start", version=VERSION, selection=selection)
    log.step("submission_source_contract_gate", **entry_report)
    failures = check_all(cases, "cpu", corpus_dir=corpus)
    log.step("independent_cpu_reference_gate", cases=len(cases), failures=failures)
    if failures:
        raise ValueError(f"independent reference gate failed: {failures}")
    operators = _extension_operators(parent)

    manifest = build_manifest(
        corpus,
        VERSION,
        load_spec().spec_version,
        operators,
        cases,
        feature_atoms_version=str(load_feature_atoms()["version"]),
    )
    write_manifest(manifest, corpus / "manifest.json")
    log.step("build_end", manifest_hash=manifest.manifest_hash, cases=len(cases), manifest_written=True)
    freeze_assets(corpus, manifest.manifest_hash)
    report = static_readiness(corpus)
    (corpus / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus" / VERSION)
    parser.add_argument("--submission", type=Path, default=ROOT / "examples/coverage-v16-ascend-submission")
    args = parser.parse_args()
    report = build(args.corpus.resolve(), args.submission.resolve())
    emit(json.dumps(report, ensure_ascii=False, indent=2))
    return 0




def _prepare_extension_references(corpus):
    for name in EXTENSIONS:
        directory = corpus / "operators" / by_name(name).entry
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "ref.py").write_text(
            f"from triton_eval.coverage_reference import bind\n\nreference, make_inputs = bind({name!r})\n"
        )
        (directory / "ref_cpu.py").write_text(
            f"from triton_eval.coverage_reference import bind_cpu\n\nreference = bind_cpu({name!r})\n"
        )


def _write_extension_plan(corpus, selection, active, parent):
    plan = dict(
        selection,
        format=1,
        active_atoms=active,
        baseline_manifest_hash=parent.manifest_hash,
        baseline_case_hashes={c.code: c.content_hash for c in parent.cases},
        sources={
            "baseline": "v0.9-official-candidates",
            "extensions": (
                "registered operator contracts; independent Torch/NumPy references; not official-source "
                "operator admissions"
            ),
        },
    )
    (corpus / "coverage-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")


def _prepare_extension_submission(submission):
    # A separate submission preserves the historical official example unchanged.
    shutil.copytree(fixture_path(ROOT / "examples/official-ascend-submission"), submission, dirs_exist_ok=True)
    shutil.copyfile(
        fixture_path(ROOT / "scripts/templates/coverage_kernels.py"),
        submission / "operators/coverage_kernels.py",
    )
    import yaml

    payload = yaml.safe_load((submission / "manifest.yaml").read_text())
    for name in EXTENSIONS:
        if name == "Transpose":
            continue  # Existing layout kernel preserves integer bits.
        payload["operators"].append(dict(name=name, entry=by_name(name).entry, path="operators/coverage_kernels.py"))
    for name in ("MaskedSoftmax", "ApplyRotaryPosEmb"):
        entry = by_name(name).entry
        target = submission / f"operators/{entry}/kernel.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture_path(ROOT / f"examples/demo-submission/operators/{entry}/kernel.py"), target)
        next(o for o in payload["operators"] if o["name"] == name)["path"] = f"operators/{entry}/kernel.py"
    shutil.copyfile(
        fixture_path(ROOT / "scripts/templates/implementation_witness_kernels.py"),
        submission / "operators/implementation_witness_kernels.py",
    )
    for name in ("Neg", "MaskedSoftmax"):
        next(o for o in payload["operators"] if o["name"] == name)["path"] = (
            "operators/implementation_witness_kernels.py"
        )
    shutil.copyfile(
        fixture_path(ROOT / "scripts/templates/compatibility_kernels.py"),
        submission / "operators/compatibility_kernels.py",
    )
    for name in ("FloorDiv", "GroupedMatmul"):
        next(o for o in payload["operators"] if o["name"] == name)["path"] = "operators/compatibility_kernels.py"
    shutil.copyfile(
        fixture_path(ROOT / "scripts/templates/stability_kernels.py"),
        submission / "operators/stability_kernels.py",
    )
    next(entry for entry in payload["operators"] if entry["name"] == "CrossEntropyLoss")["path"] = (
        "operators/stability_kernels.py"
    )
    payload["submit_id"] = VERSION
    (submission / "manifest.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))




def _extension_operators(parent):
    operators = list(parent.operators)
    existing = {o.name for o in operators}
    for name in EXTENSIONS:
        if name not in existing:
            operators.append(
                OperatorSpec(
                    name=name,
                    source=CaseSource.VENDOR_SELFTEST,
                    weight=1.0,
                    ref_path=f"operators/{by_name(name).entry}/ref.py",
                    desc="Coverage extension; not an official-source admission",
                )
            )
    return operators


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
