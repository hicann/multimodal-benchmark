# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Freeze v19 only after reproducible selection and independent reference gates."""

from __future__ import annotations
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from triton_eval.workload_generation import VERSION, PARENT, PARENT_HASH, TARGETS, select_completion
from triton_eval.corpusmod import (
    load_manifest,
    verify_manifest,
    compute_case_content_hash,
    verify_corpus_files,
)

from triton_eval import PROJECT_ROOT

ROOT = PROJECT_ROOT
CORPUS = ROOT / "corpus" / VERSION
SUBMISSION = ROOT / "examples/coverage-v19-ascend-submission"


def parent_manifest():
    from triton_eval.corpus_assets import verify_assets

    directory = ROOT / "corpus" / PARENT
    manifest = load_manifest(directory / "manifest.json")
    verify_manifest(manifest)
    if manifest.manifest_hash != PARENT_HASH:
        raise ValueError("parent manifest identity differs")
    verify_corpus_files(directory, manifest)
    verify_assets(directory, manifest.manifest_hash)
    return manifest


def _active_atoms():
    from triton_eval.specload import load_feature_atoms

    return sorted(
        atom["id"]
        for atom in load_feature_atoms()["atoms"]
        if atom.get("status", "active") == "active"
    )


def _check_release_identity(manifest, parent, expected):
    fields = ("operators", "excluded", "spec_version", "feature_atoms_version")
    if manifest.version != VERSION or any(
        getattr(manifest, key) != getattr(parent, key) for key in fields
    ):
        raise ValueError("release identity/catalogue differs")
    if len(manifest.cases) != len(expected):
        raise ValueError("retained or selected case content differs")
    return manifest


def _check_case_content(corpus, manifest, expected):
    for retained, selected in zip(manifest.cases, expected):
        if retained.code != selected.code or retained.content_hash != compute_case_content_hash(
            corpus, selected
        ):
            raise ValueError("retained or selected case content differs")


def _check_selection_trace(corpus, active, selection):
    plan = json.loads((corpus / "coverage-plan.json").read_text())
    if plan.get("parent_manifest_hash") != PARENT_HASH or plan.get("active_atoms") != active:
        raise ValueError("selection trace/denominator differs")
    if any(plan.get(key) != value for key, value in selection.items()):
        raise ValueError("selection trace/denominator differs")


def _check_reference_gate(corpus, manifest):
    from triton_eval.buildlog import read_log

    events = read_log(corpus / "generation-log.yaml")
    starts = [index for index, event in enumerate(events) if event.get("step") == "build_start"]
    attempt = events[starts[-1]:] if starts else []
    if not attempt:
        raise ValueError("missing successful complete CPU reference gate")
    ending = attempt[-1]
    if ending.get("manifest_hash") != manifest.manifest_hash or ending.get("step") != "build_end":
        raise ValueError("missing successful complete CPU reference gate")
    gates = [event for event in attempt if event.get("step") == "independent_cpu_reference_gate"]
    if len(gates) != 1:
        raise ValueError("missing successful complete CPU reference gate")
    if gates[0].get("failures") != {} or gates[0].get("cases") != len(manifest.cases):
        raise ValueError("missing successful complete CPU reference gate")


def static_readiness(corpus):
    from triton_eval.corpus_assets import verify_assets
    from triton_eval.coverage import coverage_stats

    corpus = Path(corpus)
    manifest = load_manifest(corpus / "manifest.json")
    verify_manifest(manifest)
    verify_corpus_files(corpus, manifest)
    verify_assets(corpus, manifest.manifest_hash)
    parent = parent_manifest()
    added, selection = select_completion(parent.cases)
    _check_release_identity(manifest, parent, [*parent.cases, *added])
    _check_case_content(corpus, manifest, [*parent.cases, *added])
    active = _active_atoms()
    _check_selection_trace(corpus, active, selection)
    _check_reference_gate(corpus, manifest)
    counts = Counter(case.operator for case in manifest.cases)
    if any(counts[name] != 20 for name in TARGETS):
        raise ValueError("extension workload count not twenty")
    stats = coverage_stats(manifest.cases, active)
    if len(active) != 71 or len(stats["covered"]) != 71:
        raise ValueError("static coverage drift")
    return dict(
        stats,
        version=VERSION,
        manifest_hash=manifest.manifest_hash,
        cases=len(manifest.cases),
        preserved_cases=len(parent.cases),
        added_cases=len(added),
        additions_by_operator=selection["additions_by_operator"],
        static_target_met=True,
        compilation_state="NOT_RUN",
        execution_state="NOT_RUN",
        agreement_state="AWAITING_JOINT_CONFIRMATION",
    )


def _prepare_release_directories(corpus, submission):
    import yaml
    from triton_eval.fixture_sources import fixture_path

    if corpus.exists() and any(corpus.iterdir()):
        raise ValueError("refuse to overwrite nonempty draft; inspect failed gate first")
    if submission.exists():
        raise ValueError("refuse to overwrite existing submission")
    corpus.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "corpus" / PARENT / "reference-layout.json", corpus / "reference-layout.json"
    )
    catalog = ROOT / "corpus" / PARENT / "official-catalog.json"
    if catalog.exists():
        shutil.copyfile(catalog, corpus / catalog.name)
    shutil.copytree(fixture_path(ROOT / "examples/coverage-v18-ascend-submission"), submission)
    payload = yaml.safe_load((submission / "manifest.yaml").read_text())
    payload["submit_id"] = VERSION
    (submission / "manifest.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_release_plan(corpus, selection):
    plan = dict(
        selection,
        version=VERSION,
        parent_version=PARENT,
        parent_manifest_hash=PARENT_HASH,
        active_atoms=_active_atoms(),
    )
    (corpus / "coverage-plan.json").write_text(json.dumps(plan, indent=2) + "\n")


def _run_reference_gate(corpus, submission, cases, log):
    from triton_eval.submission_preflight import require_submission_ready
    from triton_eval.refcheck import check_all

    log.step(
        "submission_source_contract_gate",
        **require_submission_ready(submission, set(case.operator for case in cases)),
    )
    failures = check_all(cases, "cpu", corpus_dir=corpus)
    log.step("independent_cpu_reference_gate", cases=len(cases), failures=failures)
    if failures:
        raise ValueError(f"independent CPU reference gate failed: {failures}")


def build(corpus=CORPUS, submission=SUBMISSION):
    from triton_eval.buildlog import GenerationLog
    from triton_eval.corpusmod import build_manifest, write_manifest
    from triton_eval.corpus_assets import freeze_assets
    from triton_eval.submission_preflight import require_submission_ready

    corpus, submission = Path(corpus), Path(submission)
    if (corpus / "manifest.json").exists():
        result = static_readiness(corpus)
        require_submission_ready(
            submission, set(case.operator for case in load_manifest(corpus / "manifest.json").cases)
        )
        return result
    parent = parent_manifest()
    added, selection = select_completion(parent.cases)
    cases = [*parent.cases, *added]
    _prepare_release_directories(corpus, submission)
    _write_release_plan(corpus, selection)
    log = GenerationLog(corpus / "generation-log.yaml")
    log.step("build_start", version=VERSION, selection=selection)
    _run_reference_gate(corpus, submission, cases, log)
    manifest = build_manifest(
        corpus,
        VERSION,
        parent.spec_version,
        parent.operators,
        cases,
        feature_atoms_version=parent.feature_atoms_version,
        excluded=parent.excluded,
    )
    write_manifest(manifest, corpus / "manifest.json")
    log.step(
        "build_end", manifest_hash=manifest.manifest_hash, cases=len(cases), manifest_written=True
    )
    freeze_assets(corpus, manifest.manifest_hash)
    report = static_readiness(corpus)
    (corpus / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
