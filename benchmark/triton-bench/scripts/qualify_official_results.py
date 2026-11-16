#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independently qualify whole operators and freeze 50 only after real NPU evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
import logging

import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from check_run_completeness import audit_results
from triton_eval.corpus_assets import freeze_assets, reference_sources, verify_assets
from triton_eval.corpusmod import build_manifest, load_manifest, verify_corpus_files, write_manifest
from triton_eval.coverage import coverage_stats
from triton_eval.official_catalog import BY_NAME, CANDIDATES, COMPILER_ONLY
from triton_eval.specload import load_feature_atoms
from triton_eval.console import emit


ENVIRONMENT_KEYS = (
    "arch",
    "npu_model",
    "cann_version",
    "compiler_name",
    "compiler_version",
    "compiler_commit",
    "torch_version",
    "torch_npu_version",
    "triton_import_version",
    "venv_lock_hash",
    "toolchain_commit",
)


def _unhealthy_device_environment(environment):
    if (environment.get("env_health") != "ok"):
        return True
    if (not environment.get("npu_model")):
        return True
    if (not environment.get("cann_version")):
        return True
    return bool(environment.get("npu_count", 0) < 1)


def _unhealthy_run_integrity(integrity, child):
    if (integrity["run_status"] != "OK"):
        return True
    if (integrity["env_mismatches"]):
        return True
    if (integrity["submission_changed_during_run"]):
        return True
    if (child.get("exit_code") != 0):
        return True
    if (child.get("signal")):
        return True
    return bool(child.get("killed_by_us"))


def _incomplete_operator_run(files, audit):
    if (len(files) != 1):
        return True
    if (not audit["complete"]):
        return True
    if (not audit["all_pass"]):
        return True
    return bool(audit["expected_cases"] != 20)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_operators(eligible, count=50, *, cases=None):
    if cases is not None:
        # Only whole operators that have already passed all evidence gates enter
        # this pool. Family diversity comes first, then marginal label coverage.
        atoms = {candidate.name: _operator_atoms(cases, {candidate.name}) for candidate in CANDIDATES}
        remaining = _eligible_candidates(eligible)
        selected, covered, families = [], set(), set()
        order = {candidate.name: index for index, candidate in enumerate(CANDIDATES)}
        while remaining and len(selected) < count:
            candidate = min(
                remaining,
                key=lambda item: (
                    -int((item.category.value, item.family) not in families),
                    -len(atoms[item.name] - covered),
                    order[item.name],
                ),
            )
            selected.append(candidate.name)
            covered.update(atoms[candidate.name])
            families.add((candidate.category.value, candidate.family))
            remaining.remove(candidate)
        return selected
    # Historical corpora keep their original selection policy.
    buckets = defaultdict(list)
    for candidate in CANDIDATES:
        if candidate.name in eligible and candidate.name not in COMPILER_ONLY:
            buckets[(candidate.category.value, candidate.family)].append(candidate.name)
    selected = []
    while len(selected) < count and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
    return selected


def audit_operator(payload, operator, partition, recorded_summary):
    expected = dict(payload, cases=[case for case in payload["cases"] if case["operator"] == operator])
    files = list(partition.glob("run-*/results.jsonl"))
    lines = files[0].read_text(encoding="utf-8").splitlines() if len(files) == 1 else []
    audit = audit_results(expected, lines)
    reasons = []
    if _incomplete_operator_run(files, audit):
        reasons.append("requires exactly 20 complete PASS records in one operator run")
    if recorded_summary.get("reference_gate_returncodes") != {"cpu": 0, "npu:0": 0}:
        reasons.append("missing successful independent CPU and NPU reference gates")
    if any(recorded_summary.get(key) != 0 for key in ("run_returncode", "score_returncode", "report_returncode")):
        reasons.append("run, scoring or report command failed or is missing")
    identity = None
    if len(files) == 1:
        try:
            integrity = json.loads(files[0].with_name("integrity.json").read_text(encoding="utf-8"))
            environment = json.loads(files[0].with_name("env.json").read_text(encoding="utf-8"))
            child = integrity["child"]
            if (
                _unhealthy_run_integrity(integrity, child)
            ):
                reasons.append("run integrity checks failed")
            if (
                _unhealthy_device_environment(environment)
            ):
                reasons.append("missing healthy real NPU environment")
            required_identity = (
                "arch",
                "compiler_name",
                "compiler_version",
                "triton_import_version",
                "torch_version",
                "torch_npu_version",
                "venv_lock_hash",
            )
            if any(not environment.get(key) for key in required_identity):
                reasons.append("missing compiler/runtime environment identity")
            identity = {key: environment.get(key) for key in ENVIRONMENT_KEYS}
            evidence_paths = (files[0], files[0].with_name("integrity.json"), files[0].with_name("env.json"))
            audit["evidence_sha256"] = {str(path.relative_to(partition)): digest(path) for path in evidence_paths}
        except (OSError, ValueError, KeyError, TypeError) as error:
            reasons.append(f"invalid integrity/environment evidence: {error}")
    audit.update(eligible=not reasons, ineligibility_reasons=reasons, environment=identity)
    return audit


def qualify(corpus, submission, results):
    manifest, expected_snapshot, suite = _verify_qualification_inputs(corpus, submission, results)

    payload = manifest.model_dump()
    reports = _operator_reports(payload, results, suite)
    environments = defaultdict(list)
    for name, report in reports.items():
        if report["eligible"]:
            environments[json.dumps(report["environment"], sort_keys=True)].append(name)
    countable = _countable_environments(environments)
    eligible = max(countable.values(), key=lambda names: (len(names), sorted(names)), default=[])
    catalogue = json.loads((corpus / "official-catalog.json").read_text())
    coverage_selection = catalogue.get("generation", {}).get("selection_policy") == "bounded-marginal-v2"
    selected = choose_operators(set(eligible), cases=manifest.cases if coverage_selection else None)
    enough = len(selected) == 50
    report = {
        "format": 1,
        "candidate_manifest_hash": manifest.manifest_hash,
        "target_scope": "50 whole operators / 1000 functionality cases; not full taskbook acceptance",
        "eligible_count": len(eligible),
        "target_met": enough,
        "selected_operators": selected,
        "compiler_only_passed": sorted(name for name in COMPILER_ONLY if reports[name]["eligible"]),
        "selection_policy": (
            "direct CANN correspondence; whole operator 20/20 PASS; one environment; category/family "
            "round-robin in pinned catalogue order; compiler-only primitives excluded"
        ),
        "submission_sha256": expected_snapshot,
        "operators": reports,
        "mixed_environments": len(environments) > 1,
    }
    if coverage_selection:
        report["selection_policy"] = (
            "direct CANN correspondence; whole operator 20/20 PASS; one environment; "
            "category/family diversity then marginal label coverage; compiler-only primitives excluded"
        )
        active = {atom["id"] for atom in load_feature_atoms()["atoms"] if atom.get("status", "active") == "active"}
        reachable = _operator_atoms(manifest.cases, eligible) & active
        covered = _operator_atoms(manifest.cases, selected) & active
        report["eligible_reachable_atoms"] = sorted(reachable)
        report["lost_eligible_atoms"] = sorted(reachable - covered)
    if enough:
        _freeze_qualified_corpus(corpus, results, manifest, selected, report)
    (results / "qualification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _operator_atoms(cases, operators):
    atoms = set()
    for case in cases:
        if case.operator in operators:
            atoms.update(case.feature_atoms)
    return atoms


def _eligible_candidates(eligible):
    candidates = []
    for candidate in CANDIDATES:
        if candidate.name in eligible and candidate.name not in COMPILER_ONLY:
            candidates.append(candidate)
    return candidates


def _submission_snapshot(submission):
    snapshot = {}
    for path in sorted(submission.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
            snapshot[path.relative_to(submission).as_posix()] = digest(path)
    return snapshot


def _operator_reports(payload, results, suite):
    reports = {}
    for candidate in CANDIDATES:
        previous = suite.get("operators", {}).get(candidate.name, {})
        reports[candidate.name] = audit_operator(payload, candidate.name, results / candidate.name, previous)
    return reports


def _countable_environments(environments):
    return {identity: _countable_names(names) for identity, names in environments.items()}


def _countable_names(names):
    return [name for name in names if name not in COMPILER_ONLY]


def _copy_selected_references(corpus, destination, operators):
    selected = {Path(operator.ref_path).parent for operator in operators}
    for relative, source in reference_sources(corpus).items():
        logical = Path(relative)
        if logical.parent in selected:
            target = destination / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def _freeze_qualified_corpus(corpus, results, manifest, selected, report):
    destination = results / "frozen-official50"
    if destination.exists():
        raise ValueError("frozen output already exists; refusing to overwrite qualification evidence")
    destination.mkdir()
    cases = [case for case in manifest.cases if case.operator in selected]
    operators = [operator for operator in manifest.operators if operator.name in selected]
    _copy_selected_references(corpus, destination, operators)
    catalogue = json.loads((corpus / "official-catalog.json").read_text())
    catalogue["operators"] = [operator for operator in catalogue["operators"] if operator["name"] in selected]
    frozen_version = manifest.version.removesuffix("-candidates") + "50-frozen"
    catalogue.update(version=frozen_version, status="NPU_QUALIFIED_FOR_RECORDED_ENVIRONMENT")
    (destination / "official-catalog.json").write_text(
        json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    frozen = build_manifest(
        destination,
        frozen_version,
        manifest.spec_version,
        operators,
        cases,
        feature_atoms_version=manifest.feature_atoms_version,
    )
    if len(cases) != 1000:
        raise ValueError("qualification must freeze exactly 1000 cases")
    write_manifest(frozen, destination / "manifest.json")
    freeze_assets(destination, frozen.manifest_hash)
    active = [atom["id"] for atom in load_feature_atoms()["atoms"] if atom.get("status", "active") == "active"]
    report["frozen_coverage"] = coverage_stats(cases, active)
    report["label_coverage_target_met"] = report["frozen_coverage"]["rate"] >= 0.9
    report["coverage_basis"] = "frozen semantic labels, not executed Triton API coverage certification"
    report["frozen_manifest_hash"] = frozen.manifest_hash
    (destination / "qualification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    report = qualify(args.corpus.resolve(), args.submission.resolve(), args.results.resolve())
    emit(
        f"NPU-qualified whole operators: {report['eligible_count']}; 50/1000 target met: {report['target_met']}"
    )
    return 0 if report["target_met"] else 1




def _verify_qualification_inputs(corpus, submission, results):
    manifest = load_manifest(corpus / "manifest.json")
    verify_corpus_files(corpus, manifest)
    verify_assets(corpus, manifest.manifest_hash)
    archived = load_manifest(results / "manifest.json")
    verify_corpus_files(corpus, archived)
    if archived.manifest_hash != manifest.manifest_hash:
        raise ValueError("result corpus differs from the pinned candidate corpus")
    if json.loads((results / "evaluation-assets.json").read_text()) != json.loads(
        (corpus / "evaluation-assets.json").read_text()
    ):
        raise ValueError("result evaluation assets differ from current candidate assets")
    expected_snapshot = _submission_snapshot(submission)
    archived_snapshot = json.loads((results / "submission-sha256.json").read_text())
    if archived_snapshot != expected_snapshot:
        raise ValueError("submission changed since the candidate run; rerun before qualification")
    suite = json.loads((results / "suite-summary.json").read_text())
    if suite["manifest_hash"] != manifest.manifest_hash:
        raise ValueError("suite summary is from another corpus")
    return manifest, expected_snapshot, suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
