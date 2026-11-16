# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Opt-in release construction; never publish without the independent CPU gate."""

from __future__ import annotations

from collections import namedtuple
import ast
import hashlib
import json
import math
import shutil
from pathlib import Path

from types import SimpleNamespace

from . import PROJECT_ROOT
from .buildlog import read_log
from .pattern_generation import (
    PARENT,
    PARENT_MANIFEST_HASH,
    PATTERN_VERSIONS,
    SCAN_REPAIR_VERSION,
    TARGETS,
    VERSION,
    candidate_plans,
    device_bytes,
    extension_pool,
    label_pattern_case,
    select_patterns,
)

from .fixture_sources import fixture_path

ROOT = PROJECT_ROOT
PARENT_CORPUS = ROOT / "corpus" / PARENT
PARENT_SUBMISSION = fixture_path(ROOT / "examples/coverage-v16-ascend-submission")
SUBMISSION = fixture_path(ROOT / "examples/coverage-v17-ascend-submission")
CORPUS = ROOT / "corpus" / VERSION
SCAN_CORPUS = ROOT / "corpus" / SCAN_REPAIR_VERSION
SCAN_SUBMISSION = fixture_path(ROOT / "examples/coverage-v18-ascend-submission")


def _incomplete_generation_attempt(attempt, manifest, gates):
    if (not attempt):
        return True
    if (attempt[-1].get("step") != "build_end"):
        return True
    if (attempt[-1].get("manifest_hash") != manifest.manifest_hash):
        return True
    if (len(gates) != 1):
        return True
    if (gates[0].get("failures") != {}):
        return True
    return bool(gates[0].get("cases") != len(manifest.cases))


def _invalid_parent_plan(plan, manifest, active, parent, parent_cases):
    if (plan.get("version") != manifest.version):
        return True
    if (plan.get("active_atoms") != active):
        return True
    if (plan.get("parent_manifest_hash") != parent.manifest_hash):
        return True
    return bool(plan.get("parent_case_hashes") != parent_cases)


def _changed_parent_contract(manifest, parent):
    if (manifest.spec_version != parent.spec_version):
        return True
    if (manifest.feature_atoms_version != parent.feature_atoms_version):
        return True
    if (manifest.operators != parent.operators):
        return True
    return bool(manifest.excluded != parent.excluded)


def verify_retained_v17(corpus, cases):
    from .corpusmod import compute_case_content_hash, load_manifest, verify_manifest

    previous = load_manifest(CORPUS / "manifest.json")
    verify_manifest(previous)
    if (
        previous.version != VERSION
        or len(previous.cases) != 1385
        or previous.manifest_hash != "95ba03ce7a13bb2e79223e9f710e95c00921499521e0987d036a1c82bc4c3ed1"
    ):
        raise ValueError("not the authenticated v17 release")
    from .corpus_assets import verify_corpus_asset_paths, verify_origin_assets

    assets = verify_origin_assets(CORPUS, "9aa9135b7055c57c92e4c9f045fce8ea02a8559fa9cb05ee612ff04cdd4cf3ad")
    payload = {key: value for key, value in assets.items() if key != "assets_hash"}
    if (
        assets.get("manifest_hash") != previous.manifest_hash
        or hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        != assets["assets_hash"]
    ):
        raise ValueError("v17 asset index identity differs")
    verify_corpus_asset_paths(CORPUS, assets["files"], "v17 portable corpus asset differs")
    current = {case.code: compute_case_content_hash(Path(corpus), case) for case in cases}
    if len(current) != len(cases) or current != {case.code: case.content_hash for case in previous.cases}:
        raise ValueError("v18 must preserve all 1385 v17 case identities")


def verify_parent(manifest):
    if manifest.manifest_hash != PARENT_MANIFEST_HASH or manifest.version != PARENT or len(manifest.cases) != 1377:
        raise ValueError("not the authenticated v16 parent")
    from .corpus_assets import verify_corpus_asset_paths, verify_origin_assets

    assets = verify_origin_assets(PARENT_CORPUS, "4f1688c8cbed58ddfe8e02453734bb3f25d73febbfe8ba70ad2eebe5d52d43ab")
    if (
        assets["manifest_hash"] != PARENT_MANIFEST_HASH
    ):
        raise ValueError("parent asset identity differs")
    payload = {key: value for key, value in assets.items() if key != "assets_hash"}
    if (
        hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        != assets["assets_hash"]
    ):
        raise ValueError("parent asset index differs")
    verify_corpus_asset_paths(PARENT_CORPUS, assets["files"], "parent reference/corpus asset differs")


def submission_source(parent=PARENT_SUBMISSION):
    result = (fixture_path(ROOT / "scripts/templates/target_pattern_kernels.py")).read_text()
    for filename, name in [
        ("kernels.py", "_matmul"),
        ("coverage_kernels.py", "_cummin"),
        ("compatibility_kernels.py", "_grouped"),
    ]:
        nodes = ast.parse((Path(parent) / "operators" / filename).read_text()).body
        node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
        result += "\n\n" + ast.unparse(node) + "\n"
    return result


def preview():
    parent = json.loads((PARENT_CORPUS / "manifest.json").read_text())
    plan = json.loads((PARENT_CORPUS / "coverage-plan.json").read_text())
    candidates = []
    for operator, dtype, shape, attrs in candidate_plans():
        candidate = SimpleNamespace(operator=operator, dtype=dtype, shape=shape, attrs=attrs)
        candidates.append(
            dict(
                operator=operator, dtype=dtype, shape=shape, attrs=attrs, estimated_device_bytes=device_bytes(candidate)
            )
        )
    return dict(
        version=VERSION,
        state="DRAFT_NOT_FROZEN",
        parent_manifest_hash=parent["manifest_hash"],
        retained_parent_cases=len(parent["cases"]),
        candidates=candidates,
        required_targets=sorted(TARGETS),
        active_atoms=plan["active_atoms"],
        projected_cases=len(parent["cases"]) + len(candidates),
        static_validation="NOT_RUN",
        independent_cpu_reference_gate="NOT_RUN",
        compilation_state="NOT_RUN",
        execution_state="NOT_RUN",
        agreement_state="AWAITING_JOINT_CONFIRMATION",
    )


def static_readiness(corpus):
    from .corpus_assets import verify_assets
    from .corpusmod import (
        load_manifest,
        verify_corpus_files,
        verify_manifest,
    )
    from .coverage import coverage_stats

    corpus = Path(corpus)
    manifest = load_manifest(corpus / "manifest.json")
    if manifest.version not in PATTERN_VERSIONS:
        raise ValueError("wrong pattern release version")
    verify_manifest(manifest)
    verify_corpus_files(corpus, manifest)
    verify_assets(corpus, manifest.manifest_hash)
    parent = load_manifest(PARENT_CORPUS / "manifest.json")
    verify_manifest(parent)
    verify_parent(parent)
    if (
        _changed_parent_contract(manifest, parent)
    ):
        raise ValueError("parent specification, operator catalogue or exclusions changed")
    parent_cases, cases, active, plan, selection, added = _check_pattern_selection(
        corpus, manifest, parent)
    if manifest.version == SCAN_REPAIR_VERSION:
        verify_retained_v17(corpus, manifest.cases)
    if any(plan.get(key) != value for key, value in selection.items()):
        raise ValueError("selection policy or evidence obligations drift")
    _check_pattern_generation_log(corpus, manifest)

    stats = coverage_stats(manifest.cases, active)
    if len(active) != 71 or len(stats["covered"]) < math.ceil(0.9 * len(active)):
        raise ValueError("static target or agreed feature denominator changed")
    return dict(
        stats,
        manifest_hash=manifest.manifest_hash,
        cases=len(cases),
        preserved_cases=len(parent_cases),
        added_cases=len(added),
        required_covered=math.ceil(0.9 * len(active)),
        static_target_met=True,
        required_targets=sorted(TARGETS),
        compilation_state="NOT_RUN",
        execution_state="NOT_RUN",
        agreement_state="AWAITING_JOINT_CONFIRMATION",
        coverage_basis="static workload-label union; not compiled API/IR coverage",
    )


def build(corpus=CORPUS, submission=SUBMISSION, *, version=VERSION):
    from .buildlog import GenerationLog
    from .corpusmod import load_manifest
    from .refcheck import check_all
    from .submission_preflight import require_submission_ready

    corpus, submission = Path(corpus), Path(submission)
    if version not in PATTERN_VERSIONS:
        raise ValueError("unknown pattern release version")
    if (corpus / "manifest.json").exists():
        if load_manifest(corpus / "manifest.json").version != version:
            raise ValueError("cannot reuse a frozen corpus as another release")
        result = static_readiness(corpus)
        require_submission_ready(submission, {case.operator for case in load_manifest(corpus / "manifest.json").cases})
        return result
    _check_pattern_draft_destinations(corpus, submission, version)

    parent, active, added, selection = _prepare_pattern_assets(corpus, submission, version)
    plan = dict(
        selection,
        format=1,
        version=version,
        active_atoms=active,
        parent_manifest_hash=parent.manifest_hash,
        parent_case_hashes={case.code: case.content_hash for case in parent.cases},
    )
    (corpus / "coverage-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    cases = [*parent.cases, *added]
    if version == SCAN_REPAIR_VERSION:
        verify_retained_v17(corpus, cases)
    log = GenerationLog(corpus / "generation-log.yaml")
    log.step("build_start", version=version, selection=selection)
    log.step(
        "submission_source_contract_gate", **require_submission_ready(submission, {case.operator for case in cases})
    )
    failures = check_all(cases, "cpu", corpus_dir=corpus)
    log.step("independent_cpu_reference_gate", cases=len(cases), failures=failures)
    if failures:
        raise ValueError(f"independent CPU reference gate failed: {failures}")
    _freeze_pattern_manifest(corpus, version, parent, cases, log)

    report = static_readiness(corpus)
    (corpus / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def required_pattern_targets(version, operators=()):
    if version == SCAN_REPAIR_VERSION and set(operators) == {"Cummin"}:
        return ("FA-PAT-SCAN-BLOCKED",)
    return tuple(TARGETS)


def pattern_gate(output):
    output = Path(output)
    path = output / "coverage-validation.json"
    proof = json.loads(path.read_text()) if path.is_file() else {}
    manifest = json.loads((output / "manifest.json").read_text())
    targets = required_pattern_targets(manifest.get("version"), proof.get("selected_operators", []))
    cases = {case["code"]: case for case in manifest["cases"]}
    checks = {}
    for atom in targets:
        checks[atom] = {}
        for metric, key in [("compilation", "compilation_witnesses"), ("execution_validation", "execution_witnesses")]:
            codes = proof.get(key, {}).get(atom, [])
            valid = (
                bool(codes)
                and len(codes) == len(set(codes))
                and all(code in cases and atom in cases[code]["feature_atoms"] for code in codes)
            )
            signatures = {
                (
                    cases[code]["operator"],
                    cases[code]["dtype"],
                    tuple(cases[code]["shape"]),
                    json.dumps(cases[code]["attrs"], sort_keys=True),
                    cases[code]["contiguous"],
                )
                for code in codes
                if code in cases and atom in cases[code]["feature_atoms"]
            }
            checks[atom][metric] = valid and len(signatures) >= 2 and atom in proof.get(metric, {}).get("covered", [])
    passed = (
        proof.get("selected_operator_validation_passed") is True
        and proof.get("manifest_hash") == manifest.get("manifest_hash")
        and {TARGETS[atom] for atom in targets} <= set(proof.get("selected_operators", []))
        and all(all(check.values()) for check in checks.values())
    )
    report = dict(
        scope="blocked_scan_repair" if len(targets) == 1 else "four_explicit_patterns",
        passed=passed,
        checks=checks,
        coverage_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        performance_acceptance="NOT_ESTABLISHED",
        taskbook_accepted=False,
    )
    (output / "target-pattern-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(not passed)


PatternCases = namedtuple('PatternCases', 'parent_cases cases active plan selection added')


def _check_pattern_selection(corpus, manifest, parent):
    from .corpusmod import compute_case_content_hash
    from .operator_registry import by_name
    from .specload import load_feature_atoms

    parent_cases = {case.code: case.content_hash for case in parent.cases}
    cases = {case.code: case for case in manifest.cases}
    if len(cases) != len(manifest.cases) or any(
        code not in cases or cases[code].content_hash != expected for code, expected in parent_cases.items()
    ):
        raise ValueError("parent cases removed, changed or duplicated")
    active = sorted(atom["id"] for atom in load_feature_atoms()["atoms"] if atom.get("status", "active") == "active")
    plan = json.loads((corpus / "coverage-plan.json").read_text())
    if (
        _invalid_parent_plan(plan, manifest, active, parent, parent_cases)
    ):
        raise ValueError("parent identity or feature denominator drift")
    expected, selection = select_patterns(parent.cases, extension_pool(), active)
    added = [case for case in manifest.cases if case.code not in parent_cases]
    if [case.code for case in added] != [case.code for case in expected]:
        raise ValueError("pattern selection differs from deterministic generation")
    for actual, generated in zip(added, expected):
        if compute_case_content_hash(corpus, actual) != compute_case_content_hash(corpus, generated):
            raise ValueError("pattern candidate content differs")
        spec = by_name(actual.operator)
        spec.validate_case_params(actual.dtype, actual.shape, actual.attrs)
        if actual.feature_atoms != label_pattern_case(actual, spec):
            raise ValueError("pattern labels are not derived from workload")
    return PatternCases(parent_cases, cases, active, plan, selection, added)


def _prepare_pattern_assets(corpus, submission, version):
    import yaml

    from .corpus_assets import copy_reference_tree
    from .corpusmod import compute_case_content_hash, load_manifest, verify_manifest
    from .specload import load_feature_atoms

    parent = load_manifest(PARENT_CORPUS / "manifest.json")
    verify_manifest(parent)
    verify_parent(parent)
    if any(case.content_hash != compute_case_content_hash(PARENT_CORPUS, case) for case in parent.cases):
        raise ValueError("parent case content mismatch")
    active = sorted(atom["id"] for atom in load_feature_atoms()["atoms"] if atom.get("status", "active") == "active")
    added, selection = select_patterns(parent.cases, extension_pool(), active)
    corpus.mkdir(parents=True, exist_ok=True)
    copy_reference_tree(PARENT_CORPUS, corpus)
    shutil.copytree(PARENT_SUBMISSION, submission, dirs_exist_ok=True)
    (submission / "operators/target_pattern_kernels.py").write_text(submission_source())
    payload = yaml.safe_load((submission / "manifest.yaml").read_text())
    for entry in payload["operators"]:
        if entry["name"] in TARGETS.values():
            entry["path"] = "operators/target_pattern_kernels.py"
    payload["submit_id"] = version
    (submission / "manifest.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    return parent, active, added, selection


def _check_pattern_generation_log(corpus, manifest):
    events = read_log(corpus / "generation-log.yaml")
    starts = [position for position, event in enumerate(events) if event.get("step") == "build_start"]
    attempt = events[starts[-1]:] if starts else []
    gates = [event for event in attempt if event.get("step") == "independent_cpu_reference_gate"]
    if (
        _incomplete_generation_attempt(attempt, manifest, gates)
    ):
        raise ValueError("independent CPU reference gate did not pass")


def _check_pattern_draft_destinations(corpus, submission, version):
    import yaml

    if corpus.resolve() == PARENT_CORPUS.resolve() or submission.resolve() == PARENT_SUBMISSION.resolve():
        raise ValueError("cannot overwrite frozen parent assets")
    if submission.exists() and any(submission.iterdir()):
        existing = submission / "manifest.yaml"
        if not existing.is_file() or yaml.safe_load(existing.read_text()).get("submit_id") != version:
            raise ValueError("cannot merge a pattern draft into another submission")
    if corpus.exists() and any(corpus.iterdir()):
        existing = corpus / "coverage-plan.json"
        if not existing.is_file() or json.loads(existing.read_text()).get("parent_version") != PARENT:
            raise ValueError("cannot merge a pattern draft into another corpus")


def _freeze_pattern_manifest(corpus, version, parent, cases, log):
    from .corpus_assets import freeze_assets
    from .corpusmod import build_manifest, write_manifest

    manifest = build_manifest(
        corpus,
        version,
        parent.spec_version,
        parent.operators,
        cases,
        feature_atoms_version=parent.feature_atoms_version,
        excluded=parent.excluded,
    )
    write_manifest(manifest, corpus / "manifest.json")
    log.step("build_end", manifest_hash=manifest.manifest_hash, cases=len(cases), manifest_written=True)
    freeze_assets(corpus, manifest.manifest_hash)
