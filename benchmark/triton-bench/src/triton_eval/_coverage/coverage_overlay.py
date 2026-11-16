# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Provenance-preserving historical overlay, separate from frozen-suite acceptance."""

from __future__ import annotations

import json
from pathlib import Path

from .coverage_evidence import _rate, evidence_path, sha
from .corpusmod import load_manifest, verify_manifest
from .envprobe.parse import COVERAGE_ENVIRONMENT_KEYS as ENVIRONMENT_KEYS





def _read(root, submission):
    root, submission = Path(root).resolve(), Path(submission).resolve()
    report = json.loads((root / "coverage-validation.json").read_text())
    summary = json.loads((root / "suite-summary.json").read_text())
    manifest = load_manifest(root / "manifest.json")
    verify_manifest(manifest)
    if report["manifest_hash"] != manifest.manifest_hash or summary["manifest_hash"] != manifest.manifest_hash:
        raise ValueError("overlay manifest identity mismatch")
    required = {
        "manifest.json",
        "suite-summary.json",
        "coverage-plan.json",
        "submission-sha256.json",
        "evaluation-assets.json",
    }
    hashes = report["evidence_sha256"]
    if not required <= set(hashes):
        raise ValueError("overlay lacks indexed identity evidence")
    for relative, expected in hashes.items():
        if sha(evidence_path(root, relative).read_bytes()) != expected:
            raise ValueError(f"overlay evidence hash mismatch: {relative}")
    snapshot = {
        str(path.relative_to(submission)): sha(path.read_bytes())
        for path in sorted(submission.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml")
    }
    if snapshot != report["submission_sha256"] or snapshot != json.loads((root / "submission-sha256.json").read_text()):
        raise ValueError("overlay submission identity mismatch")
    plan = json.loads((root / "coverage-plan.json").read_text())
    if len(plan["active_atoms"]) != len(set(plan["active_atoms"])) or set(report["static"]["covered"]) | set(
        report["static"]["missing"]
    ) != set(plan["active_atoms"]):
        raise ValueError("overlay feature denominator mismatch")
    cases = {case.code: case for case in manifest.cases}
    rows, environments = _read_rows(root, summary, hashes, cases)
    _validate_witnesses(report, summary, cases, rows)
    return dict(
        root=root,
        report=report,
        summary=summary,
        cases=cases,
        rows=rows,
        snapshot=snapshot,
        active=plan["active_atoms"],
        environments=environments,
        assets=json.loads((root / "evaluation-assets.json").read_text())["files"],
    )


def _read_rows(root, summary, hashes, cases):
    rows, environments = {}, []
    for operator in summary["selected_operators"]:
        files = list((root / operator).glob("run-*/results.jsonl"))
        if not files:
            continue
        if len(files) != 1:
            raise ValueError("overlay ambiguous results partition")
        for path in (files[0], files[0].parent / "env.json"):
            if str(path.relative_to(root)) not in hashes:
                raise ValueError("overlay unindexed result/environment")
        for line in files[0].read_text().splitlines():
            row = json.loads(line)
            code = row["case_code"]
            if code in rows or code not in cases or row["operator"] != cases[code].operator:
                raise ValueError("overlay duplicate or foreign result")
            rows[code] = row
        environment = json.loads((files[0].parent / "env.json").read_text())
        environments.append({key: environment[key] for key in ENVIRONMENT_KEYS})
    return rows, environments


def _validate_witnesses(report, summary, cases, rows):
    for metric, witnesses in (
        ("compilation", "compilation_witnesses"),
        ("execution_validation", "execution_witnesses"),
    ):
        if set(report[metric]["covered"]) != set(report[witnesses]):
            raise ValueError("overlay coverage witness mismatch")
        for atom, codes in report[witnesses].items():
            if not codes or any(code not in rows or atom not in cases[code].feature_atoms for code in codes):
                raise ValueError("overlay unknown or untagged witness")
            if metric == "execution_validation" and any(rows[code]["status"] != "PASS" for code in codes):
                raise ValueError("overlay execution witness is not PASS")
            if metric == "execution_validation" and any(
                summary["operators"][cases[code].operator].get("reference_gate_returncodes") != {"cpu": 0, "npu:0": 0}
                for code in codes
            ):
                raise ValueError("overlay execution witness lacks reference gates")


def overlay_results(baseline, repair, baseline_submission, repair_submission, *, reanalyze=False):
    import yaml

    def entries(submission):
        return {
            entry["name"]: entry
            for entry in yaml.safe_load((Path(submission) / "manifest.yaml").read_text())["operators"]
        }

    before = _read(baseline, baseline_submission)
    after = _read(repair, repair_submission)
    selected = _validate_pair(before, after)
    _validate_routing(before, after, baseline_submission, repair_submission, selected, entries)
    differences = _environment_differences(before, after)
    origins, metrics = _overlay_metrics(before, after, selected)
    report = _overlay_report(before, after, selected, differences, origins, metrics)
    if reanalyze:
        report["recognizer_reanalysis"] = _reanalyze(before, after, selected, metrics)
    return report


def _validate_pair(before, after):
    if before["active"] != after["active"] or {code: case.content_hash for code, case in before["cases"].items()} != {
        code: case.content_hash for code, case in after["cases"].items()
    }:
        raise ValueError("overlay case contents or denominator changed")
    selected = set(after["summary"]["selected_operators"])
    if not selected or not selected < set(before["summary"]["selected_operators"]):
        raise ValueError("overlay repair must select a strict subset of baseline operators")
    if not after["report"].get("selected_operator_validation_passed"):
        raise ValueError("overlay repair did not pass independent operator validation")
    for key in (
        "spec/spec.yaml",
        "spec/tolerance.yaml",
        "spec/feature_atoms.yaml",
        "src/triton_eval/inputs.py",
        "src/triton_eval/comparison.py",
        "src/triton_eval/coverage_reference.py",
        "src/triton_eval/official_reference.py",
    ):
        if before["assets"].get(key) is None or before["assets"][key] != after["assets"].get(key):
            raise ValueError(f"overlay semantic/reference asset changed: {key}")
    return selected


def _validate_routing(before, after, baseline_submission, repair_submission, selected, entries):
    old_entries, new_entries = entries(baseline_submission), entries(repair_submission)
    if set(old_entries) != set(new_entries):
        raise ValueError("overlay operator catalogue changed")
    for operator, entry in old_entries.items():
        if not (operator not in selected):
            continue
        if entry != new_entries[operator]:
            raise ValueError(f"overlay untested operator routing changed: {operator}")
    for path, expected in before["snapshot"].items():
        if path.endswith(".py") and expected != after["snapshot"].get(path):
            raise ValueError(f"overlay shared implementation changed: {path}")


def _environment_differences(before, after):
    environments = before["environments"] + after["environments"]
    differences = {
        key: sorted({environment[key] for environment in environments})
        for key in ENVIRONMENT_KEYS
        if len({environment[key] for environment in environments}) != 1
    }
    if any(key != "venv_lock_hash" for key in differences):
        raise ValueError(f"overlay hardware/compiler/runtime versions differ: {sorted(differences)}")
    return differences


def _overlay_metrics(before, after, selected):
    origins, metrics = {}, {}
    for metric, witness_key in (
        ("compilation", "compilation_witnesses"),
        ("execution_validation", "execution_witnesses"),
    ):
        witnesses = _metric_witnesses(before, after, selected, witness_key)
        origins[metric] = witnesses
        metrics[metric] = _rate(witnesses, before["active"])
    return origins, metrics


def _metric_witnesses(before, after, selected, witness_key):
    witnesses = {}
    for origin, data in (("baseline", before), ("repair", after)):
        _origin_metric_witnesses(witnesses, origin, data, selected, witness_key)
    return witnesses


def _origin_metric_witnesses(witnesses, origin, data, selected, witness_key):
    for atom, codes in data["report"][witness_key].items():
        for code in codes:
            if (origin == "repair") != (data["cases"][code].operator in selected):
                continue
            _append_metric_witness(witnesses, atom, code, origin, data)


def _append_metric_witness(witnesses, atom, code, origin, data):
    entries = witnesses.setdefault(atom, [])
    if entries and entries[-1]["source"] == origin:
        entries[-1]["witness_count"] += 1
    else:
        entries.append(
            dict(
                case_code=code,
                source=origin,
                witness_count=1,
                manifest_hash=data["report"]["manifest_hash"],
            )
        )


def _overlay_report(before, after, selected, differences, origins, metrics):
    rows = {code: row for code, row in before["rows"].items() if before["cases"][code].operator not in selected}
    rows.update(after["rows"])
    missing = sorted(set(before["cases"]) - set(rows))
    report = dict(
        format=1,
        validation_scope="historical_operator_overlay",
        analysis_only=True,
        replacement_operators=sorted(selected),
        compilation=metrics["compilation"],
        execution_validation=metrics["execution_validation"],
        feature_origins=origins,
        expected_cases=len(before["cases"]),
        passed_cases=sum(row["status"] == "PASS" for row in rows.values()),
        missing_cases=missing,
        nonpass_cases={code: row["status"] for code, row in rows.items() if row["status"] != "PASS"},
        newly_confirmed_features=sorted(
            set(metrics["compilation"]["covered"]) - set(before["report"]["compilation"]["covered"])
        ),
        static_without_compilation=sorted(
            set(before["report"]["static"]["covered"]) - set(metrics["compilation"]["covered"])
        ),
        environment_differences=differences,
        single_locked_environment=not differences,
        verified_npu_coverage_target_met=False,
        taskbook_accepted=False,
        source_reports={
            name: dict(
                path=str(data["root"]),
                manifest_hash=data["report"]["manifest_hash"],
                report_sha256=sha((data["root"] / "coverage-validation.json").read_bytes()),
            )
            for name, data in (("baseline", before), ("repair", after))
        },
    )
    return report


def _reanalyze(before, after, selected, metrics):
    additions = {}
    for origin, data in (("baseline", before), ("repair", after)):
        for code, row in data["rows"].items():
            if (origin == "repair") != (data["cases"][code].operator in selected):
                continue
            for atom, entry in _case_additions(data, code, row, metrics["compilation"]["covered"]).items():
                additions.setdefault(atom, []).append(dict(source=origin, **entry))
    return dict(
        new_features=additions,
        compilation=_rate(set(metrics["compilation"]["covered"]) | set(additions), before["active"]),
        execution_validation=_rate(
            set(metrics["execution_validation"]["covered"]) | set(additions), before["active"]
        ),
        scope=("offline reanalysis of indexed real NPU artifacts; no new device execution"),
        frozen_suite_acceptance=False,
    )


def _case_additions(data, code, row, covered):
    from .coverage_evidence import feature_witness, observation

    case = data["cases"][code]
    missing_atoms = set(case.feature_atoms) - set(covered)
    if not missing_atoms or row["status"] != "PASS":
        return {}
    paths = list((data["root"] / case.operator).glob("run-*/compilation/" + code + "/observation.json"))
    if len(paths) != 1 or str(paths[0].relative_to(data["root"])) not in data["report"]["evidence_sha256"]:
        raise ValueError("overlay reanalysis lacks indexed compiler observation")
    captured = observation(case, paths[0], data["report"]["manifest_hash"], row["run_id"])
    additions = {}
    for atom in sorted(missing_atoms):
        if atom not in ("FA-FUSE-INTERM-ELIDE", "FA-PAT-VEC-LOAD", "FA-PAT-TILE-GEMM", "FA-PROG-GRID3D"):
            continue
        ok, reason = feature_witness(atom, captured)
        gates = data["summary"]["operators"][case.operator].get("reference_gate_returncodes")
        if ok and captured["data"].get("synchronized") is True and gates == {"cpu": 0, "npu:0": 0}:
            additions[atom] = dict(case_code=code, reason=reason, observation_sha256=sha(paths[0].read_bytes()))
    return additions
