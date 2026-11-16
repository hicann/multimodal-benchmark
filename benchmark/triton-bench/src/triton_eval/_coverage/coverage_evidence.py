# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Offline audit of real NPU coverage evidence; missing evidence never passes.

Compilation coverage here is the fraction of frozen feature obligations with a
compiled workload witness. It is NOT coverage of all compiler IR operations.
IR/API inventories have no percentage without their own agreed denominator.
"""

from __future__ import annotations
from collections import namedtuple
import ast
import copy
import gzip
import os
import hashlib
import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path

from .envprobe.parse import COVERAGE_ENVIRONMENT_KEYS


def _invalid_child_integrity(integrity, child):
    if (integrity["run_status"] != "OK"):
        return True
    if (integrity["env_mismatches"]):
        return True
    if (integrity["submission_changed_during_run"]):
        return True
    if (child["exit_code"] != 0):
        return True
    if (child.get("signal")):
        return True
    return bool(child.get("killed_by_us"))


def _invalid_regression_selection(selected_operators, selected, all_operators, summary):
    if (not selected_operators):
        return True
    if (len(selected) != len(selected_operators)):
        return True
    if (not selected_operators < all_operators):
        return True
    return bool(summary.get("validation_scope") != "operator_regression")


def _invalid_observation_identity(data, manifest_hash, case, run_id):
    if (data.get("manifest_hash") != manifest_hash):
        return True
    if (data.get("case_hash") != case.content_hash):
        return True
    if (data.get("case_code") != case.code):
        return True
    if (data.get("operator") != case.operator):
        return True
    if (data.get("run_id") != run_id):
        return True
    return not str(data.get("device", "")).startswith("npu")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def evidence_path(root, relative, boundary=None):
    """Reject escapes and symlinks before resolving the filesystem path."""
    root = Path(root).resolve()
    boundary = Path(boundary).resolve() if boundary is not None else root
    if Path(relative).is_absolute():
        raise ValueError("absolute evidence path")
    path = Path(os.path.abspath(root / relative))
    if not path.is_relative_to(boundary):
        raise ValueError("artifact path outside capture directory")
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(boundary)):
        raise ValueError("symlink in evidence path")
    return path


def read_blob(root, entry):
    root = Path(root).resolve()
    path = evidence_path(root, entry["path"], root.parent)
    data = path.read_bytes()
    if sha(data) != entry["sha256"]:
        raise ValueError("artifact SHA256 mismatch")
    if entry.get("compression") == "gzip":
        data = gzip.decompress(data)
        if sha(data) != entry["content_sha256"]:
            raise ValueError("artifact content SHA256 mismatch")
    return data


def source_fingerprints(source):
    fingerprints = set()
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node = copy.deepcopy(node)
            node.decorator_list = []
            fingerprints.add(sha(ast.dump(node, include_attributes=False).encode()))
    return fingerprints


def observation(case, path, manifest_hash, run_id):
    data = json.loads(path.read_text())
    if (
        _invalid_observation_identity(data, manifest_hash, case, run_id)
    ):
        raise ValueError("observation case/run/device identity mismatch")
    if data.get("diagnostics"):
        raise ValueError("incomplete compiler capture: " + "; ".join(data["diagnostics"]))
    launches = data.get("launches", [])
    if not launches:
        raise ValueError("no captured JIT launch")
    ir, sources, inventory = _read_observation_launches(launches, path)

    _check_observation_inputs(data, case)

    return dict(
        data=data, case=case, ir="\n".join(ir), source="\n".join(sources), inventory=inventory, launches=launches
    )


def feature_witness(atom, obs):
    """Conservative implementation obligations; static proxies cannot certify IR."""
    ir, source = obs["ir"], obs["source"]
    from .pattern_generation import TARGETS

    if atom in TARGETS:
        from .pattern_evidence import pattern_witnesses

        if "pattern_witnesses" not in obs:
            obs["pattern_witnesses"] = pattern_witnesses(obs)
        proof = obs["pattern_witnesses"][atom]
        return proof["verified"], proof["reason"]
    from .implementation_evidence import ATOMS, implementation_witnesses

    if atom in ATOMS:
        if "implementation_witnesses" not in obs:
            obs["implementation_witnesses"] = implementation_witnesses(obs)
        proof = obs["implementation_witnesses"][atom]
        return proof["verified"], proof["reason"]
    if atom.startswith("FA-PROG-GRID"):
        dim = int(atom.removeprefix("FA-PROG-GRID").removesuffix("D"))
        ok = any(len(l.get("grid") or []) == dim for l in obs["launches"])
        return ok, "requires an observed explicit launch grid with the specified rank"
    if atom.startswith("FA-PAT-"):
        return False, "no dedicated recognizer for this optimization pattern"
    if atom == "FA-CTRL-STATIC-UNROLL":
        return ("tl.static_range" in source and "scf.for" not in ir), (
            "requires static_range source and loop-free captured IR"
        )
    if atom == "FA-CONCUR-ACCUM":
        return bool(re.search(r"(?:atomic|atom)[_.]", ir)), "requires an atomic operation in captured IR"
    if atom == "FA-LINALG-DOT":
        return bool(re.search(r"\b(?:tt|ttg)\.(?:dot|warp_group_dot)", ir)), "requires a dot operation in captured IR"
    # These are workload obligations, rather than assertions that a particular
    # API or optimization was used. The case must have an audited compiled
    # variant; runtime credit additionally requires synchronization and PASS.
    return True, "compiled witness for the frozen case-level semantic/input obligation"


def _rate(covered, active):
    covered = sorted(set(covered) & set(active))
    return dict(
        covered=covered,
        missing=sorted(set(active) - set(covered)),
        covered_count=len(covered),
        total=len(active),
        rate=len(covered) / len(active),
        required_covered=math.ceil(0.9 * len(active)),
        target_met=len(covered) >= math.ceil(0.9 * len(active)),
    )


def audit_suite(corpus, results, submission, *, operator_regression=False):
    from .coverage_gate import static_readiness

    ready = static_readiness(corpus)
    manifest, active, root, snapshot, functions, summary = _coverage_audit_inputs(corpus, results, submission)

    state = _coverage_audit_state(root)
    all_operators = {c.operator for c in manifest.cases}
    selected_operators = all_operators
    if operator_regression:
        selected = summary.get("selected_operators", [])
        selected_operators = set(selected)
        if (
            _invalid_regression_selection(selected_operators, selected, all_operators, summary)
        ):
            raise ValueError("invalid operator regression scope")
    for operator in sorted(selected_operators):
        _audit_coverage_operator(operator, manifest, root, summary, active, functions, state)
    return _coverage_audit_report(ready, manifest, active, root, snapshot,
                                 operator_regression, selected_operators, state)


def _read_observation_launches(launches, path):
    ir, sources, inventory = [], [], set()
    for launch in launches:
        if not launch.get("returned_compiled_kernel"):
            raise ValueError("JIT return has no compiler artifact")
        source = read_blob(path.parent, launch["source"]).decode()
        if not source_fingerprints(source):
            raise ValueError("missing captured kernel source")
        sources.append(source)
        launch["source_text"] = source
        launch["stage_text"] = {}
        stages = launch.get("artifacts", [])
        frontend = [a for a in stages if a["stage"].lower() in ("ttir", "ttgir", "llir", "llvmir", "mlir")]
        backend = []
        backend_stages = {"ptx", "amdgcn", "ascend", "npubin", "cubin", "hsaco", "obj"}
        for artifact in stages:
            if artifact["kind"] == "binary" or artifact["stage"].lower() in backend_stages:
                backend.append(artifact)
        if not frontend or not backend:
            raise ValueError("requires actual IR and backend code artifacts")
        for entry in stages:
            raw = read_blob(path.parent, entry)
            _record_stage_text(launch, entry, raw)
            if entry in frontend:
                text = raw.decode()
                ir.append(text)
                inventory.update(
                    re.findall(
                        r"\b(?:tt|ttg|arith|math|scf|cf|llvm|hivm|memref|linalg"
                        r"|tensor|affine|vector)\.[A-Za-z_][A-Za-z_0-9.]*",
                        text,
                    )
                )
        from .backend_evidence import backend_snapshots

        snapshots, binding = backend_snapshots(launch, path.parent)
        launch["stage_text"].update(snapshots)
        launch["backend_binding"] = binding
        for text in snapshots.values():
            inventory.update(re.findall(r"\b(?:hivm|memref|linalg|vector|llvm)\.[A-Za-z_][A-Za-z_0-9.]*", text))
    return ir, sources, inventory


def _record_stage_text(launch, entry, raw):
    if entry.get("kind") == "text" and not entry["stage"].startswith("backend-"):
        launch["stage_text"][entry["stage"].lower()] = raw.decode()
        if entry["stage"] == "compiler_stderr":
            from .compilation_evidence import compiler_passes

            launch["stage_text"].update(compiler_passes(raw.decode(), launch["name"], include_before=True))


def _check_observation_inputs(data, case):
    metadata = data.get("input_metadata", [])
    main_index = 1 if case.operator == "Where" else 0
    if data.get("main_input_index", 0) != main_index:
        raise ValueError("actual main input identity differs from frozen contract")
    if len(metadata) <= main_index or metadata[main_index].get("shape") != case.shape:
        raise ValueError("actual input shape differs from frozen case")
    strides = metadata[main_index].get("stride", [])
    if len(strides) != len(case.shape):
        raise ValueError("missing actual input strides")
    contiguous, expected_stride = True, 1
    for size, stride in zip(reversed(case.shape), reversed(strides)):
        if size > 1 and stride != expected_stride:
            contiguous = False
        expected_stride *= size
    if contiguous != case.contiguous:
        raise ValueError("actual input layout differs from frozen case")
    dtype_name = "torch." + case.dtype
    if not any(t.get("dtype") == dtype_name for t in metadata):
        raise ValueError("advertised dtype not present in actual input metadata")


AuditInputs = namedtuple('AuditInputs', 'manifest active root snapshot functions summary')


def _coverage_audit_inputs(corpus, results, submission):
    from .corpusmod import load_manifest

    manifest = load_manifest(Path(corpus) / "manifest.json")
    plan = json.loads((Path(corpus) / "coverage-plan.json").read_text())
    active = plan["active_atoms"]
    root = Path(results).resolve()
    submission = Path(submission).resolve()
    snapshot = {
        str(p.relative_to(submission)): sha(p.read_bytes())
        for p in sorted(Path(submission).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yaml", ".yml")
    }
    if not snapshot or snapshot != json.loads((root / "submission-sha256.json").read_text()):
        raise ValueError("submission snapshot changed or missing; previous qualification cannot transfer")
    functions = set()
    for path in Path(submission).rglob("*.py"):
        functions.update(source_fingerprints(path.read_text()))
    summary = json.loads((root / "suite-summary.json").read_text())
    _verify_audit_archive(corpus, root, manifest, summary)

    return AuditInputs(manifest, active, root, snapshot, functions, summary)


def _verify_audit_archive(corpus, root, manifest, summary):
    from .corpusmod import load_manifest, verify_manifest

    if summary.get("manifest_hash") != manifest.manifest_hash:
        raise ValueError("suite manifest identity mismatch")
    if manifest.version in (
        "v0.13-evidence-candidates",
        "v0.14-backend-evidence-candidates",
        "v0.15-backend-capture-fix-candidates",
        "v0.16-backend-pattern-certification-candidates",
        "v0.17-pattern-completion-candidates",
        "v0.18-scan-certification-candidates",
    ):
        from .reference_execution import POLICY

        if summary.get("reference_execution_policy") != POLICY:
            raise ValueError("missing frozen reference execution policy")
    archived = load_manifest(root / "manifest.json")
    verify_manifest(archived)
    if archived.manifest_hash != manifest.manifest_hash:
        raise ValueError("archived manifest identity mismatch")
    for name in ("evaluation-assets.json", "coverage-plan.json"):
        if (root / name).read_bytes() != (Path(corpus) / name).read_bytes():
            raise ValueError("archived corpus asset identity mismatch: " + name)
    health = json.loads((root / "capabilities/capabilities.json").read_text())
    if health.get("probes", {}).get("health", {}).get("status") != "PASS":
        raise ValueError("suite hardware health probe did not pass")


def _audit_coverage_operator(operator, manifest, root, summary, active, functions, state):
    cases = [c for c in manifest.cases if c.operator == operator]
    partition = root / operator
    files = list(partition.glob("run-*/results.jsonl"))
    try:
        run, index, indexed = _read_coverage_partition(
            files, root, summary, operator, manifest, cases, state)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state.errors[operator] = [str(exc)]
        return
    gates = summary.get("operators", {}).get(operator, {}).get("reference_gate_returncodes")
    for case in cases:
        _audit_coverage_case(case, run, index, indexed, gates, manifest, active, functions, state)


def _coverage_audit_state(root):
    from types import SimpleNamespace

    state = SimpleNamespace()
    state.implementation_details, state.backend_details = {}, {}
    state.records, state.errors, state.environments, state.evidence_hashes = {}, {}, set(), {}
    for name in (
        "manifest.json",
        "evaluation-assets.json",
        "coverage-plan.json",
        "submission-sha256.json",
        "suite-summary.json",
        "capabilities/capabilities.json",
    ):
        state.evidence_hashes[name] = sha((root / name).read_bytes())
    state.inventory, state.compiled_atoms, state.runtime_atoms = set(), set(), set()
    state.compiled_cases, state.runtime_cases = [], []
    state.unverified, state.compiled_witnesses, state.runtime_witnesses = {}, {}, {}
    return state


def _coverage_audit_report(ready, manifest, active, root, snapshot,
                           operator_regression, selected_operators, state):
    if len(state.environments) > 1:
        state.errors["environment"] = ["mixed hardware/compiler/runtime identities; cannot union coverage"]
        state.compiled_cases, state.runtime_cases, state.compiled_atoms, state.runtime_atoms = [], [], set(), set()
        state.compiled_witnesses, state.runtime_witnesses = {}, {}
    report = _coverage_audit_header(ready, manifest, active, operator_regression,
                                    selected_operators, state)
    report.update(
        taskbook_accepted=False,
        agreement_state="AWAITING_JOINT_CONFIRMATION",
        errors=state.errors,
        unverified_feature_requirements={k: v for k, v in state.unverified.items() if k not in state.compiled_atoms},
        compilation_witnesses=state.compiled_witnesses,
        execution_witnesses=state.runtime_witnesses,
        implementation_evidence=state.implementation_details,
        backend_evidence=state.backend_details,
        observed_ir_operations=sorted(state.inventory),
        evidence_sha256=state.evidence_hashes,
        submission_sha256=snapshot,
        scope="frozen feature-obligation coverage; IR inventory is not compiler-wide API/IR branch coverage",
        performance_acceptance="not established by smoke functionality runs",
    )
    return report


def _read_coverage_partition(files, root, summary, operator, manifest, cases, state):
    if len(files) != 1:
        raise ValueError("requires exactly one results file")
    run = files[0].parent
    index, indexed = _coverage_partition_index(run, root, state)

    audit = summary.get("operators", {}).get(operator, {})
    if audit.get("manifest_hash") != manifest.manifest_hash or any(
        audit.get(key) != 0 for key in ("run_returncode", "score_returncode", "report_returncode")
    ):
        raise ValueError("suite partition identity/command failed")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    if not any(e.get("event") == "probe" and e.get("ok") is True for e in events):
        raise ValueError("missing successful child health probe")
    integrity = json.loads((run / "integrity.json").read_text())
    child = integrity["child"]
    if (
        _invalid_child_integrity(integrity, child)
    ):
        raise ValueError("run integrity failed")
    _record_coverage_environment(run, state)

    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    if Counter(r["case_code"] for r in rows) != Counter(c.code for c in cases) or any(
        r["operator"] != operator or r["run_id"] != index["run_id"] for r in rows
    ):
        raise ValueError("missing, duplicate, foreign or wrong-run results")
    state.records.update({r["case_code"]: r for r in rows})
    return run, index, indexed


def _audit_coverage_case(case, run, index, indexed, gates, manifest, active, functions, state):
    record = state.records[case.code]
    try:
        path = run / "compilation" / case.code / "observation.json"
        if str(path.relative_to(run)) not in indexed:
            raise ValueError("missing indexed compilation observation")
        obs = observation(case, path, manifest.manifest_hash, index["run_id"])
        _record_backend_inventory(obs, case, state)

        # Every referenced blob must also be in the supervisor's index.
        _check_indexed_launches(obs, path, run, indexed)
        if not source_fingerprints(obs["source"]).issubset(functions):
            raise ValueError("captured kernel source differs from submission snapshot")
        state.compiled_cases.append(case.code)
        state.inventory.update(obs["inventory"])
        runtime_ok = (
            obs["data"].get("synchronized") is True
            and record["status"] == "PASS"
            and gates == {"cpu": 0, "npu:0": 0}
        )
        if runtime_ok:
            state.runtime_cases.append(case.code)
        _audit_feature_witnesses(case, active, obs, runtime_ok, state)

        if "implementation_witnesses" in obs:
            state.implementation_details[case.code] = obs["implementation_witnesses"]
        if "pattern_witnesses" in obs:
            state.implementation_details.setdefault(case.code, {}).update(obs["pattern_witnesses"])
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        state.errors[case.code] = [str(exc)]


def _check_indexed_launches(obs, path, run, indexed):
    for launch in obs["launches"]:
        for entry in [launch["source"], *launch["artifacts"]]:
            blob = (path.parent / entry["path"]).resolve()
            if str(blob.relative_to(run.resolve())) not in indexed:
                raise ValueError("unindexed compiler blob")


def _record_backend_inventory(obs, case, state):
    state.backend_details[case.code] = [
        dict(
            kernel=launch["name"],
            invocations=len(launch.get("backend_compilations", [])),
            cache_reused=launch.get("backend_cache_reused", False),
            bindings=launch.get("backend_binding", []),
            output_candidates=[
                dict(stage=candidate["stage"], sha256=candidate["sha256"])
                for invocation in launch.get("backend_compilations", [])
                for candidate in invocation.get("output_candidates", [])
            ],
            notes=[
                note
                for invocation in launch.get("backend_compilations", [])
                for note in invocation.get("notes", [])
            ],
        )
        for launch in obs["launches"]
    ]


def _audit_feature_witnesses(case, active, obs, runtime_ok, state):
    for atom in set(case.feature_atoms) & set(active):
        ok, why = feature_witness(atom, obs)
        if ok:
            state.compiled_atoms.add(atom)
            state.compiled_witnesses.setdefault(atom, []).append(case.code)
            if runtime_ok:
                state.runtime_atoms.add(atom)
                state.runtime_witnesses.setdefault(atom, []).append(case.code)
        else:
            state.unverified[atom] = why


def _coverage_audit_header(ready, manifest, active, operator_regression, selected_operators, state):
    compiled = _rate(state.compiled_atoms, active)
    execution = _rate(state.runtime_atoms, active)
    complete = len(state.records) == len(manifest.cases) and not state.errors
    selected_cases = [case for case in manifest.cases if case.operator in selected_operators]
    selected_passed = (len(state.records) == len(selected_cases) and not state.errors
                       and len(state.runtime_cases) == len(selected_cases))
    return dict(
        format=1,
        manifest_hash=manifest.manifest_hash,
        static=ready,
        validation_scope="operator_regression" if operator_regression else "full_corpus",
        selected_operators=sorted(selected_operators),
        selected_expected_cases=len(selected_cases),
        selected_operator_validation_passed=selected_passed,
        compilation=compiled,
        execution_validation=execution,
        compiled_cases=len(state.compiled_cases),
        validated_cases=len(state.runtime_cases),
        expected_cases=len(manifest.cases),
        complete=complete,
        all_pass=len(state.runtime_cases) == len(manifest.cases),
        verified_npu_coverage_target_met=complete
        and len(state.runtime_cases) == len(manifest.cases)
        and compiled["target_met"]
        and execution["target_met"],
    )


def _coverage_partition_index(run, root, state):
    index = json.loads((run / "artifacts.json").read_text())
    indexed = {}
    for entry in index["files"]:
        file = evidence_path(run, entry["path"])
        if entry["path"] in indexed or sha(file.read_bytes()) != entry["sha256"]:
            raise ValueError("run artifact identity/hash mismatch")
        indexed[entry["path"]] = entry["sha256"]
        state.evidence_hashes[str(file.relative_to(root.resolve()))] = entry["sha256"]
    state.evidence_hashes[str((run / "artifacts.json").relative_to(root))] = sha(
        (run / "artifacts.json").read_bytes()
    )
    for required in ("results.jsonl", "env.json", "integrity.json", "events.jsonl"):
        if required not in indexed:
            raise ValueError(f"unindexed evidence: {required}")
    return index, indexed


def _record_coverage_environment(run, state):
    env = json.loads((run / "env.json").read_text())
    if (env.get("env_health") != "ok" or env.get("npu_count", 0) < 1
            or any(not env.get(key) for key in COVERAGE_ENVIRONMENT_KEYS)):
        raise ValueError("healthy NPU/compiler environment required")
    state.environments.add(json.dumps({key: env[key] for key in COVERAGE_ENVIRONMENT_KEYS}, sort_keys=True))
