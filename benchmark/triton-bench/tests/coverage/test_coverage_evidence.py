# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Synthetic unit fixtures only: these never constitute NPU qualification."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_eval.compilation_evidence import CompilationCapture
from triton_eval.coverage_evidence import audit_suite, feature_witness, observation, read_blob, sha
from triton_eval.corpusmod import build_manifest
from triton_eval.models import CaseSpec, CaseSource
from triton_eval.envprobe.parse import COVERAGE_ENVIRONMENT_KEYS

from triton_eval.fixture_sources import fixture_path

SOURCE = "def synthetic_kernel(X):\n    return X\n"
IR = "module { arith.addf tt.load tt.store }"


class SyntheticJIT:
    src = SOURCE
    __name__ = "synthetic_kernel"

    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(asm={"ttir": IR, "npubin": b"synthetic-unit-backend"})


def triton_fixture(cls=SyntheticJIT):
    return SimpleNamespace(runtime=SimpleNamespace(jit=SimpleNamespace(JITFunction=cls)))


def test_synthetic_jit_tracks_instance_calls_without_sharing_mutable_results():
    first, second = SyntheticJIT(), SyntheticJIT()
    result = first.run(17, grid=(2,))
    result.asm.clear()
    next_result = first.run(warmup=True)
    assert first.calls == [((17,), {"grid": (2,)}), ((), {"warmup": True})]
    assert second.calls == []
    assert next_result.asm == {"ttir": IR, "npubin": b"synthetic-unit-backend"}


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def make_capture(run, case, manifest_hash="manifest", run_id=11, cls=SyntheticJIT):
    directory = run / "compilation" / case.code
    with CompilationCapture(
        triton_fixture(cls),
        directory,
        dict(
            case_code=case.code,
            case_hash=case.content_hash,
            operator=case.operator,
            manifest_hash=manifest_hash,
            run_id=run_id,
            device="npu:0",
            input_metadata=[{"shape": case.shape, "dtype": "torch.float32", "stride": [3, 1]}],
        ),
    ) as capture:
        cls().run(grid=(1,))
        capture.payload["synchronized"] = True
    return directory / "observation.json"


def index_run(run):
    files = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name != "artifacts.json":
            files.append({"path": str(path.relative_to(run)), "sha256": sha(path.read_bytes()),
                          "bytes": path.stat().st_size})
    json_write(run / "artifacts.json", {"run_id": 11, "files": files})


@pytest.fixture(name="suite")
def coverage_suite(tmp_path, monkeypatch):
    corpus, results, submission = [tmp_path / p for p in ("corpus", "results", "submission")]
    for path in (corpus, results, submission):
        path.mkdir()
    case = CaseSpec(
        code="synthetic-add",
        operator="Add",
        source=CaseSource.VENDOR_SELFTEST,
        dtype="float32",
        shape=[2, 3],
        seed=1,
        feature_atoms=["FA-DT-FP32", "FA-MATH-ADD", "FA-PROG-GRID1D"],
    )
    manifest = build_manifest(corpus, "synthetic-unit-only", "synthetic-unit-only", [], [case])
    for root in (corpus, results):
        (root / "manifest.json").write_text(manifest.model_dump_json())
        json_write(root / "coverage-plan.json", {"active_atoms": case.feature_atoms})
        json_write(root / "evaluation-assets.json", {"fixture": "synthetic-unit-only"})
    (submission / "kernel.py").write_text(SOURCE)
    json_write(results / "submission-sha256.json", {"kernel.py": sha(SOURCE.encode())})
    json_write(results / "capabilities/capabilities.json", {"probes": {"health": {"status": "PASS"}}})
    audit = dict(
        manifest_hash=manifest.manifest_hash,
        reference_gate_returncodes={"cpu": 0, "npu:0": 0},
        run_returncode=0,
        score_returncode=0,
        report_returncode=0,
    )
    json_write(results / "suite-summary.json", {"manifest_hash": manifest.manifest_hash, "operators": {"Add": audit}})
    run = results / "Add/run-11"
    path = make_capture(run, case, manifest.manifest_hash)
    json_write(run / "results.jsonl", dict(case_code=case.code, operator=case.operator, run_id=11, status="PASS"))
    _write_coverage_run_environment(run)
    index_run(run)
    monkeypatch.setattr("triton_eval.coverage_gate.static_readiness", lambda _: {"fixture": "synthetic-unit-only"})
    return SimpleNamespace(
        corpus=corpus,
        results=results,
        submission=submission,
        run=run,
        path=path,
        case=case,
        audit=lambda: audit_suite(corpus, results, submission),
    )


def _write_coverage_run_environment(run):
    json_write(
        run / "integrity.json",
        dict(
            run_status="OK",
            env_mismatches=[],
            submission_changed_during_run=[],
            child={"exit_code": 0, "signal": None, "killed_by_us": False},
        ),
    )
    env = {key: "synthetic-unit-only" for key in COVERAGE_ENVIRONMENT_KEYS}
    env.update(env_health="ok", npu_count=1)
    json_write(run / "env.json", env)
    json_write(run / "events.jsonl", dict(event="probe", ok=True))


def test_captures_cache_returns_deduplicates_and_restores_hook(tmp_path):
    original = SyntheticJIT.run
    directory = tmp_path / "compilation/case"
    with CompilationCapture(triton_fixture(), directory, {}) as capture:
        SyntheticJIT().run(grid=(2, 3))
        SyntheticJIT().run(grid=(2, 3))
        SyntheticJIT().run(grid=(2, 3), warmup=True)
        assert len(capture.payload["launches"]) == 2
        first, second = capture.payload["launches"]
        assert first["artifacts"] == second["artifacts"]
        assert first["grid"] == [2, 3]
    assert SyntheticJIT.run is original
    assert len(list((directory.parent / "blobs").iterdir())) == 3
    assert not json.loads((directory / "observation.json").read_text())["diagnostics"]


def test_restores_hook_on_impl_exception_and_records_failure(tmp_path):
    original = SyntheticJIT.run
    with pytest.raises(RuntimeError):
        with CompilationCapture(triton_fixture(), tmp_path / "case", {}):
            raise RuntimeError("unit-only")
    assert SyntheticJIT.run is original
    assert "implementation exit" in (tmp_path / "case/observation.json").read_text()


def test_unsupported_hook_never_becomes_successful_observation(tmp_path):
    with CompilationCapture(SimpleNamespace(), tmp_path / "case", {}) as capture:
        pass
    assert not capture.payload["launches"] and capture.payload["diagnostics"]


def test_jit_failure_restores_hook(tmp_path):
    class Broken(SyntheticJIT):
        def run(self, *args, **kwargs):
            raise ValueError("synthetic compiler failure")

    original = Broken.run
    with pytest.raises(ValueError):
        with CompilationCapture(triton_fixture(Broken), tmp_path / "case", {}) as capture:
            Broken().run(grid=(1,))
    assert Broken.run is original
    assert any("JIT invocation failed" in d for d in capture.payload["diagnostics"])


def test_synthetic_complete_evidence_exercises_gate_but_never_taskbook_acceptance(suite):
    report = suite.audit()
    assert report["complete"] and report["all_pass"] and report["verified_npu_coverage_target_met"]
    assert report["compiled_cases"] == report["validated_cases"] == 1
    assert report["compilation"]["rate"] == report["execution_validation"]["rate"] == 1
    assert report["compilation_witnesses"]["FA-MATH-ADD"] == [suite.case.code]
    assert "tt.load" in report["observed_ir_operations"]
    assert not report["taskbook_accepted"]


@pytest.fixture
def regression_suite(suite):
    extra = suite.case.model_copy(deep=True)
    extra.code = "synthetic-unselected-sub"
    extra.operator = "Sub"
    extra.feature_atoms = ["FA-MATH-SUB"]
    manifest = build_manifest(suite.corpus, "synthetic-unit-only", "synthetic-unit-only", [], [suite.case, extra])
    for root in (suite.corpus, suite.results):
        (root / "manifest.json").write_text(manifest.model_dump_json())
        json_write(root / "coverage-plan.json", {"active_atoms": [*suite.case.feature_atoms, "FA-MATH-SUB"]})
    data = json.loads(suite.path.read_text())
    data["manifest_hash"] = manifest.manifest_hash
    json_write(suite.path, data)
    index_run(suite.run)
    summary = json.loads((suite.results / "suite-summary.json").read_text())
    summary.update(
        manifest_hash=manifest.manifest_hash, validation_scope="operator_regression", selected_operators=["Add"]
    )
    summary["operators"]["Add"]["manifest_hash"] = manifest.manifest_hash
    json_write(suite.results / "suite-summary.json", summary)
    return suite


def test_regression_can_pass_without_shrinking_denominator_or_full_qualification(regression_suite):
    suite = regression_suite
    report = audit_suite(suite.corpus, suite.results, suite.submission, operator_regression=True)
    assert report["selected_operator_validation_passed"]
    assert report["selected_expected_cases"] == report["validated_cases"] == 1
    assert report["expected_cases"] == 2
    assert report["compilation"]["total"] == report["execution_validation"]["total"] == 4
    assert report["compilation"]["rate"] == report["execution_validation"]["rate"] == 3 / 4
    assert not report["complete"] and not report["all_pass"]
    assert not report["verified_npu_coverage_target_met"] and not report["taskbook_accepted"]
    assert not report["errors"]
    full = suite.audit()
    assert not full["selected_operator_validation_passed"]
    assert "Sub" in full["errors"]


@pytest.mark.parametrize("change", ["wrong-result", "reference-failure", "missing-evidence"])
def test_regression_preserves_runtime_and_evidence_requirements(regression_suite, change):
    suite = regression_suite
    if change == "wrong-result":
        row = json.loads((suite.run / "results.jsonl").read_text())
        row["status"] = "WRONG_RESULT"
        json_write(suite.run / "results.jsonl", row)
    elif change == "reference-failure":
        path = suite.results / "suite-summary.json"
        summary = json.loads(path.read_text())
        summary["operators"]["Add"]["reference_gate_returncodes"]["npu:0"] = 1
        json_write(path, summary)
    else:
        suite.path.unlink()
    index_run(suite.run)
    report = audit_suite(suite.corpus, suite.results, suite.submission, operator_regression=True)
    assert not report["selected_operator_validation_passed"]
    assert not report["verified_npu_coverage_target_met"]


@pytest.mark.parametrize("selected", [[], ["Add", "Add"], ["Unknown"], ["Add", "Sub"]])
def test_regression_rejects_empty_duplicate_foreign_or_full_scope(regression_suite, selected):
    suite = regression_suite
    path = suite.results / "suite-summary.json"
    summary = json.loads(path.read_text())
    summary["selected_operators"] = selected
    json_write(path, summary)
    with pytest.raises(ValueError, match="invalid operator regression scope"):
        audit_suite(suite.corpus, suite.results, suite.submission, operator_regression=True)


@pytest.mark.parametrize("change", ["wrong-result", "unsynchronized", "reference-failure"])
def test_compile_credit_does_not_imply_execution_credit(suite, change):
    if change == "wrong-result":
        row = json.loads((suite.run / "results.jsonl").read_text())
        row["status"] = "WRONG_RESULT"
        json_write(suite.run / "results.jsonl", row)
    elif change == "unsynchronized":
        obs = json.loads(suite.path.read_text())
        obs["synchronized"] = False
        json_write(suite.path, obs)
    else:
        summary = json.loads((suite.results / "suite-summary.json").read_text())
        summary["operators"]["Add"]["reference_gate_returncodes"]["npu:0"] = 1
        json_write(suite.results / "suite-summary.json", summary)
    index_run(suite.run)
    report = suite.audit()
    assert report["compiled_cases"] == 1 and report["validated_cases"] == 0
    assert report["execution_validation"]["rate"] == 0
    assert not report["verified_npu_coverage_target_met"]


@pytest.mark.parametrize(
    "change",
    [
        "case-hash",
        "run-id",
        "device",
        "dtype",
        "shape",
        "layout",
        "diagnostic",
        "no-ir",
        "no-backend",
        "no-launch",
        "source",
        "artifact-hash",
    ],
)
def test_invalid_compilation_evidence_gets_no_credit(suite, change):
    obs = json.loads(suite.path.read_text())
    if change == "case-hash":
        obs["case_hash"] = "foreign"
    if change == "run-id":
        obs["run_id"] = 999
    if change == "device":
        obs["device"] = "cpu"
    if change == "dtype":
        obs["input_metadata"][0]["dtype"] = "torch.float16"
    if change == "shape":
        obs["input_metadata"][0]["shape"] = [3, 2]
    if change == "layout":
        obs["input_metadata"][0]["stride"] = [6, 2]
    if change == "diagnostic":
        obs["diagnostics"] = ["one unobserved launch"]
    if change in ("no-ir", "no-backend"):
        stage = "ttir" if change == "no-ir" else "npubin"
        obs["launches"][0]["artifacts"] = [a for a in obs["launches"][0]["artifacts"] if a["stage"] != stage]
    if change == "no-launch":
        obs["launches"] = []
    if change in ("source", "artifact-hash"):
        entry = obs["launches"][0]["source"]
        blob = suite.path.parent / entry["path"]
        blob.write_text("def foreign_kernel(X):\n    return 0\n")
        if change == "source":
            entry["sha256"] = sha(blob.read_bytes())
    json_write(suite.path, obs)
    index_run(suite.run)
    report = suite.audit()
    assert report["compiled_cases"] == report["validated_cases"] == 0
    assert report["errors"] and not report["verified_npu_coverage_target_met"]


@pytest.mark.parametrize(
    "change", ["hash", "missing", "duplicate", "foreign-row", "command", "env", "child", "probe", "unindexed"]
)
def test_invalid_run_evidence_cannot_qualify(suite, change):
    if change == "hash":
        (suite.run / "results.jsonl").write_text("{}\n")
    elif change in ("missing", "duplicate", "foreign-row"):
        row = (suite.run / "results.jsonl").read_text()
        (suite.run / "results.jsonl").write_text(
            "" if change == "missing" else row * 2 if change == "duplicate" else row.replace("synthetic-add", "foreign")
        )
    elif change == "command":
        summary = json.loads((suite.results / "suite-summary.json").read_text())
        summary["operators"]["Add"]["run_returncode"] = 1
        json_write(suite.results / "suite-summary.json", summary)
    elif change == "env":
        env = json.loads((suite.run / "env.json").read_text())
        env["npu_count"] = 0
        json_write(suite.run / "env.json", env)
    elif change == "child":
        integrity = json.loads((suite.run / "integrity.json").read_text())
        integrity["child"]["exit_code"] = 1
        json_write(suite.run / "integrity.json", integrity)
    elif change == "probe":
        json_write(suite.run / "events.jsonl", dict(event="probe", ok=False))
    elif change == "unindexed":
        index = json.loads((suite.run / "artifacts.json").read_text())
        index["files"] = [e for e in index["files"] if e["path"] != "events.jsonl"]
        json_write(suite.run / "artifacts.json", index)
    if change not in ("hash", "unindexed"):
        index_run(suite.run)
    report = suite.audit()
    assert not report["complete"] and not report["verified_npu_coverage_target_met"]


@pytest.mark.parametrize("change", ["submission", "manifest", "archive", "health"])
def test_global_identity_mismatch_is_hard_failure(suite, change):
    if change == "submission":
        (suite.submission / "kernel.py").write_text("def altered(): pass\n")
    elif change == "manifest":
        data = json.loads((suite.results / "manifest.json").read_text())
        data["cases"][0]["seed"] += 1
        json_write(suite.results / "manifest.json", data)
    elif change == "archive":
        json_write(suite.results / "evaluation-assets.json", {})
    else:
        json_write(suite.results / "capabilities/capabilities.json", {"probes": {"health": {"status": "FAIL"}}})
    with pytest.raises(ValueError):
        suite.audit()


def test_static_structure_and_input_rank_do_not_certify_ir_patterns():
    obs = dict(ir=IR, source=SOURCE, launches=[{"grid": [8]}])
    assert feature_witness("FA-MATH-ADD", obs)[0]
    for atom in (
        "FA-PAT-VEC-LOAD",
        "FA-PROG-GRID4D",
        "FA-PROG-GRID2D",
        "FA-CTRL-STATIC-UNROLL",
        "FA-LINALG-DOT",
        "FA-CONCUR-ACCUM",
        "FA-FUSE-INTERM-ELIDE",
    ):
        assert not feature_witness(atom, obs)[0], atom
    obs["ir"] = "tt.dot tt.atomic_rmw"
    assert feature_witness("FA-LINALG-DOT", obs)[0] and feature_witness("FA-CONCUR-ACCUM", obs)[0]


def test_blob_symlink_and_escape_are_rejected(tmp_path):
    root = tmp_path / "capture/case"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"synthetic")
    (root / "link").symlink_to(outside)
    for relative in ("link", "../../outside", str(outside)):
        with pytest.raises(ValueError):
            read_blob(root, dict(path=relative, sha256=sha(b"synthetic")))


def test_where_layout_is_checked_on_data_tensor_instead_of_condition(suite):
    case = suite.case.model_copy(deep=True)
    case.operator = "Where"
    case.contiguous = False
    data = json.loads(suite.path.read_text())
    data["operator"] = "Where"
    data["main_input_index"] = 1
    data["input_metadata"] = [
        {"shape": [2, 3], "stride": [3, 1], "dtype": "torch.bool"},
        {"shape": [2, 3], "stride": [6, 2], "dtype": "torch.float32"},
    ]
    json_write(suite.path, data)
    assert observation(case, suite.path, data["manifest_hash"], 11)["launches"]
    data["main_input_index"] = 0
    json_write(suite.path, data)
    with pytest.raises(ValueError, match="main input identity"):
        observation(case, suite.path, data["manifest_hash"], 11)


def test_mixed_compiler_environments_cannot_union_feature_coverage(suite):
    import copy

    case = suite.case.model_copy(deep=True)
    case.code = "synthetic-sub"
    case.operator = "Sub"
    manifest = build_manifest(suite.corpus, "synthetic-unit-only", "synthetic-unit-only", [], [suite.case, case])
    for root in (suite.corpus, suite.results):
        (root / "manifest.json").write_text(manifest.model_dump_json())
    data = json.loads(suite.path.read_text())
    data["manifest_hash"] = manifest.manifest_hash
    json_write(suite.path, data)
    index_run(suite.run)
    summary = json.loads((suite.results / "suite-summary.json").read_text())
    summary["manifest_hash"] = manifest.manifest_hash
    summary["operators"]["Add"]["manifest_hash"] = manifest.manifest_hash
    summary["operators"]["Sub"] = copy.deepcopy(summary["operators"]["Add"])
    json_write(suite.results / "suite-summary.json", summary)
    run = suite.results / "Sub/run-11"
    make_capture(run, case, manifest.manifest_hash)
    json_write(run / "results.jsonl", dict(case_code=case.code, operator=case.operator, run_id=11, status="PASS"))
    for name in ("integrity.json", "events.jsonl"):
        (run / name).write_bytes((suite.run / name).read_bytes())
    env = json.loads((suite.run / "env.json").read_text())
    env["compiler_version"] = "another-unit-compiler"
    json_write(run / "env.json", env)
    index_run(run)
    report = suite.audit()
    assert "environment" in report["errors"]
    assert report["compiled_cases"] == report["validated_cases"] == 0
    assert not report["compilation_witnesses"] and not report["execution_witnesses"]
    assert not report["verified_npu_coverage_target_met"]


def test_runtime_tensor_binding_tracks_storage_aliases_and_constexpr_arguments(tmp_path):
    import torch

    class BoundJIT(SyntheticJIT):
        arg_names = ["X", "Out", "N"]
        constexprs = [2]

    x = torch.randn(16)
    out = torch.empty_like(x)
    directory = tmp_path / "compilation/bound"
    with CompilationCapture(triton_fixture(BoundJIT), directory, {}) as capture:
        capture.bind_inputs([x])
        BoundJIT().run(x, out, 16, grid=(1,))
        capture.bind_outputs([out])
        args = capture.payload["launches"][0]["arguments"]
        assert args["X"]["tensor_id"] == capture.payload["input_bindings"][0]["tensor_id"]
        assert args["Out"]["tensor_id"] == capture.payload["output_bindings"][0]["tensor_id"]
        assert args["X"]["storage_id"] != args["Out"]["storage_id"]
        assert args["N"] == dict(kind="scalar", value=16, constexpr=True)
        assert capture.payload["launches"][0]["runtime_arg_names"] == ["X", "Out"]
        capture.bind_outputs([x[1:]])
        alias = capture.payload["output_bindings"][0]
        assert alias["tensor_id"] != args["X"]["tensor_id"] and alias["storage_id"] == args["X"]["storage_id"]
    assert not capture.payload["diagnostics"]


def write_pass_case_rows(command, cases):
    """Materialize synthetic PASS rows; this is not hardware evidence."""
    if 'run' not in command:
        return
    operator = command[command.index('--operators') + 1]
    run = Path(command[command.index('--runs') + 1]) / 'run-1'
    run.mkdir()
    rows = [dict(case_code=case['code'], operator=operator, run_id=1, status='PASS')
            for case in cases if case['operator'] == operator]
    (run / 'results.jsonl').write_text('\n'.join(map(json.dumps, rows)))


def configure_target_suite(monkeypatch, module, corpus, output, operators):
    """Set the shared target-pattern entry contract for synthetic suite tests."""
    import sys

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv('TRITON_EVAL_REQUIRE_TARGET_PATTERNS', '1')
    monkeypatch.setenv('TRITON_EVAL_REQUIRE_BACKEND_PATTERNS', '0')
    monkeypatch.setattr(sys, 'argv', [
        'suite', '--corpus', str(corpus), '--submission',
        str(fixture_path(root / 'examples/coverage-v18-ascend-submission')), '--operators',
        ','.join(operators), '--out', str(output)])
