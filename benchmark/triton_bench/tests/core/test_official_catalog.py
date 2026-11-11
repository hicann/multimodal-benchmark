# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Admission tests distinguish source provenance, mathematical contracts and NPU evidence."""

import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

from triton_eval.official_catalog import BY_NAME, CANDIDATES, COMPILER_ONLY, SOURCE_COMMIT, VERSION


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus" / VERSION
SYNTHETIC_ENVIRONMENT = {"env_health": "ok", "npu_model": "synthetic-unit-fixture", "npu_count": 1,
                         "cann_version": "synthetic-unit-fixture", "arch": "aarch64",
                         "compiler_name": "synthetic-unit-fixture", "compiler_version": "unit-test",
                         "triton_import_version": "unit-test", "torch_version": "unit-test",
                         "torch_npu_version": "unit-test", "venv_lock_hash": "unit-test"}
sys.path.insert(0, str(ROOT / "scripts"))
import build_official_corpus as builder
import qualify_official_results as qualification

from triton_eval.fixture_sources import fixture_path


def test_candidates_are_unique_and_source_locked():
    sources = json.loads((ROOT / "reference/ascend-official/sources-v08.json").read_text())
    assert len(CANDIDATES) == len(BY_NAME) == 68
    assert {source["name"] for source in sources["operators"]} == set(BY_NAME)
    assert sources["source_commit"] == SOURCE_COMMIT
    assert sum(source["commit"] == SOURCE_COMMIT for source in sources["operators"]) == 67
    assert all(len(source["sha256"]) == 64 for source in sources["operators"])
    assert all(not source["upstream_presence_is_npu_qualification"] for source in sources["operators"])
    assert "VectorAdd" not in BY_NAME
    assert "GatherScatter" not in BY_NAME
    assert BY_NAME["BitwiseNot"].source.endswith("test_invert.py")
    assert "_examples/" in BY_NAME["Gather"].source


def test_cann_correspondence_keeps_projections_and_primitive_only_sources_explicit():
    audit = json.loads((ROOT / "reference/ascend-official/cann-correspondence-v08.json").read_text())
    mappings = {item["name"]: item for item in audit["operators"]}
    assert set(mappings) == set(BY_NAME)
    assert sum(item["cann_source"] is not None for item in mappings.values()) == 64
    identities = {(item["cann_source"]["repository"], item["cann_source"]["path"])
                  for item in mappings.values() if item["cann_source"] is not None}
    assert len(identities) == 64
    assert {name for name, item in mappings.items() if item["cann_source"] is None} == {
        "Exp2", "Log2", "Log10", "XorSum"}
    for name in ("SortValues", "TopKValues", "RmsNorm", "LayerNorm", "Gelu"):
        assert "qualification_mapping" in mappings[name]


def test_each_operator_has_twenty_distinct_cases_and_mandatory_boundaries():
    cases, reports = builder.select_cases()
    assert len(cases) == 1360
    assert set(Counter(case.operator for case in cases).values()) == {20}
    signatures = {(case.operator, case.dtype, tuple(case.shape), json.dumps(case.attrs, sort_keys=True),
                   json.dumps(case.value_range, sort_keys=True), case.contiguous) for case in cases}
    assert len(signatures) == len(cases)
    codes = {case.code for case in cases}
    assert len(codes) == len(cases)
    assert all(set(report["mandatory_cases"]) <= codes for report in reports.values())
    assert all(len(report["mandatory_cases"]) == 4 for report in reports.values())


@pytest.mark.parametrize("name,invalid", [("ArgMin", "int64"), ("Where", "bool")])
def test_generation_keeps_data_dtypes_separate_from_indices_and_predicates(name, invalid):
    from triton_eval.casegen import derive_cases
    from triton_eval.official_catalog import specification

    candidate = BY_NAME[name]
    spec = specification(candidate)
    with pytest.raises(ValueError, match="dtype"):
        spec.validate_case_params(invalid, [2, 3], spec.resolve_attrs({}))
    assert {case.dtype for case in derive_cases(spec, count=20)} == set(candidate.dtypes)


@pytest.mark.torch
@pytest.mark.parametrize("name,expected", [("FloorDiv", -4), ("Mod", -1)])
def test_signed_division_semantics(name, expected):
    import torch
    from triton_eval.official_reference import reference
    from triton_eval.official_reference_cpu import reference as independent

    dtype = torch.int32 if name == "FloorDiv" else torch.float32
    inputs = (torch.tensor([[-7]], dtype=dtype), torch.tensor([[2]], dtype=dtype))
    assert reference(name, *inputs).item() == expected
    assert independent(name, *inputs).item() == expected


@pytest.mark.torch
def test_signbit_preserves_negative_zero_and_arg_ties_choose_first():
    import torch
    from triton_eval.official_reference import reference
    from triton_eval.official_reference_cpu import reference as independent

    values = torch.tensor([[-0.0, 0.0, -1.0, 1.0]])
    expected = torch.tensor([[True, False, True, False]])
    assert torch.equal(reference("Signbit", values), expected)
    assert torch.equal(independent("Signbit", values), expected)
    ties = torch.tensor([[4.0, 4.0, -1.0]])
    assert reference("ArgMax", ties).item() == independent("ArgMax", ties).item() == 0


@pytest.mark.torch
def test_invalid_relationships_fail_before_reference_execution():
    import torch
    from triton_eval.official_reference import validate_inputs

    with pytest.raises(ValueError, match="out of bounds"):
        validate_inputs("Gather", [torch.zeros(2, 3), torch.full((2, 1), 3, dtype=torch.int64)], {"dim": -1})
    with pytest.raises(ValueError, match="contracting"):
        validate_inputs("Matmul", [torch.zeros(2, 3, dtype=torch.float16), torch.zeros(4, 5, dtype=torch.float16)], {})
    with pytest.raises(ValueError, match="positive"):
        validate_inputs("Log", [torch.zeros(1, 1)], {})
    with pytest.raises(ValueError, match="overflow"):
        validate_inputs("FloorDiv", [torch.tensor([[-2147483648]], dtype=torch.int32),
                        torch.tensor([[-1]], dtype=torch.int32)], {})


@pytest.mark.torch
def test_logical_inputs_cover_asymmetric_truth_values():
    import torch
    from triton_eval.official_reference import make_inputs

    first, second = make_inputs("LogicalOr", seed=100, dtype=torch.float32, shape=[9, 31], device="cpu", attrs={})
    assert bool(((first != 0) & (second == 0)).any())
    assert bool(((first == 0) & (second != 0)).any())


@pytest.mark.torch
def test_independent_reference_catches_a_biased_primary(monkeypatch):
    from triton_eval import official_reference
    from triton_eval.oracle import active_corpus, configure_corpus
    from triton_eval.refcheck import check_case

    previous = active_corpus()
    original = official_reference.reference
    monkeypatch.setattr(official_reference, "reference", lambda name, * \
                        inputs, **attrs: original(name, *inputs, **attrs) + 1)
    try:
        configure_corpus(CORPUS)
        case = next(case for case in builder.select_cases()[0] if case.operator == "Abs")
        assert "第二参考输出" in check_case(case)
    finally:
        configure_corpus(previous)


def test_portable_assets_reject_reference_tampering(tmp_path):
    from triton_eval.corpus_assets import copy_reference_tree, verify_assets
    from triton_eval.corpusmod import load_manifest

    destination = tmp_path / "downloaded-corpus"
    shutil.copytree(CORPUS, destination)
    copy_reference_tree(CORPUS, destination)
    manifest = load_manifest(destination / "manifest.json")
    # Exercise current snapshot tamper checks on a synthetic copy; history stays frozen.
    from triton_eval.corpus_assets import freeze_assets
    (destination / "evaluation-assets.json").unlink()
    freeze_assets(destination, manifest.manifest_hash)
    verify_assets(destination, manifest.manifest_hash)
    reference = next((destination / "operators").glob("*/ref_cpu.py"))
    reference.write_text("def reference(*inputs, **attrs):\n    return inputs[0]\n")
    with pytest.raises(ValueError, match="冻结指纹"):
        verify_assets(destination, manifest.manifest_hash)


def _evidence(tmp_path, statuses=None, healthy=True):
    case_codes = [f"abs-{index}" for index in range(20)]
    payload = {"manifest_hash": "candidate", "cases": [{"code": code, "operator": "Abs"} for code in case_codes]}
    directory = tmp_path / "Abs" / "run-1"
    directory.mkdir(parents=True)
    rows = [{"case_code": code, "operator": "Abs", "status": "PASS", "run_id": 17} for code in case_codes]
    if statuses:
        rows[0]["status"] = statuses
    (directory / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (directory / "integrity.json").write_text(json.dumps({"run_status": "OK",
                                                          "env_mismatches": [],
                                                          "submission_changed_during_run": [],
                                                          "child": {"exit_code": 0,
                                                                    "signal": None,
                                                                    "killed_by_us": False}}))
    environment = dict(SYNTHETIC_ENVIRONMENT)
    if not healthy:
        environment.update(npu_model=None, npu_count=0)
    (directory / "env.json").write_text(json.dumps(environment))
    summary = {"reference_gate_returncodes": {"cpu": 0, "npu:0": 0},
               "run_returncode": 0, "score_returncode": 0, "report_returncode": 0}
    return payload, directory.parent, summary


def test_summary_pass_cannot_hide_a_compile_failure(tmp_path):
    payload, partition, summary = _evidence(tmp_path, "COMPILE_FAIL")
    report = qualification.audit_operator(payload, "Abs", partition, summary)
    assert not report["eligible"]
    assert report["by_status"] == {"COMPILE_FAIL": 1, "PASS": 19}


def test_twenty_passes_without_real_npu_environment_are_ineligible(tmp_path):
    payload, partition, summary = _evidence(tmp_path, healthy=False)
    report = qualification.audit_operator(payload, "Abs", partition, summary)
    assert report["all_pass"] and not report["eligible"]


def test_missing_compiler_identity_is_ineligible(tmp_path):
    payload, partition, summary = _evidence(tmp_path)
    path = partition / "run-1/env.json"
    environment = json.loads(path.read_text())
    environment.pop("triton_import_version")
    path.write_text(json.dumps(environment))
    report = qualification.audit_operator(payload, "Abs", partition, summary)
    assert not report["eligible"]
    assert "missing compiler/runtime environment identity" in report["ineligibility_reasons"]


def test_missing_gate_and_dirty_source_integrity_are_ineligible(tmp_path):
    payload, partition, summary = _evidence(tmp_path)
    assert qualification.audit_operator(payload, "Abs", partition, summary)["eligible"]
    summary.pop("reference_gate_returncodes")
    assert not qualification.audit_operator(payload, "Abs", partition, summary)["eligible"]
    summary["reference_gate_returncodes"] = {"cpu": 0, "npu:0": 0}
    integrity_path = partition / "run-1/integrity.json"
    integrity = json.loads(integrity_path.read_text())
    integrity["submission_changed_during_run"] = ["operators/kernels.py"]
    integrity_path.write_text(json.dumps(integrity))
    assert not qualification.audit_operator(payload, "Abs", partition, summary)["eligible"]


def test_fifty_selection_requires_enough_whole_operators_and_preserves_families():
    all_names = set(BY_NAME)
    selected = qualification.choose_operators(all_names)
    assert len(selected) == len(set(selected)) == 50
    assert not set(selected) & COMPILER_ONLY
    assert {BY_NAME[name].category for name in selected} == {candidate.category for candidate in CANDIDATES}
    assert len(qualification.choose_operators(set(sorted(all_names - COMPILER_ONLY)[:49]))) == 49
    assert len(qualification.choose_operators(set(sorted(all_names - COMPILER_ONLY)[:49]) | COMPILER_ONLY)) == 49


def test_generated_submission_passes_static_policy():
    from triton_eval.validator import ValidationResult, validate_static

    result = ValidationResult("official")
    validate_static(fixture_path(ROOT / "examples/official-ascend-submission"), result)
    assert result.ok, result.render()


@pytest.mark.parametrize("passing_count", [49, 50])
def test_real_qualification_pipeline_freezes_only_after_fifty_whole_operators(tmp_path, passing_count):
    from triton_eval.corpusmod import load_manifest, verify_corpus_files

    corpus = tmp_path / "synthetic-current-framework-corpus"
    shutil.copytree(CORPUS, corpus)
    from triton_eval.corpus_assets import freeze_assets
    (corpus / "evaluation-assets.json").unlink()
    manifest = load_manifest(corpus / "manifest.json")
    freeze_assets(corpus, manifest.manifest_hash)
    results = tmp_path / "synthetic-unit-evidence"
    results.mkdir()
    for filename in ("manifest.json", "evaluation-assets.json"):
        shutil.copyfile(corpus / filename, results / filename)
    submission = fixture_path(ROOT / "examples/official-ascend-submission")
    snapshot = {path.relative_to(submission).as_posix(): qualification.digest(path)
                for path in submission.rglob("*") if path.is_file() and path.suffix in (".py", ".yaml", ".yml")}
    (results / "submission-sha256.json").write_text(json.dumps(snapshot))
    suite = {"manifest_hash": manifest.manifest_hash, "operators": {}}
    countable = [candidate.name for candidate in CANDIDATES if candidate.name not in COMPILER_ONLY]
    passing = set(countable[:passing_count])
    for index, candidate in enumerate(CANDIDATES):
        directory = results / candidate.name / "run-1"
        directory.mkdir(parents=True)
        rows = [{"case_code": case.code, "operator": candidate.name, "run_id": index + 1,
                 "status": "PASS" if candidate.name in passing or candidate.name in COMPILER_ONLY else "COMPILE_FAIL"}
                for case in manifest.cases if case.operator == candidate.name]
        (directory / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        (directory / "integrity.json").write_text(json.dumps({"run_status": "OK",
                                                              "env_mismatches": [],
                                                              "submission_changed_during_run": [],
                                                              "child": {"exit_code": 0,
                                                                        "signal": None,
                                                                        "killed_by_us": False}}))
        (directory / "env.json").write_text(json.dumps(SYNTHETIC_ENVIRONMENT))
        suite["operators"][candidate.name] = {"reference_gate_returncodes": {"cpu": 0, "npu:0": 0},
                                              "run_returncode": 0, "score_returncode": 0, "report_returncode": 0}
    (results / "suite-summary.json").write_text(json.dumps(suite))
    report = qualification.qualify(corpus, submission, results)
    assert report["eligible_count"] == passing_count
    assert report["target_met"] == (passing_count == 50)
    destination = results / "frozen-official50"
    assert destination.exists() == (passing_count == 50)
    if passing_count == 50:
        frozen = load_manifest(destination / "manifest.json")
        assert len(frozen.cases) == 1000
        assert len(frozen.operators) == 50
        verify_corpus_files(destination, frozen)


@pytest.mark.torch
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_argmax_ties_across_rows_and_padding_boundary(keepdim, dtype_name):
    import torch
    from triton_eval.official_reference import reference
    from triton_eval.official_reference_cpu import reference as independent

    values = torch.tensor([[2, 2, 1, 0, 2, -1, 2], [-2, -1, 3, 3, 3, 0, 3],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=getattr(torch, dtype_name))
    actual = reference("ArgMax", values, keepdim=keepdim)
    assert torch.equal(actual, independent("ArgMax", values, keepdim=keepdim))
    assert actual.flatten().tolist() == [0, 2, 0]
    assert tuple(actual.shape) == ((3, 1) if keepdim else (3,))


@pytest.mark.torch
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_signbit_uses_bits_even_when_backend_signbit_is_broken(monkeypatch, dtype_name):
    import torch
    from triton_eval.official_reference import reference
    from triton_eval.official_reference_cpu import reference as independent

    values = torch.tensor([[-0.0, 0.0, -1, 1, -0.5, 0.5]], dtype=getattr(torch, dtype_name))
    monkeypatch.setattr(torch, "signbit", lambda value: value < 0)
    assert torch.equal(reference("Signbit", values), independent("Signbit", values))
    assert reference("Signbit", values).tolist() == [[True, False, True, False, True, False]]


@pytest.mark.torch
def test_floor_division_is_exact_at_int32_limits_for_both_denominator_signs(monkeypatch):
    import numpy as np
    import torch
    from triton_eval.official_reference import reference
    from triton_eval.official_reference_cpu import reference as independent

    boundaries = [-2147483648, -2147483647, -16777217, -7, -1, 0, 1, 7,
                  16777217, 2147483646, 2147483647]
    # Exercise the precision loss seen in the old FP32 NPU division path.
    monkeypatch.setattr(torch, "floor_divide", lambda a, b: torch.floor(a.float() / b.float()).to(a.dtype))
    first, second = np.meshgrid(np.array(boundaries, dtype=np.int32), np.array(
        [-2147483648, -127, -16, -3, -2, -1, 1, 2, 3, 16, 127, 2147483647], dtype=np.int32))
    mask = ~((first == -2147483648) & (second == -1))
    inputs = [torch.from_numpy(array[mask].reshape(1, -1)) for array in (first, second)]
    assert torch.equal(reference("FloorDiv", *inputs), independent("FloorDiv", *inputs))


@pytest.mark.torch
def test_floor_division_matches_independent_oracle_for_random_int32_inputs():
    import torch
    from triton_eval.official_reference import reference
    from triton_eval.official_reference_cpu import reference as independent

    generator = torch.Generator().manual_seed(20261023)
    first = torch.randint(-2147483648, 2147483648, (100, 100), generator=generator, dtype=torch.int64).int()
    second = torch.randint(-2147483648, 2147483648, (100, 100), generator=generator, dtype=torch.int64).int()
    second[second == 0] = 1
    second[(first == -2147483648) & (second == -1)] = 1
    assert torch.equal(reference("FloorDiv", first, second), independent("FloorDiv", first, second))


def test_rebuilding_existing_corpus_rejects_changed_judging_assets(tmp_path, monkeypatch):
    destination = tmp_path / "corpus"
    shutil.copytree(CORPUS, destination)
    path = destination / "evaluation-assets.json"
    assets = json.loads(path.read_text())
    assets["files"]["src/triton_eval/refcheck.py"] = "0" * 64
    path.write_text(json.dumps(assets))
    monkeypatch.setattr(sys, "argv", ["build", "--corpus", str(destination)])
    with pytest.raises(ValueError, match="冻结指纹"):
        builder.main()
