# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""分区审计必须保留原始 run_id，且不能漏算其他算子。"""

import importlib.util
import json
import sys
from pathlib import Path

from triton_eval.fixture_sources import fixture_path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
MODULE_SPEC = importlib.util.spec_from_file_location("run_ascend_suite", SCRIPTS / "run_ascend_suite.py")
SUITE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(SUITE)


def record(code, operator, run_id):
    return json.dumps({"case_code": code, "operator": operator, "status": "PASS", "run_id": run_id})


def test_each_partition_keeps_its_own_run_identity():
    manifest = {"manifest_hash": "frozen", "cases": [
        {"code": "a", "operator": "Add"}, {"code": "b", "operator": "Mul"}]}
    first = SUITE.audit_partition(manifest, "Add", [record("a", "Add", 11)])
    second = SUITE.audit_partition(manifest, "Mul", [record("b", "Mul", 12)])
    assert first["all_pass"] and second["all_pass"]
    assert first["run_ids"] == [11] and second["run_ids"] == [12]
    assert first["manifest_hash"] == "frozen"
    assert not SUITE.audit_partition(manifest, "Add", [record("b", "Mul", 12)])["complete"]
    assert not SUITE.audit_partition(manifest, "Mul", [])["complete"]


def test_child_failure_does_not_stop_remaining_partitions(tmp_path, monkeypatch):
    output = tmp_path / "suite"
    calls = []

    def execute(command, path, timeout=None):
        calls.append([str(value) for value in command])
        if "run" in command:
            operator = command[command.index("--operators") + 1]
            partition = Path(command[command.index("--runs") + 1]) / "run-1"
            partition.mkdir()
            payload = json.loads((SCRIPTS.parent / "corpus/v0.7-ascend-contract/manifest.json").read_text())
            run_id = 41 if operator == "VectorAdd" else 42
            lines = [record(case["code"], operator, run_id) for case in payload["cases"]
                     if case["operator"] == operator]
            if operator == "VectorAdd":
                lines.pop()
            (partition / "results.jsonl").write_text("\n".join(lines))
            return int(operator == "VectorAdd")
        return 0

    monkeypatch.setattr(SUITE, "run_logged", execute)
    monkeypatch.setattr("triton_eval.corpus_assets.verify_assets", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["suite", "--corpus", str(SCRIPTS.parent / "corpus/v0.7-ascend-contract"),
                                      "--operators", "VectorAdd,VectorCmp", "--out", str(output)])
    assert SUITE.main() == 1
    summary = json.loads((output / "suite-summary.json").read_text())
    assert not summary["complete"]
    assert summary["operators"]["VectorCmp"]["all_pass"]
    score_runs = [command[command.index("--run") + 1] for command in calls if "score" in command]
    assert score_runs == ["41", "42"]


def test_reference_failure_is_isolated_when_requested(tmp_path, monkeypatch):
    output = tmp_path / "suite"

    def execute(command, path, timeout=None):
        command = [str(value) for value in command]
        if "--device" in command and "--operators" in command:
            return int(command[command.index("--operators") + 1] == "VectorAdd" and "npu:0" in command)
        if "run" in command:
            operator = command[command.index("--operators") + 1]
            partition = Path(command[command.index("--runs") + 1]) / "run-1"
            partition.mkdir()
            payload = json.loads((SCRIPTS.parent / "corpus/v0.7-ascend-contract/manifest.json").read_text())
            (
                partition /
                "results.jsonl").write_text(
                "\n".join(
                    record(
                        case["code"],
                        operator,
                        42) for case in payload["cases"] if case["operator"] == operator))
        return 0

    monkeypatch.setattr(SUITE, "run_logged", execute)
    monkeypatch.setattr("triton_eval.corpus_assets.verify_assets", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys,
                        "argv",
                        ["suite",
                         "--corpus",
                         str(SCRIPTS.parent / "corpus/v0.7-ascend-contract"),
                            "--operators",
                            "VectorAdd,VectorCmp",
                            "--isolate-reference-gates",
                            "--out",
                            str(output)])
    assert SUITE.main() == 1
    summary = json.loads((output / "suite-summary.json").read_text())
    assert summary["operators"]["VectorAdd"]["reference_gate_failure"]["device"] == "npu:0"
    assert summary["operators"]["VectorCmp"]["all_pass"]
    assert summary["operators"]["VectorCmp"]["reference_gate_returncodes"] == {"cpu": 0, "npu:0": 0}


def test_partial_official_run_does_not_require_fifty_operator_qualification(tmp_path, monkeypatch):
    from triton_eval.official_catalog import VERSION

    output = tmp_path / "suite"
    calls = []
    corpus = SCRIPTS.parent / "corpus" / VERSION
    payload = json.loads((corpus / "manifest.json").read_text())
    # Current-framework fixture, separate from the historical frozen corpus.
    import shutil
    from triton_eval.corpus_assets import freeze_assets
    from triton_eval.corpusmod import load_manifest
    destination = tmp_path / "synthetic-current-framework-corpus"
    shutil.copytree(corpus, destination)
    corpus = destination
    (corpus / "evaluation-assets.json").unlink()
    freeze_assets(corpus, load_manifest(corpus / "manifest.json").manifest_hash)

    def execute(command, path, timeout=None):
        calls.append([str(value) for value in command])
        if "run" in command:
            partition = Path(command[command.index("--runs") + 1]) / "run-1"
            partition.mkdir()
            (partition / "results.jsonl").write_text("\n".join(
                record(case["code"], "Pow", 41) for case in payload["cases"] if case["operator"] == "Pow"))
        return 0

    monkeypatch.setattr(SUITE, "run_logged", execute)
    monkeypatch.setattr(sys, "argv", ["suite", "--corpus", str(corpus), "--operators", "Pow", "--out", str(output)])
    assert SUITE.main() == 0
    summary = json.loads((output / "suite-summary.json").read_text())
    assert summary["all_pass"] and summary["expected_cases"] == 20
    assert "partial operator run" in summary["qualification_skipped"]
    assert not any(any("qualify_official_results.py" in item for item in command) for command in calls)
    assert not (output / "qualification.json").exists()


def test_failed_static_readiness_stops_before_npu_or_partition_run(tmp_path, monkeypatch):
    import pytest
    corpus = SCRIPTS.parent / 'corpus/v0.16-backend-pattern-certification-candidates'
    called = []
    monkeypatch.setattr(SUITE, 'run_logged', lambda *args, **kwargs: called.append(args))
    monkeypatch.setattr('triton_eval.corpus_assets.verify_assets', lambda *args, **kwargs: None)

    def reject(_):
        raise ValueError('synthetic insufficient static coverage')
    monkeypatch.setattr('triton_eval.coverage_gate.static_readiness', reject)
    output = tmp_path / 'must-not-start'
    monkeypatch.setattr(sys, 'argv', ['suite', '--corpus', str(corpus), '--out', str(output)])
    with pytest.raises(ValueError, match='insufficient static coverage'):
        SUITE.main()
    assert not called and not output.exists()


def test_coverage_summary_stays_immutable_after_audit(tmp_path, monkeypatch):
    corpus = SCRIPTS.parent / 'corpus/v0.18-scan-certification-candidates'
    output = tmp_path / 'synthetic-coverage-suite'
    payload = json.loads((corpus / 'manifest.json').read_text())
    audited = []
    monkeypatch.setenv('TRITON_CACHE_DIR', 'synthetic-before-suite')

    def execute(command, path, timeout=None):
        command = [str(value) for value in command]
        if 'run' in command:
            run = Path(command[command.index('--runs') + 1]) / 'run-1'
            run.mkdir()
            (run / 'results.jsonl').write_text('\n'.join(record(c['code'], 'Abs', 1)
                                                         for c in payload['cases'] if c['operator'] == 'Abs'))
        if any('audit_coverage_results.py' in arg for arg in command):
            audited.append((output / 'suite-summary.json').read_bytes())
        return 0

    monkeypatch.setattr(SUITE, 'run_logged', execute)
    monkeypatch.setattr(sys, 'argv', ['suite', '--corpus', str(corpus), '--operators', 'Abs', '--out', str(output)])
    assert SUITE.main() == 0
    assert audited == [(output / 'suite-summary.json').read_bytes()]
    assert (output / 'coverage-plan.json').read_bytes() == (corpus / 'coverage-plan.json').read_bytes()


def test_failed_submission_preflight_stops_all_partitions(tmp_path, monkeypatch):
    corpus = SCRIPTS.parent / 'corpus/v0.18-scan-certification-candidates'
    output = tmp_path / 'bad-submission-suite'
    calls = []

    def reject(command, path, timeout=None):
        calls.append([str(value) for value in command])
        path.write_text('synthetic SUB-LOD-007 Neg signature mismatch\n')
        return 1
    monkeypatch.setattr(SUITE, 'run_logged', reject)
    monkeypatch.setattr(sys, 'argv', ['suite', '--corpus', str(corpus),
                                      '--submission',
                                      str(fixture_path(SCRIPTS.parent / 'examples/coverage-v18-ascend-submission')),
                                      '--out', str(output)])
    assert SUITE.main() == 1
    assert len(calls) == 1
    assert any('check_submission_readiness.py' in value for value in calls[0])
    assert '--load' in calls[0]
    assert not (output / 'suite-summary.json').exists()
    assert not list(output.glob('*/results.jsonl'))
