# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

import triton_eval.submission_service as service_module
from triton_eval.models import CaseResultRecord, CaseStatus
from triton_eval.store import Store
from triton_eval.submission_service import EvaluationService
from triton_eval.submission_simulation import DEFAULT_SOURCE, Scenario, _entries

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus/v0.3-multi-input-frozen"


def zip_package():
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in _entries(DEFAULT_SOURCE, Scenario("TEST", "编排测试")).items():
            archive.writestr(name, data)
    return stream.getvalue()


@pytest.fixture
def service(tmp_path):
    instance = EvaluationService(tmp_path / "worker", CORPUS, execute=True)
    yield instance
    instance.close()


def create_task(service):
    submission = service.upload(zip_package())
    return service.create_task(dict(submission_id=submission["id"]))


def intercept_engine_processes(monkeypatch, service, handler):
    original = service_module.subprocess.Popen

    def spawn(command, **options):
        assert command[:4] == [sys.executable, "-m", "triton_eval", "--db"]
        database = Path(command[4])
        current = service.task(database.parent.name)
        assert command[5] in {"run", "score", "report"}
        replacement = handler(current, command[5], command)
        return original(command if replacement is None else replacement, **options)

    monkeypatch.setattr(service_module.subprocess, "Popen", spawn)


def engine_fixture(service, task, directory, *, status="PASS", omit_case=False, wrong_identity=False, validation=None):
    run_directory = directory / "runs/run-1"
    run_directory.mkdir(parents=True)
    with Store(directory / "engine.sqlite") as store:
        store.create_schema()
        corpus_id = store.import_corpus(service.manifest)
        compiler_id = store.upsert_submission("fixture-not-real-npu", "test", None, "source")
        run_id = store.create_run(corpus_id, compiler_id, "smoke", {}, "fixture", "OK", run_dir=str(run_directory))
        record_id = store.insert_submission_record(
            submit_id="fixture",
            root_path="fixture",
            operators=["Exp"],
            snapshot_fp="fixture",
            validation_ok=not validation,
            validation=validation or [],
            package_sha256="wrong" if wrong_identity else task["package_sha256"],
        )
        store.set_run_submission_record(run_id, record_id)
        cases = [case for case in service.manifest.cases if case.code in task["case_codes"]]
        if omit_case:
            cases = cases[:-1]
        records = []
        for case in cases:
            record = CaseResultRecord(
                run_id=run_id, case_code=case.code, operator=case.operator, status=CaseStatus(status)
            )
            store.insert_case_result(record)
            records.append(record.model_dump_json())
        (run_directory / "results.jsonl").write_text("\n".join(records) + "\n")
        store.finish_run(run_id, "INVALID" if validation else "OK")


@pytest.mark.parametrize(
    "case_status,expected_score",
    [("PASS", 100), ("WRONG_RESULT", 0), ("RUNTIME_FAIL", 0), ("SUBMISSION_INVALID", None)],
)
def test_adapter_uses_real_scorer_and_report_but_fixture_device_results(
    service, monkeypatch, case_status, expected_score
):
    task = create_task(service)
    phases = []

    def invoke(current, phase, command):
        phases.append(phase)
        if phase == "run":
            engine_fixture(service, current, Path(command[4]).parent, status=case_status)
            return [sys.executable, "-c", "raise SystemExit(0)"]
        return None

    intercept_engine_processes(monkeypatch, service, invoke)
    assert service.work_once()
    actual = service.task(task["id"])
    assert actual["status"] == "COMPLETED", actual
    result = service.results(task["id"])
    assert result["completed_cases"] == result["planned_cases"]
    assert (
        result["score"]["coverage_ops_core"] == expected_score
        if expected_score is not None
        else result["score"] is None
    )
    assert phases == (["run", "score", "report"] if expected_score is not None else ["run", "report"])
    assert result["certification"]["execution"] == "NOT_COMPUTED"
    assert not result["performance_available"]
    if result["score"]:
        assert "by_category_json" not in result["score"]
        assert isinstance(result["score"]["by_category"], dict)
    artifacts = service.artifacts(task["id"])
    assert {"results.jsonl", "report.md", "report.html"} <= {artifact["name"] for artifact in artifacts}
    for artifact in artifacts:
        assert hashlib.sha256(service.artifact(task["id"], artifact["name"])).hexdigest() == artifact["sha256"]


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("scope", "RESULT-SCOPE"),
        ("identity", "RESULT-IDENTITY"),
        ("environment", "EVALUATOR-FAILURE"),
        ("load", "ENGINE-VALIDATION"),
        ("score", "ENGINE-SCORE"),
    ],
)
def test_engine_failures_do_not_turn_into_successful_scores(service, monkeypatch, fault, expected):
    task = create_task(service)

    def invoke(current, phase, command):
        if phase == "run":
            validation = [dict(rule_id="SUB-LOD-005", stage="load", operator="Exp", detail="fixture")]
            engine_fixture(
                service,
                current,
                Path(command[4]).parent,
                omit_case=fault == "scope",
                wrong_identity=fault == "identity",
                status="ENV_ABORT" if fault == "environment" else "PASS",
                validation=validation if fault == "load" else None,
            )
            exit_code = 1 if fault == "load" else 0
            return [sys.executable, "-c", f"raise SystemExit({exit_code})"]
        return [sys.executable, "-c", "raise SystemExit(1)"]

    intercept_engine_processes(monkeypatch, service, invoke)
    assert service.work_once()
    actual = service.task(task["id"])
    assert actual["status"] == ("REJECTED" if fault == "load" else "FAILED")
    assert actual["error"] == expected
    assert service.results(task["id"])["score"] is None


def test_worker_verifies_original_upload_before_start(service, monkeypatch):
    task = create_task(service)
    path = service.root / "submissions" / task["submission_id"] / "submission.zip"
    path.write_bytes(b"changed after upload")

    def unexpected(*args):
        pytest.fail("被篡改提交不能启动执行器")

    intercept_engine_processes(monkeypatch, service, unexpected)
    assert service.work_once()
    assert service.task(task["id"])["error"] == "PACKAGE-CHANGED"


def test_running_cancellation_terminates_real_process_group_and_pauses_worker(service, monkeypatch):
    task = create_task(service)
    marker = service.root / "pid.txt"
    child = (
        f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )

    def invoke(current, phase, command):
        assert phase == "run"
        return [sys.executable, "-c", child]

    intercept_engine_processes(monkeypatch, service, invoke)
    thread = threading.Thread(target=service.work_once)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        process_id = int(marker.read_text())
        service.cancel(task["id"])
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert service.task(task["id"])["status"] == "CANCELLED"
        assert service.task(task["id"])["worker_paused"]
        with pytest.raises(ProcessLookupError):
            os.kill(process_id, 0)
        assert not service.work_once()
    finally:
        service.stopping.set()
        thread.join(timeout=10)


def test_command_cannot_use_directory_or_latest_run(service, monkeypatch):
    task = create_task(service)
    phases = []

    def invoke(current, phase, command):
        phases.append(phase)
        if phase == "run":
            package = Path(command[command.index("--submission") + 1])
            assert package.is_file() and package.name == "submission.zip"
            assert hashlib.sha256(package.read_bytes()).hexdigest() == current["package_sha256"]
            assert command[command.index("--operators") + 1] == "Exp"
            assert "--limit" not in command
            engine_fixture(service, current, Path(command[4]).parent)
            return [sys.executable, "-c", "raise SystemExit(0)"]
        assert command[command.index("--run") + 1] == "1"
        return None

    intercept_engine_processes(monkeypatch, service, invoke)
    assert service.work_once()
    assert phases == ["run", "score", "report"]
    assert service.task(task["id"])["status"] == "COMPLETED"
