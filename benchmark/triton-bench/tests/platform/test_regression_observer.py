# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Observer regressions use fake API replies and never acquire an NPU lease."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "regression_observer", ROOT / "scripts/platform_examples/run_independent_regression.py"
)
observer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer
SPEC.loader.exec_module(observer)


def observation(tmp_path):
    args = SimpleNamespace(url="http://127.0.0.1:1", resume_task="known", revision="test")
    return observer.Observation(
        args, tmp_path, {"manifest_hash": "manifest"}, b"source", "package", SimpleNamespace(state=lambda: "stopped"),
        {"success": False, "issues": []}, 0.0,
    )


def test_progress_logger_does_not_recurse(monkeypatch):
    messages = []
    monkeypatch.setattr(observer, "emit", lambda value, **kwargs: messages.append(json.loads(value)))
    observer.log_event(kind="progress", completed=3)
    assert messages == [{"kind": "progress", "completed": 3}]


def test_resume_rejects_unrelated_task_without_arming_cleanup(tmp_path):
    audit = observation(tmp_path)
    (tmp_path / "task.json").write_text(json.dumps({"id": "known"}))

    requests = []

    def response(path):
        requests.append(path)
        if path == "/api/v1/catalog":
            return dict(manifest_hash="manifest", worker_enabled=True,
                        execution_backend="remote", worker_paused=False)
        return {"id": "other", "package_sha256": "package"}

    audit.request = response
    with pytest.raises(ValueError, match="identity"):
        audit.prepare_task()
    assert audit.task is None
    audit.cleanup()
    assert requests == ["/api/v1/catalog", "/api/v1/evaluations/known"]
    assert "cleanup_error" not in audit.result


def test_sample_audit_reports_real_measurement_inconsistencies(tmp_path):
    audit = observation(tmp_path)
    row = dict(case_code="frozen-case", status="PASS", samples_ms=[1.0, 2.0, 3.0],
               perf_median_ms=7.0, baseline_median_ms=4.0, speedup=2.0, perf_cv=0.0)
    audit.task = {"id": "known", "status": "FAILED"}
    audit.expected.update(case_codes=["frozen-case"], expected_status_counts={"PASS": 1})
    audit.task["remote"] = dict(release_state="SHUTDOWN", artifacts_verified=True)

    def response(path):
        if path.endswith("/results"):
            return {"cases": [dict(row, operator="Abs")]}
        return {"items": [], "next_cursor": 0}

    audit.request = response
    audit.collect()
    issues = audit.result["issues"]
    assert len(issues) == 3
    assert any("median" in item for item in issues)
    assert any("CV" in item for item in issues)
    assert any("speedup" in item for item in issues)


def test_artifact_path_is_rejected_before_network_or_write(tmp_path):
    audit = observation(tmp_path)

    requests = []
    audit.task = {"id": "known", "status": "FAILED"}

    def response(path):
        requests.append(path)
        if path.endswith("/events?after=0"):
            return {"items": [], "next_cursor": 0}
        if path.endswith("/results"):
            return {"cases": []}
        return {"items": [{"name": "../escaped"}]}

    audit.request = response
    with pytest.raises(ValueError, match="Unsafe artifact"):
        audit.collect()
    assert requests == ["/api/v1/evaluations/known/" + suffix
                        for suffix in ("results", "events?after=0", "artifacts")]
    assert not (tmp_path.parent / "escaped").exists()
