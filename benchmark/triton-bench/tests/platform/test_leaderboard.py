# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Real SQLite and scorer fixtures; these device records are explicitly synthetic."""

from io import BytesIO
import json
import math
import zipfile
import pytest
from triton_eval.leaderboard import view
from triton_eval.models import CaseResultRecord, CaseStatus
from triton_eval.scorer import compute_run_score
from triton_eval.store import Store
from triton_eval.submission_service import EvaluationService, ServiceError
from triton_eval.submission_simulation import DEFAULT_CORPUS
from test_submission_http import raw_http


@pytest.fixture
def service(tmp_path):
    value = EvaluationService(tmp_path / "board", DEFAULT_CORPUS)
    yield value
    value.close()


def complete(
    service,
    label,
    ratio=2.0,
    *,
    env="fixture-env",
    state="COMPLETED",
    profile="standard",
    status="PASS",
    missing=False,
    omit=False,
    package=None,
):
    package = _package(label) if package is None else package
    submission = service.upload(package)
    task = service.create_task(dict(submission_id=submission["id"], profile=profile))
    directory = service.task_directory(task["id"])
    directory.mkdir(parents=True)
    with Store(directory / "engine.sqlite") as store:
        store.create_schema()
        corpus = store.import_corpus(service.manifest)
        compiler = store.upsert_submission("cpu-fixture-not-device", "test", None, "source")
        run = store.create_run(
            corpus,
            compiler,
            profile,
            dict(
                compiler_name="fixture",
                triton_import_version="test",
                npu_model="fixture",
                clock_locked=False,
                driver_version=None,
            ),
            env,
            "PASS",
        )
        records = []
        cases = [c for c in service.manifest.cases if c.code in task["case_codes"]]
        for index, case in enumerate(cases[:-1] if omit else cases):
            timed = profile == "standard" and not (missing and index == 0)
            record = CaseResultRecord(
                run_id=run,
                case_code=case.code,
                operator=case.operator,
                status=CaseStatus(status),
                samples_ms=[1.0, 1.0, 1.0] if timed else [],
                perf_median_ms=1.0 if timed else None,
                baseline_median_ms=ratio if timed else None,
                speedup=ratio if timed else None,
                perf_cv=0.0 if timed else None,
            )
            records.append(record)
            store.insert_case_result(record)
        store.insert_run_score(
            compute_run_score(run, records, {"Exp": 1}, [1.0, 1.2, 2.0], 0.05, "fixture")
        )
        store.finish_run(run, "OK")
    service.transition_task(service.object_record("task", task["id"]), state)
    return task, package


def test_shared_anchor_changes_score_when_task_self_score_is_absent(service):
    slow, _ = complete(service, "first", 2)
    first = view(service)
    assert first["items"][0]["shared_perf_score"] == 100
    fast, _ = complete(service, "second", math.sqrt(8))
    board = view(service)
    assert [e["task_id"] for e in board["items"]] == [fast["id"], slow["id"]]
    assert board["items"][0]["shared_perf_score"] == pytest.approx(150)
    assert board["anchors"][0]["task_id"] == slow["id"]
    assert service.results(fast["id"])["score"]["perf_score"] is None
    revision = board["baseline_revision"]
    service.close()
    restored = EvaluationService(service.root, DEFAULT_CORPUS)
    try:
        again = view(restored)
        assert again["baseline_revision"] == revision
        assert again["items"][0]["shared_perf_score"] == pytest.approx(150)
    finally:
        restored.close()


def test_exact_environment_and_profile_are_separate_groups(service):
    first, _ = complete(service, "one", 2, env="one")
    second, _ = complete(service, "two", 4, env="two")
    smoke, _ = complete(service, "smoke", profile="smoke")
    board = view(service, task_id=first["id"])
    assert len(board["cohorts"]) == 3
    assert [e["task_id"] for e in board["items"]] == [first["id"]]
    assert view(service, task_id=second["id"])["items"][0]["shared_perf_score"] == 100
    assert view(service, task_id=smoke["id"])["items"] == []
    assert (
        view(service, metric="functional", task_id=smoke["id"])["items"][0]["functional_score"]
        == 100
    )


def test_incomplete_or_failed_measurement_never_creates_anchor(service):
    failed, _ = complete(service, "failed", state="FAILED")
    missing, _ = complete(service, "missing", missing=True)
    wrong, _ = complete(service, "wrong", status="WRONG_RESULT")
    short, _ = complete(service, "short", omit=True)
    board = view(service)
    assert board["anchors"] == [] and board["items"] == []
    reasons = {e["task_id"]: e["reason"] for e in board["excluded"]}
    assert all(t["id"] in reasons for t in [failed, missing, wrong, short])
    functional = view(service, metric="functional")
    assert {e["task_id"] for e in functional["items"]} == {missing["id"], wrong["id"]}


def test_duplicate_package_best_only_and_ties_use_competition_ranks(service):
    one, package = complete(service, "one", 2)
    duplicate, _ = complete(service, "again", math.sqrt(8), package=package)
    tie, _ = complete(service, "tie", math.sqrt(8))
    slower, _ = complete(service, "slower", 2)
    board = view(service)
    assert len(board["items"]) == 3
    assert {e["task_id"] for e in board["items"]} == {duplicate["id"], tie["id"], slower["id"]}
    assert [e["rank"] for e in board["items"]] == [1, 1, 3]
    assert board["anchors"][0]["task_id"] == one["id"]


def test_near_unit_anchor_is_explicitly_unscored_and_not_refrozen(service):
    first, _ = complete(service, "unit", 1)
    assert view(service)["items"] == []
    complete(service, "later", 2)
    board = view(service)
    assert board["items"] == []
    assert board["anchors"][0]["task_id"] == first["id"]


def test_query_contract_and_loopback_security(service):
    assert raw_http(service, "GET", "/api/v1/leaderboard")[0] == 200
    for query in [
        "metric=x",
        "metric=",
        "metric=functional&metric=performance",
        "cohort=x",
        "task_id=x",
        "extra=1",
        "cohort=" + "a" * 64 + "&task_id=" + "b" * 32,
    ]:
        assert raw_http(service, "GET", "/api/v1/leaderboard?" + query)[0] == 400
    assert raw_http(service, "GET", "/api/v1/leaderboard?cohort=" + "a" * 64)[0] == 404
    assert raw_http(service, "GET", "/api/v1/leaderboard?task_id=" + "b" * 32)[0] == 409
    assert (
        raw_http(
            service,
            "GET",
            "/api/v1/leaderboard",
            extra_headers={"Origin": "https://invalid.example"},
        )[0]
        == 403
    )
    assert (
        raw_http(service, "GET", "/api/v1/leaderboard", extra_headers={"Host": "invalid.example"})[
            0
        ]
        == 403
    )


def _package(label):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "manifest.yaml",
            f"submit_id: {label}\noperators:\n- name: Exp\n  entry: exp\n  path: kernel.py\n",
        )
        archive.writestr(
            "kernel.py", "def exp(x, base=-1.0, scale=1.0, shift=0.0):\n    return x\n"
        )
    return stream.getvalue()
