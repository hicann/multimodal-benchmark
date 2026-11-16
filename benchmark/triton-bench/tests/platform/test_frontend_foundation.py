# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Frontend contracts and real fixed-target loopback gateway; no device execution."""

from contextlib import ExitStack, contextmanager
from http.client import HTTPConnection
import json
import threading

import pytest

from triton_eval.frontend_gateway import make_gateway
from triton_eval.orchestrator import RemoteConfig
from triton_eval import package
from triton_eval.submission_http import MAX_REQUEST_BYTES, make_server, openapi
from triton_eval.submission_policy import submission_policy
from triton_eval.submission_service import EvaluationService, ServiceError
from triton_eval.submission_simulation import DEFAULT_CORPUS
from test_submission_http import call, upload_bytes


@pytest.fixture
def service(tmp_path):
    value = EvaluationService(tmp_path / "service", DEFAULT_CORPUS)
    try:
        yield value
    finally:
        value.close()


def test_catalog_publishes_actual_validator_limits(service, monkeypatch):
    policy = service.catalog()["submission_policy"]
    assert policy["max_request_bytes"] == MAX_REQUEST_BYTES
    assert policy["max_file_bytes"] == package.MAX_FILE_BYTES
    assert set(policy["allowed_suffixes"]) == package.ALLOWED_SUFFIXES
    assert {item["rule_id"] for item in policy["rules"]} == set(package.RULES)
    monkeypatch.setattr(package, "MAX_FILE_BYTES", 123)
    assert submission_policy()["max_file_bytes"] == 123
    profiles = {item["id"]: item for item in service.catalog()["profile_capabilities"]}
    assert not profiles["smoke"]["performance_scoring"]
    assert profiles["standard"]["performance_scoring"] and profiles["standard"]["experimental"]
    assert not any(
        item["performance_accepted"] or item["full_suite_accepted"] for item in profiles.values()
    )


def make_task(service):
    submission = service.upload(upload_bytes())
    return service.create_task(dict(submission_id=submission["id"]))


def test_cancel_capability_tracks_pending_and_terminal(service):
    task = make_task(service)
    assert task["available_actions"]["cancel"]["enabled"]
    assert task["available_actions"]["recover"]["reason"] == "REMOTE_JOB_REQUIRED"
    task.update(status="RUNNING", cancel_requested=True)
    service.save_object("task", task)
    assert (
        service.task(task["id"])["available_actions"]["cancel"]["reason"] == "CANCELLATION_PENDING"
    )
    service.transition_task(task, "CANCELLED")
    assert service.task(task["id"])["available_actions"]["cancel"]["reason"] == "TASK_TERMINAL"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"status": "QUEUED"}, "TASK_NOT_RECOVERABLE"),
        ({"release_state": "UNCONFIRMED"}, "RELEASE_NOT_CONFIRMED"),
        ({"remote_workdir": None}, "REMOTE_RESULT_NOT_LOCATED"),
        ({"instance": "b" * 32}, "INSTANCE_CHANGED"),
        ({"installed": True}, "RESULTS_ALREADY_INSTALLED"),
        ({}, None),
    ],
)
def test_recovery_capability_matches_mutation_eligibility(tmp_path, change, reason):
    service = EvaluationService(
        tmp_path / "remote",
        DEFAULT_CORPUS,
        remote_config=RemoteConfig(instance="a" * 32, ssh_host="fixture.npu"),
    )
    try:
        task = make_task(service)
        service.remote_jobs.update(
            task["id"], status="FAILED", release_state="SHUTDOWN", remote_workdir="/fixture/job"
        )
        if change.get("installed"):
            (service.root / "evaluations" / task["id"]).mkdir(parents=True)
        else:
            service.remote_jobs.update(task["id"], **change)
        action = service.task(task["id"])["available_actions"]["recover"]
        assert action["reason"] == reason and action["enabled"] == (reason is None)
        assert action["starts_instance"] and action["requires_acknowledgement"]
        if reason is None:
            recovered = service.recover_task(task["id"], {"acknowledge_recovery": True})
            assert recovered["status"] == "QUEUED"
            assert not recovered["available_actions"]["recover"]["enabled"]
        else:
            with pytest.raises(ServiceError) as exc:
                service.recover_task(task["id"], {"acknowledge_recovery": True})
            assert exc.value.code == "RECOVERY-NOT-APPLICABLE"
    finally:
        service.close()


def test_generated_contract_exposes_typed_events_and_capabilities():
    schemas = openapi()["components"]["schemas"]
    assert schemas["Events"]["properties"]["items"]["items"]["$ref"].endswith("/PublicEvent")
    assert len(schemas["RemoteEnvelope"]["oneOf"]) == 5
    assert schemas["EngineCaseEvent"]["properties"]["record"]["$ref"].endswith("/CaseResult")
    assert "available_actions" in schemas["Task"]["required"]
    assert "submission_policy" in schemas["Catalog"]["required"]


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def gateway(service, tmp_path):
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Static test fixture</title>")
    (static / "asset.js").write_text("/* test fixture */")
    (static / ".env").write_text("not-publishable")
    outside = tmp_path / "outside.txt"
    outside.write_text("not-publishable")
    (static / "link.txt").symlink_to(outside)
    with ExitStack() as stack:
        try:
            backend = stack.enter_context(serving(make_server(service, 0)))
            frontend = stack.enter_context(serving(make_gateway(static, backend.server_port, 0)))
        except PermissionError:
            pytest.skip("沙箱禁止 loopback；不是网关测试通过")
        yield f"http://127.0.0.1:{frontend.server_port}", backend, service


def test_gateway_validates_frontend_origin_before_rewriting(gateway):
    base, backend, _ = gateway
    assert call(base, "/api/v1/catalog", headers={"Origin": base})[0] == 200
    for headers in (
        {"Origin": "https://other.invalid"},
        {"Origin": f"http://127.0.0.1:{backend.server_port}"},
        {"Host": "other.invalid"},
        {"Sec-Fetch-Site": "cross-site"},
    ):
        assert call(base, "/api/v1/catalog", headers=headers)[0] == 403
    assert call(base, "/openapi.json")[1] == openapi()
    assert call(base, "/api/v1/not-found")[0] == 404


def test_gateway_upload_idempotency_cancel_stream_and_download(gateway):
    base, backend, service = gateway
    headers = {"Origin": base, "Idempotency-Key": "gateway-upload"}
    code, submission = call(base, "/api/v1/submissions", upload_bytes(), "application/zip", headers)
    assert code == 201 and submission["status"] == "READY"
    assert (
        call(base, "/api/v1/submissions", upload_bytes(), "application/zip", headers)[1]["id"]
        == submission["id"]
    )
    code, rejected = call(base, "/api/v1/submissions", b"not-a-zip", "application/zip")
    assert code == 201 and rejected["status"] == "REJECTED"
    code, task = call(base, "/api/v1/evaluations", {"submission_id": submission["id"]})
    assert code == 202
    prefix = "/api/v1/evaluations/" + task["id"]
    assert call(base, prefix + "/results")[1]["score"] is None
    port = int(base.rsplit(":", 1)[1])
    client = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.request("GET", prefix + "/stream")
        response = client.getresponse()
        assert response.status == 200
        # Must arrive while the task is still QUEUED, not after stream EOF.
        assert response.readline() == b"retry: 2000\n"
        assert service.task(task["id"])["status"] == "QUEUED"
        assert call(base, prefix + "/cancel", {})[1]["status"] == "CANCELLED"
        content = response.read()
        assert b"event: update\n" in content and b"event: end\n" in content
    finally:
        client.close()
    events = call(base, prefix + "/events")[1]
    assert len(events["items"]) == 2
    client = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.request(
            "GET", prefix + "/stream?after=0", headers={"Last-Event-ID": str(events["next_cursor"])}
        )
        assert b"event: update" not in client.getresponse().read()
    finally:
        client.close()
    _check_gateway_download(base, service, task)


def _check_gateway_download(base, service, task):
    prefix = "/api/v1/evaluations/" + task["id"]
    port = int(base.rsplit(":", 1)[1])
    reports = service.task_directory(task["id"]) / "reports"
    reports.mkdir(parents=True)
    data = b"<html>download-only report fixture</html>"
    (reports / "report-run-1.html").write_bytes(data)
    artifact = call(base, prefix + "/artifacts")[1]["items"][0]
    assert artifact["bytes"] == len(data)
    client = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.request("GET", prefix + "/artifacts/report.html")
        response = client.getresponse()
        assert response.getheader("Content-Disposition") == 'attachment; filename="report.html"'
        assert response.read() == data
    finally:
        client.close()


@pytest.mark.parametrize(
    "path,status",
    [
        ("/", 200),
        ("/submit", 200),
        ("/workspace/jobs/" + "a" * 32, 200),
        ("/asset.js", 200),
        ("/.env", 400),
        ("/%2e%2e/outside.txt", 400),
        ("/link.txt", 404),
        ("/api/no-such-route", 404),
        ("/missing.js", 404),
    ],
)
def test_gateway_static_routes_and_file_boundaries(gateway, path, status):
    base, _, _ = gateway
    client = HTTPConnection("127.0.0.1", int(base.rsplit(":", 1)[1]), timeout=3)
    try:
        client.request("GET", path)
        response = client.getresponse()
        assert response.status == status
        assert b"not-publishable" not in response.read()
    finally:
        client.close()


def test_gateway_rejects_oversize_and_duplicate_framing(gateway):
    base, _, _ = gateway
    port = int(base.rsplit(":", 1)[1])
    for lengths, expected in [([MAX_REQUEST_BYTES + 1], 413), ([0, 0], 400)]:
        client = HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            client.putrequest("POST", "/api/v1/submissions")
            for length in lengths:
                client.putheader("Content-Length", str(length))
            client.endheaders()
            response = client.getresponse()
            assert response.status == expected
            response.read()
        finally:
            client.close()


def test_gateway_reports_unreachable_backend_as_json(gateway, tmp_path):
    base, backend, service = gateway
    unused = make_server(service, 0)
    unused_port = unused.server_port
    unused.server_close()
    static = tmp_path / "offline-dist"
    static.mkdir()
    (static / "index.html").write_text("<title>Static test fixture</title>")
    with serving(make_gateway(static, unused_port, 0)) as entry:
        code, response = call(f"http://127.0.0.1:{entry.server_port}", "/api/v1/catalog")
        assert code == 502 and response["error"]["code"] == "GATEWAY-UPSTREAM"
