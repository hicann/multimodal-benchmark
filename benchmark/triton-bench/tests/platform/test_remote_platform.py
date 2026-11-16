# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Control-plane integration; device records here are explicitly CPU fixtures."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from triton_eval.models import CaseResultRecord, CaseStatus
from triton_eval.orchestrator import Orchestrator, RemoteConfig
from triton_eval.remote_bundle import build_bundle, sha256
from triton_eval.remote_jobs import canonical
from triton_eval.remote_runner import Journal, run as runner_run
from triton_eval.store import Store
from triton_eval.submission_service import EvaluationService, ServiceError
from triton_eval.submission_simulation import DEFAULT_CORPUS
from test_submission_http import raw_http, upload_bytes


def config(**changes):
    return RemoteConfig(instance="a" * 32, ssh_host="test.npu", poll_interval=0.001, **changes)


@pytest.fixture
def remote(tmp_path):
    service = EvaluationService(tmp_path / "platform", DEFAULT_CORPUS, remote_config=config())
    yield service
    service.close()


def queued(service):
    submission = service.upload(upload_bytes())
    return service.create_task(dict(submission_id=submission["id"]))


def envelope(job, sequence=1, kind="heartbeat", payload=None):
    return dict(
        protocol_version=1, job_id=job["id"], event_seq=sequence, kind=kind, payload=payload or {}
    )


def test_schema_migration_preserves_objects_and_queue_cancel(remote):
    task = queued(remote)
    job = remote.remote_jobs.get(task["id"])
    assert job["status"] == "QUEUED"
    with remote.database_connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'remote_job%'"
        ).fetchall()
    assert remote.cancel(task["id"])["status"] == "CANCELLED"
    assert remote.remote_jobs.get(task["id"])["status"] == "CANCELLED"
    assert remote.remote_jobs.claim() is None


def test_atomic_claim_fifo_and_single_active_job(remote):
    first, second = queued(remote), queued(remote)
    job = remote.remote_jobs.claim()
    assert job["task_id"] == first["id"]
    assert remote.task(first["id"])["status"] == "RUNNING"
    assert remote.remote_jobs.claim() is None
    remote.remote_jobs.update(first["id"], status="FAILED")
    assert remote.remote_jobs.claim()["task_id"] == second["id"]


def test_event_replay_partial_utf8_and_same_sequence_conflict(remote):
    task = queued(remote)
    job = remote.remote_jobs.claim()
    event = envelope(job, payload={"note": "启动"})
    data = (canonical(event) + "\n").encode()
    assert remote.remote_jobs.consume(task["id"], data[:-2]) == 0
    assert remote.remote_jobs.consume(task["id"], data) == len(data)
    assert remote.remote_jobs.accept(task["id"], event, len(data)) is False
    with pytest.raises(ServiceError, match="不同内容"):
        remote.remote_jobs.accept(task["id"], {**event, "payload": {"note": "tampered"}}, len(data))
    with pytest.raises(ServiceError, match="不连续"):
        remote.remote_jobs.accept(task["id"], envelope(job, 3), len(data) + 100)
    with pytest.raises(ServiceError, match="身份"):
        remote.remote_jobs.accept(
            task["id"], {**envelope(job, 2), "job_id": "b" * 32}, len(data) + 100
        )
    with pytest.raises(ServiceError, match="协议"):
        remote.remote_jobs.accept(task["id"], [], len(data) + 100)
    assert remote.remote_jobs.get(task["id"])["event_seq"] == 1
    assert remote.remote_jobs.get(task["id"])["offset"] == len(data)


def test_event_invalid_case_and_phase_are_atomic(remote):
    task = queued(remote)
    job = remote.remote_jobs.claim()
    for event in [
        envelope(job, kind="engine", payload={"event": "case", "record": {}}),
        envelope(job, kind="phase", payload={"phase": "completed"}),
    ]:
        with pytest.raises(ServiceError):
            remote.remote_jobs.accept(task["id"], event, 100)
    assert remote.remote_jobs.get(task["id"])["event_seq"] == 0
    assert len(remote.events(task["id"])["items"]) == 2
    remote.remote_jobs.accept(
        task["id"], envelope(job, kind="phase", payload={"phase": "reporting"}), 100
    )
    with pytest.raises(ServiceError, match="倒退"):
        remote.remote_jobs.accept(task["id"], envelope(job, 2, "phase", {"phase": "running"}), 200)


def test_sse_drains_batches_and_uses_global_cursor(remote):
    task, other = queued(remote), queued(remote)
    job = remote.remote_jobs.claim()
    for sequence in range(1, 206):
        remote.remote_jobs.accept(task["id"], envelope(job, sequence), sequence * 200)
    remote.remote_jobs.update(task["id"], status="FAILED")
    remote.cancel(other["id"])
    prefix = "/api/v1/evaluations/" + task["id"]
    status, body = raw_http(remote, "GET", prefix + "/stream")
    assert status == 200 and body.startswith(b"retry: 2000\n\n")
    assert body.count(b"event: update\n") == len(remote.events(task["id"])["items"]) + 8
    assert body.endswith(b"event: end\ndata: {}\n\n")
    cursor = remote.events(task["id"])["next_cursor"]
    status, body = raw_http(
        remote, "GET", prefix + "/stream", extra_headers={"Last-Event-ID": str(cursor)}
    )
    assert status == 200 and body.count(b"event: update\n") == 8
    assert (
        raw_http(
            remote,
            "GET",
            prefix + "/stream",
            extra_headers={"Last-Event-ID": "99999999999999999999999"},
        )[0]
        == 400
    )
    assert (
        raw_http(
            remote, "GET", prefix + "/stream", extra_headers={"Origin": "https://invalid.example"}
        )[0]
        == 403
    )
    slots = [remote.stream_slots.acquire(False) for _ in range(8)]
    try:
        assert all(slots)
        assert raw_http(remote, "GET", prefix + "/stream")[0] == 429
    finally:
        for _ in slots:
            remote.stream_slots.release()


def test_task_list_pagination_and_backend_contract(remote):
    tasks = [queued(remote) for _ in range(3)]
    page = remote.list_tasks(limit=2)
    assert [task["id"] for task in page["items"]] == [task["id"] for task in tasks[::-1][:2]]
    assert remote.list_tasks(page["next_cursor"], 2)["items"][0]["id"] == tasks[0]["id"]
    assert remote.catalog()["execution_backend"] == "remote"
    assert raw_http(remote, "GET", "/api/v1/evaluations?limit=0")[0] == 400
    assert raw_http(remote, "GET", "/api/v1/evaluations?limit=2")[0] == 200


class FakeCloud:
    def __init__(self, order, failure=None):
        self.value = "stopped"
        self.order, self.failure = order, failure

    def state(self):
        return self.value

    def start(self):
        self.order.append("start")
        self.value = "running"
        if self.failure == "start":
            raise ServiceError("UNCERTAIN-START", "lost response")

    def stop(self):
        self.order.append("stop")
        if self.failure == "stop":
            raise ServiceError("STOP-FAIL", "fixture")
        self.value = "stopped"


class FakeTransport:
    def __init__(self, directory, order, failure=None):
        self.directory, self.order, self.failure = directory, order, failure

    def check(self, phase):
        self.order.append(phase)
        if self.failure == phase:
            raise ServiceError("FIXTURE-" + phase, "fixture failure")

    def probe(self, **kwargs):
        self.check("probe")

    def send(self, archive, remote, digest, **kwargs):
        self.check("send")
        assert sha256(archive) == digest
        self.directory.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(self.directory, filter="data")

    def launch(self, remote, **kwargs):
        self.check("launch")
        assert runner_run(self.directory, health_probe=lambda: {"ok": True, "fixture": True}) == 0

    def read(self, remote, offset, **kwargs):
        self.check("read")
        return (self.directory / "remote-events.jsonl").read_bytes()[offset: offset + 2048]

    def download(self, remote, output, expected_bytes, **kwargs):
        self.check("download")
        shutil.copyfile(self.directory / "result.tar.gz", output)
        if self.failure == "corrupt":
            with output.open("ab") as stream:
                stream.write(b"corruption")


@pytest.fixture
def device_fixture(monkeypatch):
    original = EvaluationService.invoke_phase

    def invoke(service, task, phase):
        if phase != "run":
            return original(service, task, phase)
        directory = service.task_directory(task["id"])
        run_directory = directory / "runs/run-1"
        run_directory.mkdir(parents=True)
        with Store(directory / "engine.sqlite") as store:
            store.create_schema()
            corpus_id = store.import_corpus(service.manifest)
            compiler_id = store.upsert_submission("CPU-FIXTURE-NOT-NPU", "fixture", None, "source")
            run_id = store.create_run(
                corpus_id,
                compiler_id,
                task["profile"],
                {},
                "fixture",
                "OK",
                run_dir=str(run_directory),
            )
            record_id = store.insert_submission_record(
                submit_id="fixture",
                root_path="fixture",
                operators=task["operators"],
                snapshot_fp="fixture",
                validation_ok=True,
                validation=[],
                package_sha256=task["package_sha256"],
            )
            store.set_run_submission_record(run_id, record_id)
            results = []
            for case in service.manifest.cases:
                if case.code not in task["case_codes"]:
                    continue
                record = CaseResultRecord(
                    run_id=run_id,
                    case_code=case.code,
                    operator=case.operator,
                    status=CaseStatus.PASS,
                )
                store.insert_case_result(record)
                results.append(record.model_dump_json())
            (run_directory / "results.jsonl").write_text("\n".join(results) + "\n")
            (run_directory / "events.jsonl").write_text(
                "".join(
                    canonical({"event": "case", "record": json.loads(row)}) + "\n"
                    for row in results
                )
            )
            store.finish_run(run_id, "OK")
        return 0

    monkeypatch.setattr(EvaluationService, "_invoke", invoke)


def setup_execution(service, tmp_path, failure=None):
    order = []
    cloud = FakeCloud(order, failure)
    transport = FakeTransport(tmp_path / "npu-fixture", order, failure)
    service.orchestrator = Orchestrator(
        service, config(shutdown_timeout=1), cloud=cloud, transport=transport
    )
    task = queued(service)
    return task, order, cloud, transport


def test_remote_round_trip_real_sqlite_score_report_and_shutdown(remote, tmp_path, device_fixture):
    task, order, cloud, transport = setup_execution(remote, tmp_path)
    assert remote.work_once()
    actual = remote.task(task["id"])
    assert actual["status"] == "COMPLETED", actual
    assert actual["completed_cases"] == 20
    assert actual["remote"]["release_state"] == "SHUTDOWN"
    assert actual["remote"]["artifacts_verified"] is True
    assert order.index("download") < order.index("stop")
    results = remote.results(task["id"])
    assert results["score"]["coverage_ops_core"] == 100
    assert results["score"]["perf_score"] is None
    assert not results["certification"]["full_suite_accepted"]
    assert {"report.md", "report.html", "results.jsonl"} <= {row['name'] for row in remote.artifacts(task['id'])}
    assert cloud.value == "stopped"
    with remote.database_connection() as db:
        source_rows = db.execute(
            'SELECT payload FROM remote_job_event WHERE job_id=?', (actual['remote']['id'],),
        )
        payloads = [json.loads(row[0]) for row in source_rows]
    assert [event["payload"]["phase"] for event in payloads if event["kind"] == "phase"] == [
        "running",
        "scoring",
        "reporting",
    ]
    # The same runner never reexecutes a completed task.
    assert runner_run(transport.directory, health_probe=lambda: pytest.fail("must not rerun")) == 0


@pytest.mark.parametrize(
    "failure", ["start", "probe", "send", "launch", "read", "download", "corrupt"]
)

def test_failures_release_lease_and_never_publish_score(remote, tmp_path, device_fixture, failure):
    task, order, cloud, _ = setup_execution(remote, tmp_path, failure)
    assert remote.work_once()
    actual = remote.task(task["id"])
    assert actual["status"] == "FAILED", actual
    assert order[-1] == "stop" and cloud.value == "stopped"
    assert remote.results(task["id"])["score"] is None


def test_busy_instance_is_not_stopped(remote, tmp_path):
    task, order, cloud, _ = setup_execution(remote, tmp_path)
    cloud.value = "running"
    assert remote.work_once()
    assert remote.task(task["id"])["error"] == "REMOTE-INSTANCE-BUSY"
    assert "stop" not in order and "start" not in order
    assert remote.task(task["id"])["worker_paused"]


def test_stop_failure_blocks_next_job_until_reconciled(remote, tmp_path):
    task, order, cloud, _ = setup_execution(remote, tmp_path, "probe")
    cloud.failure = "stop"
    assert remote.work_once()
    assert remote.task(task["id"])["error"] == "REMOTE-STOP-UNCONFIRMED"
    queued(remote)
    cloud.failure = None
    assert remote.work_once()  # reconciles release, does not run the next queued task
    assert cloud.value == "stopped"
    assert remote.task(task["id"])["status"] == "INTERRUPTED"
    assert order.count("start") == 1
    assert not remote.work_once()  # manual health gate remains


def test_restart_releases_active_lease_without_reexecution(remote, tmp_path):
    task = queued(remote)
    job = remote.remote_jobs.claim()
    remote.remote_jobs.update(task["id"], start_intent=True, release_state="UNCONFIRMED")
    root = remote.root
    remote.close()
    restored = EvaluationService(root, DEFAULT_CORPUS, remote_config=config())
    order = []
    cloud = FakeCloud(order)
    cloud.value = "running"
    restored.orchestrator = Orchestrator(restored, config(), cloud=cloud)
    try:
        assert restored.work_once()
        assert restored.task(task["id"])["status"] == "INTERRUPTED"
        assert restored.task(task["id"])["remote"]["release_state"] == "SHUTDOWN"
        assert order == ["stop"]
    finally:
        restored.close()


def test_completion_requires_local_artifacts_and_release(remote):
    task = queued(remote)
    remote.remote_jobs.claim()
    with pytest.raises(ServiceError, match="实例尚未释放"):
        remote.remote_jobs.update(task["id"], status="COMPLETED")
    assert remote.task(task["id"])["status"] == "RUNNING"


def test_journal_sequences_survive_reopen(tmp_path):
    path = tmp_path / "events.jsonl"
    Journal(path, "a" * 32).emit("heartbeat", {})
    Journal(path, "a" * 32).emit("heartbeat", {})
    assert [json.loads(row)["event_seq"] for row in path.read_text().splitlines()] == [1, 2]


def test_bundle_exact_content_and_no_local_credentials(remote, tmp_path):
    task = queued(remote)
    job = remote.remote_jobs.get(task["id"])
    output = tmp_path / "bundle.tar.gz"
    index = build_bundle(remote, task, job, output)
    assert index["manifest_hash"] == task["manifest_hash"]
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert "submission.zip" in names and "scripts/run-job.sh" in names
        assert not any(
            ".git" in Path(name).parts or ".credentials.json" in name or ".env.gatecheck" in name
            for name in names
        )
        assert "corpus/v0.9-official-candidates/manifest.json" in names
    second = tmp_path / "second.tar.gz"
    assert build_bundle(remote, task, job, second)["bundle_sha256"] == index["bundle_sha256"]


def test_watchdog_releases_after_owner_death_and_budget(tmp_path):
    from triton_eval.remote_watchdog import atomic_json, monitor

    lease = tmp_path / "lease.json"
    order = []
    cloud = FakeCloud(order)
    cloud.value = "running"
    atomic_json(lease, {"armed": True, "released": False})
    assert monitor(cloud, lease, 123, 99999999999, 1, alive=lambda pid: False) == "SHUTDOWN"
    assert order == ["stop"]
    cloud.value = "running"
    assert monitor(cloud, lease, 123, 0, 1, alive=lambda pid: True) == "SHUTDOWN"
    atomic_json(lease, {"armed": True, "released": True})
    assert monitor(cloud, lease, 123, 0, 1) == "DISARMED"
    atomic_json(lease, {"armed": False, "released": False})
    assert monitor(cloud, lease, 123, 0, 1) == "NOT_ACQUIRED"


def test_running_cancel_releases_without_publishing(remote, tmp_path, device_fixture):
    task, order, cloud, transport = setup_execution(remote, tmp_path)
    original = transport.read

    def cancel_read(*args, **kwargs):
        remote.cancel(task["id"])
        return original(*args, **kwargs)

    transport.read = cancel_read
    assert remote.work_once()
    assert remote.task(task["id"])["status"] == "CANCELLED"
    assert cloud.value == "stopped"
    assert remote.results(task["id"])["score"] is None


def test_corrupt_bundle_runner_reports_failure_without_execution(remote, tmp_path):
    task, order, cloud, transport = setup_execution(remote, tmp_path)
    original = transport.launch

    def corrupt(remote_path, **kwargs):
        (transport.directory / "submission.zip").write_bytes(b"tampered")
        assert (
            runner_run(transport.directory, health_probe=lambda: pytest.fail("must verify first"))
            == 1
        )

    transport.launch = corrupt
    assert remote.work_once()
    assert remote.task(task["id"])["status"] == "FAILED"
    assert remote.task(task["id"])["error"] == "RUNNER-ValueError"
    assert cloud.value == "stopped"


def test_oversized_event_line_and_after_final_are_rejected(remote):
    from triton_eval.remote_jobs import MAX_EVENT_BYTES

    task = queued(remote)
    job = remote.remote_jobs.claim()
    with pytest.raises(ServiceError, match="过大"):
        remote.remote_jobs.consume(task["id"], b"x" * (MAX_EVENT_BYTES + 1))
    remote.remote_jobs.accept(task["id"], envelope(job, 1, "final", {"status": "FAILED"}), 100)
    with pytest.raises(ServiceError, match="final"):
        remote.remote_jobs.accept(task["id"], envelope(job, 2), 200)


def test_independent_watchdog_survives_real_sigkill(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    from dataclasses import asdict
    from triton_eval.remote_watchdog import atomic_json
    from triton_eval.corpus_assets import ROOT

    state, marker, child = _watchdog_owner_fixture(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", child], env=dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        process.kill()
        process.wait(timeout=3)
        result = tmp_path / "watchdog-result.json"
        deadline = time.monotonic() + 5
        while not result.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert json.loads(result.read_text())["release_state"] == "SHUTDOWN"
        assert state.read_text() == "stopped"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def test_live_remote_case_visible_before_archive_and_no_score(remote):
    task = queued(remote)
    job = remote.remote_jobs.claim()
    case = next(case for case in remote.manifest.cases if case.code in task["case_codes"])
    record = CaseResultRecord(
        run_id=1, case_code=case.code, operator=case.operator, status=CaseStatus.PASS
    )
    event = envelope(
        job, kind="engine", payload={"event": "case", "record": record.model_dump(mode="json")}
    )
    remote.remote_jobs.accept(task["id"], event, 1000)
    assert remote.task(task["id"])["completed_cases"] == 1
    result = remote.results(task["id"])
    assert result["cases"][0]["case_code"] == case.code
    assert result["score"] is None and result["score_reason"] == "TASK_NOT_COMPLETED"


def test_resume_requires_explicit_health_and_released_lease(remote):
    remote.pause_worker("fixture")
    for request in (
        {},
        {"acknowledge_health": False},
        {"acknowledge_health": 1},
        {"acknowledge_health": True, "extra": 1},
    ):
        with pytest.raises(ServiceError):
            remote.resume_worker(request)
    task = queued(remote)
    remote.remote_jobs.claim()
    with pytest.raises(ServiceError, match="租约"):
        remote.resume_worker({"acknowledge_health": True})
    remote.remote_jobs.update(task["id"], status="FAILED")
    code, body = raw_http(remote, "POST", "/api/v1/worker/resume", b'{"acknowledge_health":true}')
    assert code == 200 and not json.loads(body)["worker_paused"]


def test_enabling_remote_migrates_existing_queue_without_new_task(tmp_path):
    service = EvaluationService(tmp_path / "migration", DEFAULT_CORPUS)
    task = queued(service)
    service.close()
    restored = EvaluationService(tmp_path / "migration", DEFAULT_CORPUS, remote_config=config())
    try:
        assert restored.remote_jobs.get(task["id"])["status"] == "QUEUED"
        assert restored.remote_jobs.get(task["id"])["instance"] == "a" * 32
        assert restored.list_tasks()["items"][0]["id"] == task["id"]
    finally:
        restored.close()


def test_recovery_never_stops_different_configured_instance(remote, tmp_path):
    task = queued(remote)
    remote.remote_jobs.claim()
    remote.remote_jobs.update(task["id"], start_intent=True, release_state="UNCONFIRMED")
    order = []
    cloud = FakeCloud(order)
    cloud.value = "running"
    changed = RemoteConfig(instance="b" * 32, ssh_host="other.npu")
    remote.orchestrator = Orchestrator(remote, changed, cloud=cloud)
    assert remote.work_once()
    assert order == []
    assert remote.task(task["id"])["error"] == "REMOTE-STOP-UNCONFIRMED"
    assert remote.task(task["id"])["worker_paused"]


def test_runner_launch_detaches_all_ssh_descriptors(tmp_path, monkeypatch):
    import os
    import signal
    import subprocess
    import sys
    import time
    import shlex
    from triton_eval.orchestrator import SSHTransport

    directory = tmp_path / "launch"
    (directory / "scripts").mkdir(parents=True)
    (directory / "scripts/run-job.sh").write_text("#!/bin/sh\nsleep 30\n")
    transport = SSHTransport(config())
    processes = []

    def local_command(command, **kwargs):
        result = subprocess.run(shlex.split(command), capture_output=True, timeout=2, check=True)
        processes.append(int(result.stdout))
        return result.stdout

    monkeypatch.setattr(transport, "command", local_command)
    try:
        transport.launch(str(directory))
        assert len(processes) == 1
        os.kill(processes[0], 0)  # launcher has returned while the detached runner is alive
    finally:
        for pid in processes:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_owned_lease_posts_stop_when_get_temporarily_fails(remote):
    task = queued(remote)
    remote.remote_jobs.claim()
    remote.remote_jobs.update(task["id"], start_intent=True, release_state="UNCONFIRMED")
    order = []
    cloud = FakeCloud(order)
    cloud.value = "running"
    original = cloud.state
    calls = [0]

    def state():
        calls[0] += 1
        if calls[0] == 1:
            raise OSError("GET unavailable")
        return original()

    cloud.state = state
    worker = Orchestrator(remote, config(), cloud=cloud)
    assert worker.release_instance(task["id"])
    assert order == ["stop"]
    assert remote.remote_jobs.get(task["id"])["release_state"] == "SHUTDOWN"


def test_watchdog_posts_stop_when_get_unavailable(tmp_path):
    from triton_eval.remote_watchdog import atomic_json, monitor

    lease = tmp_path / "lease.json"
    atomic_json(lease, {"armed": True, "released": False})
    cloud = FakeCloud([])
    cloud.value = "running"
    original = cloud.state

    def cloud_state():
        return original() if cloud.order else (_ for _ in ()).throw(OSError('GET unavailable'))

    cloud.state = cloud_state
    assert monitor(cloud, lease, 123, 0, 2, alive=lambda pid: False) == "SHUTDOWN"
    assert cloud.order == ["stop"]


def test_bundle_reports_in_isolated_python_using_packaged_templates(
    remote, tmp_path, device_fixture
):
    import os
    import subprocess
    import sys

    task = queued(remote)
    job = remote.remote_jobs.get(task["id"])
    remote.invoke_phase(task, "run")  # explicitly fake device records, real SQLite
    assert remote.invoke_phase(task, "score") == 0
    bundle = tmp_path / "isolated-bundle.tar.gz"
    build_bundle(remote, task, job, bundle)
    extracted = tmp_path / "isolated-engine"
    extracted.mkdir()
    with tarfile.open(bundle) as archive:
        archive.extractall(extracted, filter="data")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "triton_eval",
            "--db",
            str(remote.task_directory(task["id"]) / "engine.sqlite"),
            "report",
            "--run",
            "1",
            "--out",
            str(extracted / "reports"),
        ],
        cwd=extracted,
        env=dict(os.environ, PYTHONPATH=str(extracted / "src")),
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode() + result.stdout.decode()
    assert (extracted / "reports/report-run-1.md").is_file()
    assert (extracted / "reports/report-run-1.html").is_file()


def test_transport_failure_recovery_only_pulls_original_job(remote, tmp_path, device_fixture):
    task, order, cloud, transport = setup_execution(remote, tmp_path, "read")
    assert remote.work_once()
    assert remote.task(task["id"])["status"] == "FAILED"
    assert not remote.task_directory(task["id"]).exists()
    transport.failure = None
    assert remote.recover_task(task["id"], {"acknowledge_recovery": True})["status"] == "QUEUED"
    assert remote.work_once()
    assert remote.task(task["id"])["status"] == "COMPLETED"
    assert order.count("send") == order.count("launch") == 1
    assert order.count("start") == order.count("stop") == 2
    assert remote.results(task["id"])["score"]["coverage_ops_core"] == 100


def test_recovery_rejects_queued_and_unacknowledged(remote):
    task = queued(remote)
    with pytest.raises(ServiceError):
        remote.recover_task(task["id"], {"acknowledge_recovery": 1})
    with pytest.raises(ServiceError, match="已释放"):
        remote.recover_task(task["id"], {"acknowledge_recovery": True})


def _watchdog_owner_fixture(tmp_path):
    from dataclasses import asdict
    from triton_eval.remote_watchdog import atomic_json

    state = tmp_path / "cloud-state"
    state.write_text("running")
    helper = tmp_path / "fake_helper.py"
    helper.write_text(
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        "class API:\n"
        " def __init__(self,*a,**kw): pass\n"
        f" def get(self): return SimpleNamespace(state=Path({str(state)!r}).read_text()),{{}}\n"
        f' def request(self,*a,**kw): Path({str(state)!r}).write_text("stopped"); return 200,{{}}\n'
    )
    instance_lock = tmp_path / "instance.lock"
    source_rows = asdict(config(helper=helper, running_timeout=30, shutdown_timeout=3)).items()
    values = asdict(config(helper=helper, running_timeout=30, shutdown_timeout=3))
    payload = {}
    for key, value in values.items():
        payload[key] = str(value) if isinstance(value, Path) else value
    config_path = tmp_path / "config.json"
    atomic_json(config_path, payload)
    marker = tmp_path / "owner-ready"
    child = (
        "import fcntl,time; from pathlib import Path; "
        "from triton_eval.remote_watchdog import LeaseWatchdog; "
        "from triton_eval.orchestrator import RemoteConfig; "
        f'f=Path({str(instance_lock)!r}).open("a"); fcntl.flock(f,fcntl.LOCK_EX); '
        f"g=LeaseWatchdog(RemoteConfig.load(Path({str(config_path)!r})),Path({str(tmp_path)!r}),f.fileno()); "
        f'g.arm(); Path({str(marker)!r}).write_text("ready"); time.sleep(30)'
    )
    return state, marker, child
