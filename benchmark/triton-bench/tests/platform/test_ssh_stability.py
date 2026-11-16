# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Transport failure classification, side-effect boundaries and local session ownership."""

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_eval.orchestrator import SSHTransport, Orchestrator
from triton_eval.submission_service import ServiceError
from test_remote_platform import remote as _remote_fixture, device_fixture as _device_fixture
from test_remote_platform import queued, setup_execution, config


def test_private_effective_config_avoids_bootstrap_each_poll(tmp_path, monkeypatch):
    calls = []
    effective = (
        b"hostname 127.0.0.1\nport 41201\nuser developer\nstricthostkeychecking false\nidentityfile /private/key\n"
    )

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=effective if "-G" in argv else b"ok", stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    transport = SSHTransport(config())
    try:
        transport.open_session(tmp_path)
        assert transport.command("printf ok") == b"ok"
        assert transport.command("printf ok") == b"ok"
        assert sum("-G" in call for call in calls) == 1
        assert transport.options[1] != str(config().ssh_config)
        assert Path(transport.options[1]).read_bytes() == effective
        assert Path(transport.options[1]).stat().st_mode & 0o777 == 0o600
        assert "ControlMaster=auto" in transport.options
        assert str(transport.control_path).startswith("/tmp/te-ssh-")
    finally:
        transport.close()


@pytest.mark.parametrize(
    "rc,stderr,code",
    [
        (255, b"Permission denied (publickey)", "REMOTE-SSH-AUTH"),
        (255, b"Connection reset by peer", "REMOTE-SSH-TRANSPORT"),
        (1, b"Traceback: FileNotFoundError", "REMOTE-COMMAND"),
    ],
)

def test_stderr_classification_redacts_tokens(tmp_path, monkeypatch, rc, stderr, code):
    tail = stderr + b" wss://private.example/path?token=secret Bearer very-private-token"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=rc, stdout=b"", stderr=tail))
    transport = SSHTransport(config())
    transport.log_path = tmp_path / "diagnostics.jsonl"
    with pytest.raises(ServiceError) as error:
        transport.command("fixed helper")
    assert error.value.code == code
    saved = transport.log_path.read_text()
    assert "very-private-token" not in saved and "private.example" not in saved
    assert json.loads(saved)["exit_code"] == rc


def test_transient_read_reconnects_in_same_job_without_relaunch(remote, tmp_path, device_fixture):
    task, order, cloud, transport = setup_execution(remote, tmp_path)
    original = transport.read
    count = [0]

    def read(*args, **kw):
        count[0] += 1
        if count[0] == 1:
            raise ServiceError("REMOTE-SSH-TRANSPORT", "injected reset")
        return original(*args, **kw)

    transport.read = read

    def cloud_reconnect(**kw):
        return order.append('reconnect')

    cloud.reconnect = cloud_reconnect
    assert remote.work_once()
    assert remote.task(task["id"])["status"] == "COMPLETED"
    assert order.count("reconnect") == 1
    assert order.count("start") == order.count("send") == order.count("launch") == 1


def test_remote_script_error_does_not_refresh_tunnel(remote, tmp_path, device_fixture):
    task, order, cloud, transport = setup_execution(remote, tmp_path)

    def transport_read(*a, **kw):
        return (_ for _ in ()).throw(ServiceError('REMOTE-COMMAND', 'exit1'))

    transport.read = transport_read

    def cloud_reconnect(**kw):
        return pytest.fail('remote program exit is not a tunnel failure')

    cloud.reconnect = cloud_reconnect
    assert remote.work_once()
    assert remote.task(task["id"])["error"] == "REMOTE-COMMAND"
    assert order.count("launch") == 1


def test_partial_download_retries_and_hash_checks_full_archive(remote, tmp_path, device_fixture):
    task, order, cloud, transport = setup_execution(remote, tmp_path)
    original = transport.download
    count = [0]

    def download(remote_path, output, expected_bytes, **kw):
        count[0] += 1
        if count[0] == 1:
            output.write_bytes(b"partial")
            raise ServiceError("REMOTE-SSH-TRANSPORT", "injected download reset")
        return original(remote_path, output, expected_bytes, **kw)

    transport.download = download

    def cloud_reconnect(**kw):
        return order.append('reconnect')

    cloud.reconnect = cloud_reconnect
    assert remote.work_once()
    assert remote.task(task["id"])["status"] == "COMPLETED"
    assert count[0] == 2 and order.count("launch") == 1


def test_uncertain_launch_never_retried(remote, tmp_path):
    task, order, cloud, transport = setup_execution(remote, tmp_path)

    def launch(*a, **kw):
        order.append("launch")
        raise ServiceError("REMOTE-SSH-TRANSPORT", "launch response lost")

    transport.launch = launch

    def cloud_reconnect(**kw):
        return pytest.fail('must not blindly repeat a launch')

    cloud.reconnect = cloud_reconnect
    assert remote.work_once()
    assert remote.task(task["id"])["status"] == "FAILED"
    assert order.count("launch") == 1 and order[-1] == "stop"


def test_hub_lease_spans_start_execution_and_shutdown(remote, tmp_path, device_fixture):
    task, order, cloud, transport = setup_execution(remote, tmp_path)

    class Lease:
        @staticmethod
        def close():
            order.append("lease-close")

        @staticmethod
        def snapshot():
            return {"successes": 9}

        def start(self):
            order.append("lease-start")
            return self

    def cloud_hub_lease():
        return Lease()

    cloud.hub_lease = cloud_hub_lease
    assert remote.work_once()
    assert order.index("lease-start") < order.index("start")
    assert order.index("stop") < order.index("lease-close")
    job = remote.remote_jobs.get(task["id"])
    summary = json.loads((remote.root / "remote" / job["id"] / "connection-summary.json").read_text())
    assert summary["heartbeat"]["successes"] == 9


def test_recovery_budget_includes_the_first_timed_out_read(monkeypatch, tmp_path):
    import triton_eval.orchestrator as module

    clock = [100.0]
    budgets = []
    calls = [0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    worker = recovery_worker(lambda delay: clock.__setitem__(0, clock[0] + delay))

    def reconnect(timeout):
        budgets.append(timeout)
        clock[0] += 10

    worker.cloud = SimpleNamespace(reconnect=reconnect)
    worker.transport = object()

    def operation(timeout):
        calls[0] += 1
        if calls[0] == 1:
            clock[0] += 35
            raise ServiceError("REMOTE-SSH-TIMEOUT", "read timeout")
        return b"ok"

    data, outage = worker.read_with_recovery("task", 1000, tmp_path, operation)
    assert data == b"ok" and budgets == [55]
    assert outage == 45.5


@pytest.mark.parametrize("instance_deadline,expected", [(1000, "REMOTE-SSH-RECOVERY-TIMEOUT"), (180, "REMOTE-BUDGET")])
def test_expired_reconnect_window_is_not_misreported_as_instance_budget(
    monkeypatch, tmp_path, instance_deadline, expected
):
    import triton_eval.orchestrator as module

    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    worker = recovery_worker(lambda delay: None)

    def reconnect(timeout):
        clock[0] += timeout + 1

    worker.cloud = SimpleNamespace(reconnect=reconnect)
    worker.transport = SimpleNamespace(open_session=lambda *a, **kw: pytest.fail("repair budget already expired"))

    def operation(timeout):
        clock[0] += 35
        raise ServiceError("REMOTE-SSH-TIMEOUT", "bounded transient read timeout")

    with pytest.raises(ServiceError) as error:
        worker.read_with_recovery("task", instance_deadline, tmp_path, operation)
    assert error.value.code == expected


def recovery_worker(wait):
    service = SimpleNamespace(
        remote_jobs=object(),
        stopping=SimpleNamespace(is_set=lambda: False, wait=wait),
        object_record=lambda kind, identifier: {"cancel_requested": False},
    )
    return Orchestrator(service, config(reconnect_timeout=90), cloud=SimpleNamespace(), transport=SimpleNamespace())


remote_service = pytest.fixture(name="remote")(_remote_fixture.__wrapped__)
synthetic_device = pytest.fixture(name="device_fixture")(_device_fixture.__wrapped__)
