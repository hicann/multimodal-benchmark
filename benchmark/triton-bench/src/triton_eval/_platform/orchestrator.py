# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Single-instance FIFO orchestrator, bounded SSH operations and finally shutdown."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from triton_eval.ssh_transport import SSHTransport, TRANSIENT_SSH_ERRORS
from triton_eval.remote_bundle import build_bundle, restore_archive
from triton_eval.submission_service import ServiceError


@dataclass(frozen=True)
class RemoteConfig:
    instance: str
    ssh_host: str
    helper: Path = Path.home() / "cann-lab-http-retry/cann_retry.py"
    credentials: Path = Path.home() / "cann-lab-http-retry/.credentials.json"
    ssh_config: Path = Path.home() / ".atomgitdevenv/.ssh/config"
    remote_root: str = "/mnt/workspace/jobs"
    python: str = "/home/developer/venv/te-compiler/bin/python"
    cann_env: str = "/home/developer/Ascend/cann-9.0.0/set_env.sh"
    startup_timeout: int = 150
    shutdown_timeout: int = 180
    running_timeout: int = 1800
    idle_timeout: int = 60
    ssh_timeout: int = 20
    poll_interval: float = 2.0
    reconnect_timeout: int = 90
    max_result_bytes: int = 256 * 1024 * 1024

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text())
        for key in ("helper", "credentials", "ssh_config"):
            if key in value:
                value[key] = Path(value[key]).expanduser().absolute()
        config = cls(**value)
        config.validate()
        return config

    def validate(self):
        if not re.fullmatch("[0-9a-f]{32}", self.instance) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.@-]*", self.ssh_host
        ):
            raise ValueError("Invalid instance or SSH host")
        for value in (self.remote_root, self.python, self.cann_env):
            if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in Path(value).parts:
                raise ValueError("Remote paths must be absolute, normalized operator configuration")
        for value in (
            self.startup_timeout,
            self.shutdown_timeout,
            self.running_timeout,
            self.idle_timeout,
            self.ssh_timeout,
            self.max_result_bytes,
            self.reconnect_timeout,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Timeouts and limits must be positive integers")
        if not isinstance(self.poll_interval, (int, float)) or not 0 < self.poll_interval <= 30:
            raise ValueError("Invalid polling interval")


class CannCloud:
    def __init__(self, config: RemoteConfig):
        from triton_eval.remote_credentials import validate_credentials

        validate_credentials(config.credentials)
        self.config = config
        # Import the user's existing retry client. Never duplicate OAuth or record tokens.
        spec = importlib.util.spec_from_file_location("_te_cann_retry", config.helper)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.module = module
        self.api = module.API(config.instance, auth_file=str(config.credentials), timeout=10)

    def state(self):
        return self.api.get()[0].state

    def start(self):
        status, _ = self.api.request("POST", "/start")
        if not 200 <= status < 300:
            raise ServiceError("REMOTE-START", "实例启动请求失败")

    def stop(self):
        status, _ = self.api.request("POST", "/stop")
        if not 200 <= status < 300:
            raise ServiceError("REMOTE-STOP", "实例停止请求失败")

    def prepare(self):
        # Match cann_retry --run's complete ready path, including connection prep.
        snapshot, resource = self.api.get()
        if snapshot.state != "running":
            raise ServiceError("REMOTE-PREPARE", "连接准备前实例不是运行态")
        try:
            self.module.prepare_webide(self.api, resource, False)
            self.reconnect()
        except Exception as exc:
            detail = str(exc) if isinstance(exc, self.module.Halt) else "cann_retry 连接准备失败"
            raise ServiceError("REMOTE-PREPARE", detail) from None

    def hub_lease(self):
        if not hasattr(self.module, "VscodeHubLease"):
            raise ServiceError(
                "REMOTE-HELPER-UPGRADE", "cann_retry 缺少独立 hub 心跳；请更新本机脚本"
            )
        return self.module.VscodeHubLease(self.api.instance)

    def reconnect(self, timeout=None):
        if hasattr(self.module, "connect_vscode"):
            try:
                self.module.connect_vscode(
                    self.api.instance, timeout=min(timeout or self.config.startup_timeout, 600)
                )
            except self.module.Halt:
                raise ServiceError(
                    "REMOTE-SSH-TRANSPORT", "HTTP 连接刷新未就绪，保留原任务证据"
                ) from None




@dataclass
class ExecutionState:
    job: dict
    task: dict
    local: Path
    status: str = "FAILED"
    error: str | None = None
    deadline: float | None = None
    guard: object = None
    hub_lease: object = None
    recovering: bool = False
    remote: str = ""
    bundle: dict | None = None


class Orchestrator:
    def __init__(self, service, config: RemoteConfig, *, cloud=None, transport=None):
        config.validate()
        self.service, self.config, self.jobs = service, config, service.remote_jobs
        self.cloud, self.transport = cloud, transport or SSHTransport(config)
        self.guard_enabled = cloud is None
        self.instance_fd = None

    @staticmethod
    def _validate_result_scope(task, engine):
        actual = [case["case_code"] for case in engine["cases"]]
        invalid_identity = engine["run_status"] != "OK" or engine["package_sha256"] != task["package_sha256"]
        invalid_cases = len(actual) != len(set(actual)) or set(actual) != set(task["case_codes"])
        if invalid_identity or invalid_cases:
            raise ServiceError("REMOTE-RESULT", "结果身份、用例集合或运行状态不一致")

    @staticmethod
    def _disarm_guard(guard):
        try:
            guard.disarm()
        except Exception:
            pass  # Confirmed shutdown remains authoritative; retain guard diagnostics.

    def work_once(self) -> bool:
        # Across service roots, a single local operator can own this one instance.
        path = (
            Path(tempfile.gettempdir()) / f"triton-eval-{os.getuid()}-{self.config.instance}.lock"
        )
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            recovery = self._reconcile_active_jobs()
            if recovery is not None:
                return recovery
            if self.service.stopping.is_set() or (self.service.root / "worker-paused.txt").exists():
                return False
            job = self.jobs.claim()
            if job is None:
                return False
            self.instance_fd = lock.fileno()
            try:
                self._execute(job)
            finally:
                self.instance_fd = None
            return True

    def read_with_recovery(self, task_id, deadline, local, operation):
        """Perform one read under the same recovery and instance lifetime budgets."""
        return self._read_retry(task_id, deadline, local, operation)

    def release_instance(self, task_id):
        """Reconcile and release this task's existing, instance-bound lease."""
        return self._release(task_id)

    def _reconcile_active_jobs(self):
        active = self.jobs.active()
        if not active:
            return None
        if self.cloud is None:
            try:
                self.cloud = CannCloud(self.config)
            except Exception:
                self.service.pause_worker("无法恢复远端控制客户端；保留实例租约等待修复配置")
                return False
        for job in active:
            released = self._release(job["task_id"])
            self.jobs.update(
                job["task_id"], status="INTERRUPTED", phase="interrupted",
                error="SERVICE-RESTART" if released else "REMOTE-STOP-UNCONFIRMED",
            )
        return True

    def _cancelled(self, task_id):
        if self.service.stopping.is_set():
            raise ServiceError("REMOTE-INTERRUPTED", "服务停止，终止本次远端任务")
        if self.service.object_record("task", task_id)["cancel_requested"]:
            raise ServiceError("TASK-CANCELLED", "用户请求取消远端任务")

    def _wait_state(self, state, deadline, task_id=None):
        while time.monotonic() < deadline:
            if task_id:
                self._cancelled(task_id)
            try:
                if self.cloud.state() == state:
                    return
            except Exception:
                pass  # Bounded GET retry; never blindly repeat start.
            time.sleep(min(self.config.poll_interval, max(0, deadline - time.monotonic())))
        raise ServiceError("REMOTE-STATE-TIMEOUT", "实例状态未在预算内确认")

    def _release(self, task_id):
        job = self.jobs.get(task_id)
        if not job["start_intent"]:
            return True
        if job.get("instance") != self.config.instance:
            self.service.pause_worker("当前配置与遗留实例租约不符；请恢复原实例配置再释放")
            return False
        self.jobs.update(task_id, phase="releasing", release_state="UNCONFIRMED")
        deadline = time.monotonic() + self.config.shutdown_timeout
        try:
            self._stop_and_confirm(deadline)
            self.jobs.update(task_id, release_state="SHUTDOWN")
            return True
        except Exception:
            self.service.pause_worker("远端实例关停未确认；下次启动先恢复释放，不派发后续任务")
            return False

    def _stop_and_confirm(self, deadline):
        try:
            initial_state = self.cloud.state()
        except Exception:
            initial_state = None
        if initial_state != "stopped":
            try:
                self.cloud.stop()
            except Exception:
                # A lost POST response is uncertain; first reconcile with GET.
                if self.cloud.state() not in ("stopped", "stopping"):
                    self.cloud.stop()
            self._wait_state("stopped", deadline)

    def _validate_result(self, task, final):
        if self.jobs.get(task["id"])["device_health"] != "HEALTHY":
            raise ServiceError("REMOTE-HEALTH", "缺少实际设备健康准入证据")
        engine = self.service.execution_evidence(task["id"])
        self._validate_result_identity(task, engine)
        if final["status"] == "REJECTED":
            if engine["run_status"] != "INVALID" or not engine["validation"]:
                raise ServiceError("REMOTE-RESULT", "拒收结果缺少引擎证据")
            return
        self._validate_result_scope(task, engine)
        self._validate_result_scores(task, engine)

    def _validate_result_identity(self, task, engine):
        returned_task = json.loads(
            (self.service.task_directory(task["id"]) / "task.json").read_text()
        )
        identity_fields = (
            "manifest_hash", "corpus_version", "package_sha256", "scope", "operators", "case_codes", "profile"
        )
        if any(returned_task.get(key) != task[key] for key in identity_fields):
            raise ServiceError("REMOTE-RESULT-IDENTITY", "回传任务身份与原任务不一致")
        if engine["package_sha256"] != task["package_sha256"]:
            raise ServiceError("REMOTE-RESULT-IDENTITY", "执行器记录的提交指纹不一致")
        from triton_eval.store import Store

        with Store(self.service.task_directory(task["id"]) / "engine.sqlite") as store:
            row = store.conn.execute(
                "SELECT c.manifest_hash,r.profile FROM eval_run r "
                "JOIN corpus_version c ON c.id=r.corpus_version_id"
            ).fetchone()
            if (
                row is None
                or row["manifest_hash"] != task["manifest_hash"]
                or row["profile"] != task["profile"]
            ):
                raise ServiceError("REMOTE-RESULT-IDENTITY", "运行的语料或档位身份不一致")

    def _validate_result_scores(self, task, engine):
        from triton_eval.models import CaseStatus, is_counted

        if any(
            case["status"] in {"ENV_ABORT", "ORACLE_FAIL", "SPEC_ERROR"} for case in engine["cases"]
        ):
            raise ServiceError("EVALUATOR-FAILURE", "评测方异常不作为提交方得分")
        if (
            any(is_counted(CaseStatus(case["status"])) for case in engine["cases"])
            and not engine["scores"]
        ):
            raise ServiceError("REMOTE-SCORE", "结果缺少评分记录")
        if not {"results.jsonl", "report.md", "report.html"} <= {
            item["name"] for item in self.service.artifacts(task["id"])
        }:
            raise ServiceError("REMOTE-ARTIFACTS", "必要产物不完整")

    def _execute(self, job):
        state = self._execution_state(job)
        try:
            self._prepare_execution(state)
            self._start_instance(state)
            self._connect_instance(state)
            self._sync_execution(state)
            self._collect_events(state)
            self._collect_archive(state)
            self._accept_final(state)
        except ServiceError as exc:
            (state.local / "error.json").write_text(
                json.dumps(dict(code=exc.code, detail=exc.detail), ensure_ascii=False) + "\n"
            )
            state.error = exc.code
            states = {"TASK-CANCELLED": "CANCELLED", "REMOTE-INTERRUPTED": "INTERRUPTED"}
            state.status = states.get(exc.code, "FAILED")
        except Exception as exc:
            state.error = "REMOTE-" + type(exc).__name__
        finally:
            self._finish_execution(state)

    def _execution_state(self, job):
        task = self.service.object_record("task", job["task_id"])
        local = self.service.root / "remote" / job["id"]
        local.mkdir(parents=True, exist_ok=True)
        return ExecutionState(job=job, task=task, local=local)

    def _prepare_execution(self, state):
        task_id = state.task["id"]
        self._cancelled(task_id)
        if state.job.get("instance") != self.config.instance:
            raise ServiceError("REMOTE-INSTANCE-CHANGED", "任务绑定的实例与当前配置不一致")
        state.recovering = state.job.get("recovery_requested", False)
        if not state.recovering:
            state.bundle = build_bundle(
                self.service, state.task, state.job, state.local / "bundle.tar.gz"
            )
            self.jobs.update(
                task_id, **{key: value for key, value in state.bundle.items() if key != "files"}
            )
            (state.local / "bundle-index.json").write_text(
                json.dumps(state.bundle, indent=2) + "\n"
            )
        if self.cloud is None:
            self.cloud = CannCloud(self.config)
        if self.cloud.state() != "stopped":
            self.service.pause_worker("实例已被占用；未取得租约，不自动关停他人任务")
            raise ServiceError("REMOTE-INSTANCE-BUSY", "实例初态不是 SHUTDOWN")

    def _start_instance(self, state):
        task_id = state.task["id"]
        self._cancelled(task_id)
        if self.guard_enabled:
            from triton_eval.remote_watchdog import LeaseWatchdog

            state.guard = LeaseWatchdog(self.config, state.local, self.instance_fd)
        self.jobs.update(task_id, start_intent=True, release_state="UNCONFIRMED", phase="starting")
        if state.guard is not None:
            state.guard.arm()
        state.deadline = time.monotonic() + self.config.running_timeout
        if hasattr(self.cloud, "hub_lease"):
            state.hub_lease = self.cloud.hub_lease().start()
        self.cloud.start()
        self._wait_state(
            "running", min(state.deadline, time.monotonic() + self.config.startup_timeout), task_id
        )
        self.jobs.update(task_id, phase="connecting")
        if hasattr(self.cloud, "prepare"):
            self.cloud.prepare()

    def _connect_instance(self, state):
        deadline = min(state.deadline, time.monotonic() + self.config.startup_timeout)
        for attempt in range(3):
            self._cancelled(state.task["id"])
            try:
                if hasattr(self.transport, "open_session"):
                    self.transport.open_session(state.local, timeout=self._remaining(deadline))
                self.transport.probe(timeout=self._remaining(deadline))
                return
            except ServiceError as exc:
                if (
                    exc.code not in TRANSIENT_SSH_ERRORS
                    or attempt == 2
                    or time.monotonic() >= deadline
                ):
                    raise
                if hasattr(self.cloud, "reconnect"):
                    self.cloud.reconnect(timeout=max(1, deadline - time.monotonic()))
                self.service.stopping.wait(self.config.poll_interval)

    def _sync_execution(self, state):
        task_id = state.task["id"]
        self._cancelled(task_id)
        state.remote = (
            state.job["remote_workdir"]
            if state.recovering
            else self.config.remote_root + "/" + state.job["id"]
        )
        if state.recovering:
            self.jobs.update(task_id, phase="recovering")
            return
        self.jobs.update(task_id, phase="syncing", remote_workdir=state.remote)
        self.transport.send(
            state.local / "bundle.tar.gz",
            state.remote,
            state.bundle["bundle_sha256"],
            timeout=self._remaining(state.deadline),
        )
        self._cancelled(task_id)
        self.jobs.update(task_id, phase="running")
        self.transport.launch(state.remote, timeout=self._remaining(state.deadline))

    def _collect_events(self, state):
        task_id = state.task["id"]
        last_event = time.monotonic()
        while not self.jobs.get(task_id).get("final"):
            self._cancelled(task_id)
            current = self.jobs.get(task_id)
            data, outage = self._read_retry(
                task_id,
                state.deadline,
                state.local,
                lambda timeout: self.transport.read(
                    state.remote, current["offset"], timeout=timeout
                ),
            )
            last_event += outage
            if data:
                last_event = self._consume_progress(task_id, current, data, last_event)
            if time.monotonic() - last_event > self.config.idle_timeout:
                raise ServiceError("REMOTE-IDLE", "远端专用事件流超时")
            self.service.stopping.wait(
                min(self.config.poll_interval, max(0, state.deadline - time.monotonic()))
            )
        self._cancelled(task_id)

    def _consume_progress(self, task_id, previous, data, last_event):
        self.jobs.consume(task_id, data)
        current = self.jobs.get(task_id)
        if current["device_health"] == "UNHEALTHY":
            raise ServiceError("REMOTE-HEALTH", "远端设备健康准入失败")
        return time.monotonic() if current["offset"] > previous["offset"] else last_event

    def _collect_archive(self, state):
        task_id = state.task["id"]
        final = self.jobs.get(task_id)["final"]
        if not final.get("archive_sha256"):
            return
        expected_bytes = final.get("archive_bytes")
        if (
            type(expected_bytes) is not int
            or not 0 < expected_bytes <= self.config.max_result_bytes
        ):
            raise ServiceError("REMOTE-ARCHIVE-SIZE", "结果包大小超出预算")
        self.jobs.update(task_id, phase="collecting")
        self._read_retry(
            task_id,
            state.deadline,
            state.local,
            lambda timeout: self.transport.download(
                state.remote, state.local / "result.tar.gz", expected_bytes, timeout=timeout
            ),
        )
        restore_archive(
            state.local / "result.tar.gz",
            self.service.task_directory(task_id),
            final,
            max_bytes=self.config.max_result_bytes,
        )
        if final.get("status") in {"COMPLETED", "REJECTED"}:
            self._validate_result(state.task, final)
            self.jobs.update(task_id, artifacts_verified=True)

    def _accept_final(self, state):
        final = self.jobs.get(state.task["id"])["final"]
        if final.get("status") not in {"COMPLETED", "REJECTED", "FAILED"}:
            raise ServiceError("REMOTE-FINAL", "远端终态非法")
        state.status, state.error = final["status"], final.get("error")
        if (
            state.status == "COMPLETED"
            and not self.jobs.get(state.task["id"])["artifacts_verified"]
        ):
            raise ServiceError("REMOTE-ARTIFACTS", "远端完成但未取回必要产物")

    def _close_execution_connections(self, state):
        released = self._release(state.task["id"])
        if hasattr(self.transport, "close"):
            self.transport.close()
        if state.hub_lease is not None:
            state.hub_lease.close()
        (state.local / "connection-summary.json").write_text(
            json.dumps(
                dict(
                    heartbeat=state.hub_lease.snapshot() if state.hub_lease else None,
                    ssh=getattr(self.transport, "metrics", None),
                ),
                indent=2,
            )
            + "\n"
        )
        return released

    def _finish_execution(self, state):
        released = self._close_execution_connections(state)
        if state.guard is not None and released:
            self._disarm_guard(state.guard)
        if not released:
            state.status, state.error = "FAILED", "REMOTE-STOP-UNCONFIRMED"
        if released and self.service.object_record("task", state.task["id"])["cancel_requested"]:
            state.status, state.error = "CANCELLED", "TASK-CANCELLED"
        try:
            self.jobs.update(
                state.task["id"], status=state.status, phase=state.status.lower(), error=state.error
            )
        except ServiceError:
            self.jobs.update(
                state.task["id"], status="FAILED", phase="failed", error="REMOTE-INCOMPLETE"
            )

    def _read_retry(self, task_id, deadline, local, operation):
        outage_started = None
        recovery_deadline = deadline

        def remaining():
            # A failed tunnel's reconnect window is not the instance lifetime.
            # Preserve the distinction even when a repair consumed its whole budget.
            now = time.monotonic()
            if now >= deadline:
                raise ServiceError("REMOTE-BUDGET", "实例运行总预算耗尽")
            if outage_started is not None and now >= recovery_deadline:
                raise ServiceError("REMOTE-SSH-RECOVERY-TIMEOUT", "SSH 连接未在恢复预算内重新就绪")
            return self._remaining(recovery_deadline)

        for attempt in range(3):
            self._cancelled(task_id)
            attempt_started = time.monotonic()
            try:
                value = operation(remaining())
                return value, time.monotonic() - outage_started if outage_started is not None else 0
            except ServiceError as exc:
                if exc.code not in TRANSIENT_SSH_ERRORS or attempt == 2:
                    raise
                if outage_started is None:
                    outage_started = attempt_started
                    recovery_deadline = min(
                        deadline, outage_started + self.config.reconnect_timeout
                    )
                self._repair_read_session(local, recovery_deadline, remaining)
                self.service.stopping.wait(min(0.5 * (attempt + 1), remaining()))
        raise AssertionError("unreachable")

    def _repair_read_session(self, local, recovery_deadline, remaining):
        try:
            if hasattr(self.cloud, "reconnect"):
                self.cloud.reconnect(timeout=max(1, recovery_deadline - time.monotonic()))
            if hasattr(self.transport, "open_session"):
                self.transport.open_session(local, timeout=remaining())
        except ServiceError as repair:
            if repair.code not in TRANSIENT_SSH_ERRORS:
                raise

    def _remaining(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ServiceError("REMOTE-BUDGET", "实例运行总预算耗尽")
        return min(remaining, self.config.ssh_timeout)
