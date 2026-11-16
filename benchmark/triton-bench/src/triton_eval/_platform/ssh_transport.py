# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Job-owned SSH multiplexing; bootstrap once, bounded and redacted diagnostics."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import shutil
import tempfile
import time
from pathlib import Path

from triton_eval.submission_service import ServiceError

TRANSIENT_SSH_ERRORS = {
    "REMOTE-SSH",
    "REMOTE-SSH-TRANSPORT",
    "REMOTE-SSH-TIMEOUT",
    "REMOTE-SSH-NOT-READY",
}


def stderr_category(data: bytes) -> str:
    value = data.decode(errors="replace").lower()
    for name, terms in [
        ("AUTH", ("permission denied", "invalid format", "bad permissions")),
        ("HOST_KEY", ("host key verification failed", "host identification has changed")),
        ("LOCAL_REFUSED", ("connection refused", "econnrefused")),
        ("RESET", ("connection reset", "connection closed", "broken pipe")),
        ("BOOTSTRAP", ("bootstrap", "refreshed connect_url")),
        ("MUX", ("mux_client", "control socket", "master is not running")),
        ("TIMEOUT", ("timed out", "timeout")),
    ]:
        if any(term in value for term in terms):
            return name
    return "OTHER"


def redacted_tail(data: bytes) -> str:
    value = data.decode(errors="replace")[-2000:]
    value = re.sub(r'(?:wss?|https?)://[^\s"\']+', "<redacted-url>", value)
    value = re.sub(r'(?i)(bearer\s+)[^\s"\']+', r"\1<redacted>", value)
    value = re.sub(
        r"\b[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b", "<redacted-token>", value
    )
    return value.replace(str(Path.home()), "<home>")


class SSHTransport:
    def __init__(self, config):
        self.config = config
        self.base = [
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
        ]
        self.options = ["-F", str(config.ssh_config), *self.base]
        self.temporary = None
        self.control_path = None
        self.log_path = None
        self.metrics = dict(commands=0, failures=0, sessions=0)

    def open_session(self, directory: Path, *, timeout=None):
        self.log_path = directory / "ssh-diagnostics.jsonl"
        self.drop_master()
        if self.temporary is None:
            self.metrics = dict(commands=0, failures=0, sessions=0)
            # Short path for the macOS Unix socket limit; never share a user's master.
            self.temporary = tempfile.TemporaryDirectory(prefix="te-ssh-", dir="/tmp")
            self.control_path = Path(self.temporary.name) / "mux"
        # Resolve the plugin's Match !exec exactly once per connection generation.
        # A private effective-config snapshot preserves auth/host-key settings while
        # avoiding bootstrap processes and refresh races on every file poll.
        resolved = self._run(
            ["ssh", "-G", "-F", str(self.config.ssh_config), *self.base, self.config.ssh_host],
            label="resolve-config",
            timeout=timeout,
        )
        options = dict(line.split(" ", 1) for line in resolved.decode().splitlines() if " " in line)
        if options.get("proxycommand") == "false":
            raise ServiceError("REMOTE-SSH-NOT-READY", "插件 bootstrap 尚未就绪，不创建会话")
        if "hostname" not in options or "port" not in options:
            raise ServiceError("REMOTE-SSH-CONFIG", "SSH 生效配置缺少目标信息")
        snapshot = Path(self.temporary.name) / "config"
        with snapshot.open("wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(resolved)
        self.options = [
            "-F",
            str(snapshot),
            *self.base,
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=30",
            "-o",
            "ControlPath=" + str(self.control_path),
        ]
        self.metrics["sessions"] += 1
        self._audit(operation="session-opened", generation=self.metrics["sessions"])

    def drop_master(self):
        if self.control_path is not None and self.control_path.exists():
            try:
                subprocess.run(
                    [
                        _client_program("ssh"),
                        *self.options,
                        "-S",
                        str(self.control_path),
                        "-O",
                        "exit",
                        self.config.ssh_host,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
            except subprocess.TimeoutExpired:
                pass
            self.control_path.unlink(missing_ok=True)
            self._audit(operation="master-closed")

    def close(self):
        self.drop_master()
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None
        self.control_path = None
        self.options = ["-F", str(self.config.ssh_config), *self.base]

    def command(self, command: str, *, timeout=None, output=None, label="command") -> bytes:
        return self._run(
            ["ssh", *self.options, self.config.ssh_host, command],
            label=label,
            timeout=timeout,
            output=output,
        )

    def probe(self, *, timeout=None):
        if (
            self.command("printf TE_REMOTE_READY", timeout=timeout, label="probe")
            != b"TE_REMOTE_READY"
        ):
            raise ServiceError("REMOTE-PROTOCOL", "SSH 就绪探针内容不匹配")

    def send(self, archive: Path, remote: str, digest: str, *, timeout=None):
        self.command(
            "mkdir -p " + shlex.quote(self.config.remote_root) + " && mkdir " + shlex.quote(remote),
            timeout=timeout,
            label="create-directory",
        )
        self._run(
            [
                "scp",
                *self.options,
                str(archive),
                self.config.ssh_host + ":" + remote + "/bundle.tar.gz",
            ],
            label="upload-bundle",
            timeout=timeout,
        )
        command = (
            "cd "
            + shlex.quote(remote)
            + " && printf %s "
            + shlex.quote(digest + "  bundle.tar.gz\n")
            + " | sha256sum -c - >/dev/null && tar xzf bundle.tar.gz"
        )
        self.command(command, timeout=timeout, label="verify-bundle")

    def launch(self, remote: str, *, timeout=None):
        script = (
            "import subprocess,sys,pathlib; d=sys.argv[1]; "
            'log=(pathlib.Path(d)/"runner.log").open("ab"); '
            'p=subprocess.Popen(["bash",d+"/scripts/run-job.sh",d,sys.argv[2],sys.argv[3]],'
            "cwd=d,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,"
            "start_new_session=True,close_fds=True); print(p.pid)"
        )
        return self.command(
            shlex.join(["python3", "-c", script, remote, self.config.python, self.config.cann_env]),
            timeout=timeout,
            label="launch-once",
        )

    def read(self, remote: str, offset: int, *, timeout=None) -> bytes:
        script = (
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            'f=p.open("rb") if p.exists() else None; f and f.seek(int(sys.argv[2])); '
            'sys.stdout.buffer.write(f.read(1048577) if f else b"")'
        )
        return self.command(
            shlex.join(["python3", "-c", script, remote + "/remote-events.jsonl", str(offset)]),
            timeout=timeout,
            label="read-events",
        )

    def download(self, remote: str, output: Path, expected_bytes: int, *, timeout=None):
        script = (
            "import pathlib,sys,shutil; p=pathlib.Path(sys.argv[1]); "
            "valid=p.is_file() and not p.is_symlink() and p.stat().st_size==int(sys.argv[2]); "
            "valid or sys.exit(2); "
            'shutil.copyfileobj(p.open("rb"),sys.stdout.buffer)'
        )
        with output.open("wb") as stream:
            self.command(
                shlex.join(
                    ["python3", "-c", script, remote + "/result.tar.gz", str(expected_bytes)]
                ),
                timeout=timeout,
                output=stream,
                label="download-results",
            )

    def _audit(self, **value):
        if self.log_path:
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(dict(time=time.time(), **value), ensure_ascii=False) + "\n")

    def _run(self, argv, *, label, timeout=None, output=None):
        argv = [_client_program(argv[0]), *argv[1:]]
        started = time.monotonic()
        self.metrics["commands"] += 1
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=output or subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout or self.config.ssh_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            self.metrics["failures"] += 1
            self._audit(
                operation=label,
                outcome="TIMEOUT",
                elapsed_s=round(time.monotonic() - started, 3),
                stderr_tail=redacted_tail(exc.stderr or b""),
            )
            raise ServiceError(
                "REMOTE-SSH-TIMEOUT", "SSH 操作在预算内未完成；详见连接诊断"
            ) from None
        if completed.returncode:
            self.metrics["failures"] += 1
            category = stderr_category(completed.stderr)
            self._audit(
                operation=label,
                exit_code=completed.returncode,
                category=category,
                elapsed_s=round(time.monotonic() - started, 3),
                stderr_tail=redacted_tail(completed.stderr),
            )
            if completed.returncode == 255 and category in {"AUTH", "HOST_KEY"}:
                raise ServiceError(
                    "REMOTE-SSH-" + category, "SSH 认证或主机身份检查失败，不自动重试"
                )
            transport_failure = completed.returncode == 255 or category in {
                "LOCAL_REFUSED",
                "RESET",
                "MUX",
                "TIMEOUT",
                "BOOTSTRAP",
            }
            code = "REMOTE-SSH-TRANSPORT" if transport_failure else "REMOTE-COMMAND"
            raise ServiceError(
                code, f"SSH 操作失败 exit={completed.returncode} category={category}；详见连接诊断"
            )
        self._audit(operation=label, exit_code=0, elapsed_s=round(time.monotonic() - started, 3))
        return completed.stdout or b""


def _client_program(name):
    executable = shutil.which(name)
    if executable is None:
        raise ServiceError("REMOTE-CLIENT", "缺少本机 SSH 文件通道程序")
    return str(Path(executable).resolve())
