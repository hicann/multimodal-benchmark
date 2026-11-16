# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Independent local lease guard; survives SIGKILL of the HTTP/orchestrator process."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def owner_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def monitor(
    cloud,
    lease_path: Path,
    owner: int,
    deadline: float,
    shutdown_timeout: int,
    *,
    alive=owner_alive,
    sleep=time.sleep,
    clock=time.time,
) -> str:
    while True:
        lease = json.loads(lease_path.read_text())
        if lease.get("released"):
            return "DISARMED"
        if not alive(owner) or clock() >= deadline:
            if not lease.get("armed"):
                return "NOT_ACQUIRED"
            break
        sleep(0.5)
    return _shutdown_lease(cloud, shutdown_timeout, sleep=sleep, clock=clock)


def _shutdown_lease(cloud, shutdown_timeout, *, sleep, clock):
    stop_deadline, next_stop = clock() + shutdown_timeout, 0
    while clock() < stop_deadline:
        try:
            state = cloud.state()
        except Exception as exc:
            logging.getLogger(__name__).warning("Watchdog state read failed: %s", type(exc).__name__)
            state = None
        if state == "stopped":
            return "SHUTDOWN"
        if state != "stopping" and clock() >= next_stop:
            stopped = _request_shutdown(cloud)
            next_stop = clock() + 15
            if stopped:
                return "SHUTDOWN"
        sleep(min(1, max(0, stop_deadline - clock())))
    return "UNCONFIRMED"


def _request_shutdown(cloud):
    try:
        cloud.stop()
        return cloud.state() == "stopped"
    except Exception as exc:
        logging.getLogger(__name__).warning("Watchdog stop reconciliation failed: %s", type(exc).__name__)
        return False


class LeaseWatchdog:
    def __init__(self, config, directory: Path, instance_fd: int):
        from triton_eval.corpus_assets import ROOT

        ready = directory / "watchdog-ready.json"
        ready.unlink(missing_ok=True)  # A recovery must wait for its NEW guard.
        self.path = directory / "lease.json"
        atomic_json(self.path, dict(armed=False, released=False))
        config_path = directory / "watchdog-config.json"
        atomic_json(
            config_path,
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
        )
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        # Inheriting the flock's open file description keeps other orchestrators out
        # until BOTH parent and guard have finished releasing this instance.
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "triton_eval.remote_watchdog",
                "--config",
                str(config_path),
                "--lease",
                str(self.path),
                "--owner",
                str(os.getpid()),
                "--deadline",
                str(time.time() + config.running_timeout),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(instance_fd,),
            env=environment,
        )
        # Confirm the guard loaded its configuration before any start intent / POST.
        until = time.monotonic() + 10
        while not ready.exists() and self.process.poll() is None and time.monotonic() < until:
            time.sleep(0.05)
        if not ready.exists():
            atomic_json(self.path, dict(armed=False, released=True))
            self.process.wait(timeout=10)
            raise RuntimeError("Watchdog failed to initialize")

    def arm(self):
        atomic_json(self.path, dict(armed=True, released=False))

    def disarm(self):
        atomic_json(self.path, dict(armed=True, released=True))
        self.process.wait(timeout=10)


def main():
    from triton_eval.orchestrator import CannCloud, RemoteConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--owner", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    config = RemoteConfig.load(args.config)
    cloud = CannCloud(config)
    atomic_json(args.lease.with_name("watchdog-ready.json"), dict(ready=True))
    state = monitor(cloud, args.lease, args.owner, args.deadline, config.shutdown_timeout)
    atomic_json(args.lease.with_name("watchdog-result.json"), dict(release_state=state))
    return 0 if state != "UNCONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
