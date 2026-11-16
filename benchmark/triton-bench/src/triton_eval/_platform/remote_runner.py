# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""One remote execution, existing evaluator underneath; stdout is diagnostic only."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import threading
import time
from pathlib import Path

from triton_eval.remote_bundle import sha256, verify_job
from triton_eval.remote_jobs import MAX_EVENT_BYTES, canonical
from triton_eval.submission_service import EvaluationService, TERMINAL


class Journal:
    def __init__(self, path: Path, job_id: str):
        self.path, self.job_id, self.sequence = path, job_id, 0
        if path.exists():
            for line in path.read_bytes().splitlines(keepends=True):
                event = json.loads(line)
                if (
                    not line.endswith(b"\n")
                    or event["job_id"] != job_id
                    or event["event_seq"] != self.sequence + 1
                ):
                    raise ValueError("Damaged remote journal")
                self.sequence += 1

    def emit(self, kind: str, payload: dict):
        event = dict(
            protocol_version=1,
            job_id=self.job_id,
            event_seq=self.sequence + 1,
            kind=kind,
            payload=payload,
        )
        data = (canonical(event) + "\n").encode()
        if len(data) > MAX_EVENT_BYTES:
            raise ValueError("Event exceeds protocol limit")
        with self.path.open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1


def bridge_engine(journal: Journal, path: Path, offset: int) -> int:
    if not path.exists():
        return offset
    with path.open("rb") as stream:
        stream.seek(offset)
        for line in stream:
            if len(line) > MAX_EVENT_BYTES:
                raise ValueError("Engine event exceeds limit")
            if not line.endswith(b"\n"):
                break
            journal.emit("engine", json.loads(line))
            offset += len(line)
    return offset


def probe_health() -> dict:
    from triton_eval.envprobe.collect import probe_remote_health

    return probe_remote_health()


def collect(service, task_id: str, directory: Path) -> dict:
    source = service.task_directory(task_id)
    files = {"engine.sqlite": source / "engine.sqlite"}
    files.update(
        {
            path.relative_to(source).as_posix(): path
            for path in service.artifact_paths(task_id).values()
            if path.is_file()
        }
    )
    for path in (source / "runs/run-1").glob("*.json*"):
        files[path.relative_to(source).as_posix()] = path
    files["task.json"] = directory / "task-result.json"
    files["task.json"].write_text(canonical(service.task(task_id)) + "\n")
    index = {}
    archive_path = directory / "result.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, path in sorted(files.items()):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Invalid result member")
            index[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
            archive.add(path, arcname=name, recursive=False)
    task = service.task(task_id)
    return dict(
        status=task["status"],
        error=task["error"],
        archive_bytes=archive_path.stat().st_size,
        archive_sha256=sha256(archive_path),
        files=index,
    )


def run(directory: Path, *, health_probe=probe_health) -> int:
    directory = directory.resolve()
    spec = json.loads((directory / "job.json").read_text())
    # Never silently rerun a partially executed submission after a runner restart.
    marker = directory / "runner-started"
    if (directory / "final.json").exists():
        return 0
    with marker.open("x") as stream:
        stream.write(str(os.getpid()))
    journal = Journal(directory / "remote-events.jsonl", spec["job_id"])
    service = None
    try:
        spec = verify_job(directory)
        health = health_probe()
        journal.emit("health", health)
        if health.get("ok") is not True:
            raise RuntimeError("Health admission rejected")
        service = EvaluationService(directory / "service", directory / spec["corpus"], execute=True)
        submission = service.upload((directory / "submission.zip").read_bytes())
        request = {key: spec["task"][key] for key in ("scope", "profile")}
        request["submission_id"] = submission["id"]
        if request["scope"] != "full":
            request["operators"] = spec["task"]["operators"]
        task = service.create_task(request)
        if task["case_codes"] != spec["task"]["case_codes"]:
            raise ValueError("Remote task scope changed")
        _run_worker_journal(service, task, journal)
        final = collect(service, task["id"], directory)
        journal.emit("final", final)
        (directory / "final.json").write_text(canonical(final) + "\n")
        return 0
    except Exception as exc:
        # No raw exception strings: third party imports may expose environment/URLs.
        if service is not None:
            service.close()
        journal.emit("final", dict(status="FAILED", error="RUNNER-" + type(exc).__name__))
        return 1


def _run_worker_journal(service, task, journal):
    cursor, engine_offset, previous, last_heartbeat = 0, 0, None, time.monotonic()
    service.start_worker()
    while True:
        events = service.events(task["id"], cursor)
        for event in events["items"]:
            status = event["status"]
            if status in {"RUNNING", "SCORING", "REPORTING"} and status != previous:
                journal.emit("phase", dict(phase=status.lower()))
                previous = status
        cursor = events["next_cursor"]
        path = service.task_directory(task["id"]) / "runs/run-1/events.jsonl"
        engine_offset = bridge_engine(journal, path, engine_offset)
        if service.task(task["id"])["status"] in TERMINAL:
            break
        if time.monotonic() - last_heartbeat >= 10:
            journal.emit("heartbeat", dict(elapsed_s=round(time.monotonic() - last_heartbeat)))
            last_heartbeat = time.monotonic()
        time.sleep(0.1)
    service.close()
    # The worker may finish scoring/reporting while fsyncing a burst of case
    # events. Drain its durable state journal after joining, too.
    for event in service.events(task["id"], cursor)["items"]:
        status = event["status"]
        if status in {"RUNNING", "SCORING", "REPORTING"} and status != previous:
            journal.emit("phase", dict(phase=status.lower()))
            previous = status
    bridge_engine(journal, path, engine_offset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    return run(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
