# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Durable remote control plane. Task, event and cursor updates commit together."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

from triton_eval.submission_service import ServiceError, _now

REMOTE_JOB_DDL = """
CREATE TABLE IF NOT EXISTS remote_job (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL UNIQUE REFERENCES objects(id),
    status TEXT NOT NULL, phase TEXT NOT NULL, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS remote_job_queue ON remote_job(status, created_at, id);
CREATE TABLE IF NOT EXISTS remote_job_event (
    seq INTEGER PRIMARY KEY REFERENCES events(seq),
    job_id TEXT NOT NULL REFERENCES remote_job(id), event_seq INTEGER NOT NULL,
    digest TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(job_id, event_seq)
);
"""
MAX_EVENT_BYTES = 1024 * 1024
KINDS = {"phase", "health", "engine", "heartbeat", "final"}


def canonical(value: dict) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class RemoteJobs:
    def __init__(self, service):
        self.service = service
        with service.database_connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ServiceError("REMOTE-SCHEMA", "不支持的服务数据库版本", 409)
            db.executescript(
                "BEGIN IMMEDIATE;\n" + REMOTE_JOB_DDL + "\nPRAGMA user_version=1; COMMIT;"
            )

    @staticmethod
    def _write(db, job, task, *, emit=True):
        job["updated_at"] = task["updated_at"] = _now()
        db.execute(
            "UPDATE remote_job SET status=?,phase=?,updated_at=?,payload=? WHERE id=?",
            (job["status"], job["phase"], job["updated_at"], canonical(job), job["id"]),
        )
        db.execute(
            "UPDATE objects SET payload=? WHERE kind='task' AND id=?", (canonical(task), task["id"])
        )
        if emit:
            db.execute(
                "INSERT INTO events(task_id,payload) VALUES (?,?)",
                (
                    task["id"],
                    canonical(
                        dict(
                            time=_now(),
                            status=task["status"],
                            error=task.get("error"),
                            kind="state",
                            cancel_requested=task["cancel_requested"],
                            phase=job["phase"],
                            release_state=job["release_state"],
                        )
                    ),
                ),
            )

    def enqueue(self, db, task: dict) -> None:
        job = dict(
            id=uuid.uuid4().hex,
            task_id=task["id"],
            status="QUEUED",
            phase="queued",
            created_at=task["created_at"],
            updated_at=_now(),
            offset=0,
            event_seq=0,
            release_state="NOT_ACQUIRED",
            start_intent=False,
            artifacts_verified=False,
            device_health="NOT_PROBED",
            instance=self.service.remote_config.instance,
        )
        db.execute(
            "INSERT INTO remote_job VALUES (?,?,?,?,?,?,?)",
            (
                job["id"],
                task["id"],
                job["status"],
                job["phase"],
                job["created_at"],
                job["updated_at"],
                canonical(job),
            ),
        )

    def get(self, task_id: str) -> dict | None:
        with self.service.database_connection() as db:
            row = db.execute(
                "SELECT payload FROM remote_job WHERE task_id=?", (task_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def active(self) -> list[dict]:
        with self.service.database_connection() as db:
            rows = db.execute(
                "SELECT payload FROM remote_job WHERE status IN ('RUNNING','SCORING','REPORTING') "
                "OR json_extract(payload,'$.release_state')='UNCONFIRMED'"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def claim(self) -> dict | None:
        with self.service.lock, self.service.database_connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM remote_job WHERE status IN ('RUNNING','SCORING','REPORTING') "
                "OR json_extract(payload,'$.release_state')='UNCONFIRMED'"
            ).fetchone():
                return None
            row = db.execute(
                "SELECT payload FROM remote_job WHERE status='QUEUED' "
                "ORDER BY created_at,id LIMIT 1"
            ).fetchone()
            if not row:
                return None
            job = json.loads(row[0])
            task = json.loads(
                db.execute("SELECT payload FROM objects WHERE id=?", (job["task_id"],)).fetchone()[
                    0
                ]
            )
            job.update(status="RUNNING", phase="preparing")
            task.update(status="RUNNING", error=None)
            self._write(db, job, task)
            return job

    def update(self, task_id, **changes) -> dict:
        with self.service.lock, self.service.database_connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = json.loads(
                db.execute("SELECT payload FROM remote_job WHERE task_id=?", (task_id,)).fetchone()[
                    0
                ]
            )
            task = json.loads(
                db.execute("SELECT payload FROM objects WHERE id=?", (task_id,)).fetchone()[0]
            )
            job.update(changes)
            if "status" in changes:
                if changes["status"] == "COMPLETED" and (
                    not job["artifacts_verified"] or job["release_state"] != "SHUTDOWN"
                ):
                    raise ServiceError("REMOTE-INCOMPLETE", "产物未核验或实例尚未释放", 409)
                task["status"] = changes["status"]
            if "error" in changes:
                task["error"] = changes["error"]
            self._write(db, job, task)
            return job

    def accept(self, task_id: str, envelope: dict, end_offset: int) -> bool:
        """Commit one authenticated, ordered event and its public cursor atomically."""
        serialized = _serialize_envelope(envelope)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        with self.service.lock, self.service.database_connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = json.loads(
                db.execute("SELECT payload FROM remote_job WHERE task_id=?", (task_id,)).fetchone()[
                    0
                ]
            )
            if envelope["job_id"] != job["id"]:
                raise ServiceError("REMOTE-EVENT-IDENTITY", "事件任务身份不一致", 409)
            if _replayed_event(db, job, envelope, digest, end_offset):
                return False
            _require_event_order(job, envelope, end_offset)
            task = json.loads(
                db.execute("SELECT payload FROM objects WHERE id=?", (task_id,)).fetchone()[0]
            )
            _apply_event_payload(job, task, envelope)
            job.update(offset=end_offset, event_seq=envelope["event_seq"])
            self._write(db, job, task, emit=False)
            _persist_event(db, job, task, envelope, serialized)
        return True

    def consume(self, task_id: str, data: bytes) -> int:
        """An incomplete trailing line is fetched again at the committed byte offset."""
        offset = self.get(task_id)["offset"]
        for line in data.splitlines(keepends=True):
            if len(line) > MAX_EVENT_BYTES:
                raise ServiceError("REMOTE-EVENT-SIZE", "远端事件行过大", 409)
            if not line.endswith(b"\n"):
                break
            try:
                value = json.loads(line)
            except ValueError:
                raise ServiceError("REMOTE-EVENT-JSON", "远端事件行不可解析", 409) from None
            self.accept(task_id, value, offset + len(line))
            offset += len(line)
        return offset


def _envelope_fields(envelope):
    if set(envelope) != {"protocol_version", "job_id", "event_seq", "kind", "payload"}:
        return False
    if type(envelope["protocol_version"]) is not int or envelope["protocol_version"] != 1:
        return False
    if type(envelope["event_seq"]) is not int or envelope["event_seq"] <= 0:
        return False
    return envelope["kind"] in KINDS and isinstance(envelope["payload"], dict)


def _serialize_envelope(envelope):
    try:
        serialized = canonical(envelope)
        valid = _envelope_fields(envelope) and len(serialized.encode()) <= MAX_EVENT_BYTES
    except (TypeError, ValueError, KeyError, AttributeError):
        valid = False
    if not valid:
        raise ServiceError("REMOTE-EVENT", "远端事件协议无效", 409)
    return serialized


def _replayed_event(db, job, envelope, digest, end_offset):
    previous = db.execute(
        "SELECT digest FROM remote_job_event WHERE job_id=? AND event_seq=?",
        (job["id"], envelope["event_seq"]),
    ).fetchone()
    if not previous:
        return False
    if previous[0] != digest:
        raise ServiceError("REMOTE-EVENT-CONFLICT", "相同事件序号对应不同内容", 409)
    if end_offset > job["offset"]:
        raise ServiceError("REMOTE-EVENT-OFFSET", "重复事件出现在新的文件位置", 409)
    return True


def _require_event_order(job, envelope, end_offset):
    if envelope["event_seq"] != job["event_seq"] + 1 or end_offset <= job["offset"]:
        raise ServiceError("REMOTE-EVENT-GAP", "事件序号或文件偏移不连续", 409)
    if job.get("final"):
        raise ServiceError("REMOTE-EVENT-FINAL", "final 后不能追加事件", 409)


def _apply_phase(job, task, payload):
    phase = payload.get("phase")
    states = {"running": "RUNNING", "scoring": "SCORING", "reporting": "REPORTING"}
    if phase not in states:
        raise ServiceError("REMOTE-PHASE", "远端阶段无效", 409)
    ranks = {"RUNNING": 0, "SCORING": 1, "REPORTING": 2}
    if job["status"] not in ranks or ranks.get(states.get(phase), -1) < ranks.get(job["status"], -1):
        raise ServiceError("REMOTE-PHASE", "远端阶段不能倒退", 409)
    job.update(phase=phase, status=states[phase])
    task["status"] = states[phase]


def _require_case_event(task, payload):
    from triton_eval.models import CaseResultRecord

    try:
        record = CaseResultRecord.model_validate(payload.get("record"))
    except ValueError:
        raise ServiceError("REMOTE-CASE", "用例事件不符合结果契约", 409) from None
    if record.case_code not in task["case_codes"] or record.operator not in task["operators"]:
        raise ServiceError("REMOTE-CASE", "用例事件不在任务范围内", 409)


def _apply_event_payload(job, task, envelope):
    kind, payload = envelope["kind"], envelope["payload"]
    if kind == "phase":
        _apply_phase(job, task, payload)
    elif kind == "health":
        job["device_health"] = "HEALTHY" if payload.get("ok") is True else "UNHEALTHY"
    elif kind == "final":
        job["final"] = payload
    elif kind == "engine" and payload.get("event") == "case":
        _require_case_event(task, payload)


def _persist_event(db, job, task, envelope, serialized):
    public = dict(time=_now(), status=task["status"], kind=envelope["kind"], remote=envelope)
    cursor = db.execute(
        "INSERT INTO events(task_id,payload) VALUES (?,?)", (task["id"], canonical(public))
    ).lastrowid
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    db.execute(
        "INSERT INTO remote_job_event VALUES (?,?,?,?,?)",
        (cursor, job["id"], envelope["event_seq"], digest, serialized),
    )
