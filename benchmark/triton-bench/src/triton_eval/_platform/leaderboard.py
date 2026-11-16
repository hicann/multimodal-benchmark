# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Trial rankings with exact-condition cohorts and persistent per-operator anchors.

Published task scores remain untouched. Ranking performance uses this control
plane's shared anchors, never each isolated task's self-anchored score.
"""

from __future__ import annotations
import hashlib
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass
from triton_eval.bench.stats import geometric_mean
from triton_eval.scorer import perf_score_against_anchor
from triton_eval.anchors import weighted_perf_score
from triton_eval.specload import load_spec
from triton_eval.submission_service import ServiceError


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def metadata(service, task):
    path = service.task_directory(task["id"]) / "engine.sqlite"
    if not path.is_file():
        return None
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT env_json,env_fp,status,probe_status FROM eval_run"
            ).fetchall()
            scores = connection.execute(
                "SELECT scorer_version FROM run_score ORDER BY id DESC LIMIT 1"
            ).fetchall()
    except sqlite3.Error:
        return None
    if len(rows) != 1:
        return None
    row = dict(rows[0])
    row["scorer_version"] = scores[0][0] if scores else None
    try:
        row["environment"] = json.loads(row.pop("env_json"))
    except (ValueError, TypeError):
        return None
    return row if isinstance(row["environment"], dict) and row["env_fp"] else None


def timing(row, repeats):
    samples = row.get("samples_ms") or []
    values = [
        row.get("perf_median_ms"),
        row.get("baseline_median_ms"),
        row.get("speedup"),
        *samples,
    ]
    if len(samples) != repeats or not all(
        isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in values
    ):
        return False
    return math.isclose(
        statistics.median(samples), row["perf_median_ms"], rel_tol=1e-8
    ) and math.isclose(
        row["baseline_median_ms"] / row["perf_median_ms"], row["speedup"], rel_tol=1e-8
    )


def cohort_for(service, task, meta):
    remote = service.remote_jobs.get(task["id"])
    identity = dict(
        manifest_hash=task["manifest_hash"],
        assets_hash=(
            remote.get("assets_hash")
            if remote
            else json.loads((service.corpus / "evaluation-assets.json").read_text())["assets_hash"]
        ),
        profile=task["profile"],
        scope=task["scope"],
        operators=sorted(task["operators"]),
        env_fp=meta["env_fp"],
        scorer_version=meta["scorer_version"],
        protocol_version=1,
        execution="remote" if remote else "local",
    )
    env = meta["environment"]
    field_names = (
        "compiler_name", "triton_import_version", "cann_version", "npu_model", "driver_version", "clock_locked"
    )
    public = {k: env.get(k) for k in field_names}
    return dict(
        id=digest(identity),
        **identity,
        corpus_version=task["corpus_version"],
        planned_cases=task["planned_cases"],
        environment=public,
        task_count=0,
    )


@dataclass
class CohortGroups:
    cohorts: dict
    members: dict
    excluded: list
    target: str | None


def _candidate_inputs(service, task, meta):
    if not _run_admitted(task, meta):
        return None, "运行未通过完整性或设备探针检查"
    if not _remote_admitted(service, task):
        return None, "远端文件核验或资源释放未确认"
    result = service.results(task["id"])
    expected = {case.code for case in service.manifest.cases if case.operator in task["operators"]}
    if not _result_admitted(service, task, result, expected):
        return None, "计分、用例集合或资产身份不完整"
    return result, expected


def _run_admitted(task, meta):
    invalid_task = task["cancel_requested"] or task.get("error")
    invalid_run = meta["status"] != "OK" or meta["probe_status"] != "PASS"
    return not (invalid_task or invalid_run)


def _remote_admitted(service, task):
    remote = service.remote_jobs.get(task["id"])
    if not remote:
        return True
    released = remote.get("release_state") == "SHUTDOWN" and remote.get("artifacts_verified")
    identity = remote.get("manifest_hash") == task["manifest_hash"] and remote.get("assets_hash")
    return released and identity


def _result_admitted(service, task, result, expected):
    if (
        task["manifest_hash"] != service.manifest.manifest_hash
        or result["engine_error"]
        or not result["score"]
    ):
        return False
    rows = result["cases"]
    if len(rows) != len(expected):
        return False
    return {row["case_code"] for row in rows} == expected and result["counted_cases"] == len(
        expected
    )


def _operator_geomeans(task, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["operator"]].append(row)
    profile = load_spec().profile(task["profile"])
    op_geo = {}
    for op, cases in groups.items():
        if profile.measure_perf and all(
            r["status"] == "PASS" and timing(r, profile.repeats) for r in cases
        ):
            op_geo[op] = geometric_mean([r["speedup"] for r in cases])
    return op_geo, groups


def candidate(service, task, meta):
    result, expected = _candidate_inputs(service, task, meta)
    if result is None:
        return None, expected
    rows = result["cases"]
    op_geo, groups = _operator_geomeans(task, rows)
    warnings = []
    if not meta["environment"].get("clock_locked"):
        warnings.append("运行频率未锁定")
    if not meta["environment"].get("driver_version"):
        warnings.append("驱动版本未记录")
    profile = load_spec().profile(task["profile"])
    unstable = sum((r.get("perf_cv") or 0) > profile.cv_gate for r in rows)
    if unstable:
        warnings.append(f"{unstable} 个用例测速波动超阈值")
    submission = service.submission(task["submission_id"])
    value = dict(
        task_id=task["id"],
        submit_id=submission["declared_submit_id"] or task["id"],
        package_sha256=task["package_sha256"],
        created_at=task["created_at"],
        functional_score=result["score"]["coverage_ops_core"],
        strict_score=result["score"]["coverage_strict"],
        passed_cases=sum(r["status"] == "PASS" for r in rows),
        planned_cases=len(expected),
        geometric_speedup=(
            geometric_mean([r["speedup"] for r in rows]) if len(op_geo) == len(groups) else None
        ),
        shared_perf_score=None,
        scored_operators=0,
        total_operators=len(groups),
        unstable_cases=unstable,
        warnings=warnings,
        op_geo=op_geo,
    )
    return value, None


def _load_tasks(service):
    with service.database_connection() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS leaderboard_anchor (
            cohort TEXT NOT NULL, operator TEXT NOT NULL, task_id TEXT NOT NULL,
            geo REAL NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(cohort,operator))""")
        source_rows = connection.execute(
            "SELECT payload FROM objects WHERE kind='task' ORDER BY json_extract(payload,'$.created_at'),id",
        )
        tasks = [json.loads(row[0]) for row in source_rows]
    return tasks


def _group_tasks(service, tasks, task_id):
    cohorts, members, excluded, target = {}, defaultdict(list), [], None
    for task in tasks:
        if task["status"] != "COMPLETED":
            excluded.append(dict(task_id=task["id"], reason="任务尚未完整完成或已失败/取消"))
            continue
        meta = metadata(service, task)
        if not meta:
            excluded.append(dict(task_id=task["id"], reason="没有唯一可追溯的运行环境"))
            continue
        group = cohort_for(service, task, meta)
        key = group["id"]
        cohorts.setdefault(key, group)
        cohorts[key]["task_count"] += 1
        members[key].append((task, meta))
        if task["id"] == task_id:
            target = key
    return CohortGroups(cohorts, members, excluded, target)


def _select_cohort(groups, cohort, task_id):
    cohorts, members, target = groups.cohorts, groups.members, groups.target
    if task_id and target is None:
        raise ServiceError("LEADERBOARD-TASK", "该任务尚未形成完整榜单成绩", 409)
    selected = (
        cohort
        or target
        or (max(cohorts, key=lambda key: members[key][-1][0]["created_at"]) if cohorts else None)
    )
    if selected is not None and selected not in cohorts:
        raise ServiceError("NOT-FOUND", "榜单范围不存在", 404)
    return selected


def _cohort_entries(service, selected, members, excluded):
    entries = []
    for task, meta in members.get(selected, []):
        try:
            entry, reason = candidate(service, task, meta)
        except (ValueError, sqlite3.Error, ServiceError):
            entry, reason = None, "计分数据或提交身份不可读取"
        if reason:
            excluded.append(dict(task_id=task["id"], reason=reason))
            continue
        entries.append(entry)
        # Each operator freezes independently; a bad operator cannot replace
        # another operator's existing anchor or cause it to be refrozen.
        with service.database_connection() as connection:
            for operator, geo in entry["op_geo"].items():
                connection.execute(
                    "INSERT OR IGNORE INTO leaderboard_anchor VALUES (?,?,?,?,?)",
                    (
                        selected,
                        operator,
                        entry["task_id"],
                        geo,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
    return entries


def _load_anchors(service, selected):
    with service.database_connection() as connection:
        source_rows = connection.execute(
            'SELECT operator,task_id,geo FROM leaderboard_anchor WHERE cohort=? ORDER BY operator', (selected,),
        )
        anchors = [dict(operator=r[0], task_id=r[1], geometric_speedup=r[2]) for r in source_rows]
    return anchors


def _score_entries(service, entries, anchors):
    weights = {o.name: o.weight for o in service.manifest.operators}
    valid_anchors = {
        a["operator"]: a
        for a in anchors
        if perf_score_against_anchor(a["geometric_speedup"], a["geometric_speedup"]) is not None
    }
    for anchor in anchors:
        anchor["scoreable"] = anchor["operator"] in valid_anchors
    for entry in entries:
        values = {}
        for op, anchor in valid_anchors.items():
            values[op] = perf_score_against_anchor(entry["op_geo"].get(op), anchor["geometric_speedup"])
        complete = len(entry["op_geo"]) == entry["total_operators"]
        if complete and values and all(v is not None for v in values.values()):
            entry["shared_perf_score"] = weighted_perf_score(values, weights)
            entry["scored_operators"] = len(values)
        entry.pop("op_geo")


def _rank_entries(entries, metric, excluded):
    field = "functional_score" if metric == "functional" else "shared_perf_score"
    eligible = []
    for entry in entries:
        if entry[field] is None:
            excluded.append(
                dict(
                    task_id=entry["task_id"],
                    reason="性能要求所选算子全部通过且测速齐全，并有可用共享基准",
                )
            )
        else:
            eligible.append(entry)
    # ZIP identity follows the submission API contract. A repeated upload
    # cannot occupy multiple ranks; ties use competition ranking (1,1,3).
    best = {}
    for entry in sorted(
        eligible, key=lambda e: (-round(e[field], 4), e["created_at"], e["task_id"])
    ):
        best.setdefault(entry["package_sha256"], entry)
    ranked = list(best.values())
    last = None
    rank = 0
    for index, entry in enumerate(ranked, 1):
        value = round(entry[field], 4)
        if value != last:
            rank = index
        entry["rank"] = rank
        last = value
    return ranked


def view(service, *, metric="performance", cohort=None, task_id=None):
    if metric not in ("functional", "performance"):
        raise ServiceError("LEADERBOARD-QUERY", "榜单类型须为 functional 或 performance")
    if cohort and task_id:
        raise ServiceError("LEADERBOARD-QUERY", "不能同时指定榜单范围与任务")
    with service.lock:
        tasks = _load_tasks(service)
        groups = _group_tasks(service, tasks, task_id)
        selected = _select_cohort(groups, cohort, task_id)
        entries = _cohort_entries(service, selected, groups.members, groups.excluded)
        anchors = _load_anchors(service, selected)
        _score_entries(service, entries, anchors)
        ranked = _rank_entries(entries, metric, groups.excluded)
        return dict(
            protocol_version=1,
            mode="trial",
            metric=metric,
            cohorts=list(groups.cohorts.values()),
            selected_cohort=groups.cohorts.get(selected),
            baseline_revision=digest(anchors) if anchors else None,
            anchors=anchors,
            items=ranked[:100],
            total_entries=len(ranked),
            excluded=groups.excluded,
            rules=[
                "评测集、判定资产、模式、算子范围和记录的环境指纹完全一致才同组",
                "共享算子基准持久冻结；不使用任务自身的100分排序",
                "同一ZIP保留最佳有效成绩；同分并列；显示前100条",
                "试运行成绩，不等同于正式覆盖认证或跨编译器排名",
            ],
        )
