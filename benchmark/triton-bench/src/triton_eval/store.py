# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""结果存储（L0，SQLite）。

设计要点：
  - 9 张表，主键全部自增 INTEGER，业务唯一性另设 UNIQUE 约束
  - 表名用 eval_case / eval_run 而非 case / run：`CASE` 是 SQL 保留字，
    直接用作表名在 SQLite 与 PG 都需要加引号，改用非保留名最省事
  - results.jsonl 是真相源，本模块是查询加速层：数据库可从 jsonl 完整重建
  - 不用 SQLite 专有类型，便于将来迁 PostgreSQL
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import CaseResultRecord, CaseSpec, CorpusManifest, OperatorSpec, RunScoreRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_version (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    version        TEXT    NOT NULL UNIQUE,
    manifest_hash  TEXT    NOT NULL,
    case_count     INTEGER NOT NULL,
    frozen_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS operator (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus_version_id INTEGER NOT NULL REFERENCES corpus_version(id),
    name              TEXT    NOT NULL,
    source            TEXT    NOT NULL,
    weight            REAL    NOT NULL,
    ref_path          TEXT    NOT NULL,
    UNIQUE (corpus_version_id, name)
);

-- 不再存 kernel_path / entry：实现由提交提供，用例只描述数据。
-- 这是支持用户上传的前提，也在 schema 层面杜绝「用例自带实现」。
CREATE TABLE IF NOT EXISTS eval_case (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_id      INTEGER NOT NULL REFERENCES operator(id),
    code             TEXT    NOT NULL UNIQUE,
    dtype            TEXT    NOT NULL,
    shape_json       TEXT    NOT NULL,
    seed             INTEGER NOT NULL,
    launch_json      TEXT    NOT NULL,
    timeout_s        INTEGER NOT NULL DEFAULT 60,
    feature_atoms_json TEXT  NOT NULL,
    content_hash     TEXT    NOT NULL
);

-- 列名用 git_commit：`COMMIT` 是 SQL 保留字，直接作列名会语法错误
CREATE TABLE IF NOT EXISTS compiler_submission (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    version      TEXT    NOT NULL,
    git_commit   TEXT,
    install_mode TEXT    NOT NULL,
    UNIQUE (name, version, git_commit)
);

CREATE TABLE IF NOT EXISTS eval_run (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus_version_id INTEGER NOT NULL REFERENCES corpus_version(id),
    submission_id     INTEGER NOT NULL REFERENCES compiler_submission(id),
    profile           TEXT    NOT NULL,
    started_at        TEXT    NOT NULL,
    finished_at       TEXT,
    status            TEXT    NOT NULL,
    invalid_reason    TEXT,
    env_json          TEXT    NOT NULL,
    env_fp            TEXT    NOT NULL,
    probe_status      TEXT    NOT NULL,
    run_dir           TEXT
);

CREATE TABLE IF NOT EXISTS case_result (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES eval_run(id),
    case_id        INTEGER NOT NULL REFERENCES eval_case(id),
    status         TEXT    NOT NULL,
    duration_ms    REAL,
    compile_err    TEXT,
    max_rel_err    REAL,
    max_abs_err    REAL,
    samples_ms_json TEXT,
    perf_median_ms REAL,
    perf_cv        REAL,
    baseline_median_ms REAL,
    speedup        REAL,
    log_path       TEXT,
    UNIQUE (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS run_score (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER NOT NULL REFERENCES eval_run(id),
    coverage_ops_core  REAL    NOT NULL,
    coverage_strict    REAL    NOT NULL,
    severe_count       INTEGER NOT NULL,
    perf_geo           REAL,
    perf_score         REAL,
    fast_1             REAL,
    fast_1_2           REAL,
    fast_2             REAL,
    unstable_rate      REAL    NOT NULL DEFAULT 0,
    baseline_anchor_ref TEXT   NOT NULL,
    scorer_version     TEXT    NOT NULL,
    computed_at        TEXT    NOT NULL,
    -- 按 category / level 的分组结果，JSON。随 scorer_version 一并留痕。
    by_category_json   TEXT,
    by_level_json      TEXT,
    UNIQUE (run_id, scorer_version)
);
"""

# 提交记录表：目录式提交下，「这份提交是谁、在哪、合规吗」必须可追溯。
#
# 曾经只有 compiler_submission（编译器身份），提交方身份完全缺失：
# 报告能说清测的是哪个编译器，却说不清测的是哪份提交、它有没有违规。
# 上传算子闭环里，提交是一等公民，不能隐形。
SUBMISSION_RECORD_DDL = """
CREATE TABLE IF NOT EXISTS submission_record (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    submit_id        TEXT    NOT NULL,
    root_path        TEXT    NOT NULL,
    install_mode     TEXT    NOT NULL DEFAULT 'source',
    operators_json   TEXT    NOT NULL,
    snapshot_fp      TEXT    NOT NULL,
    validation_ok    INTEGER NOT NULL,
    validation_json  TEXT    NOT NULL,
    load_failures_json TEXT  NOT NULL DEFAULT '{}'
);
"""

# 分组聚合结果表：随 scorer_version 记录，支持重算留痕
GROUP_SCORE_DDL = """
CREATE TABLE IF NOT EXISTS group_score (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES eval_run(id),
    dimension      TEXT    NOT NULL,   -- category / level
    group_key      TEXT    NOT NULL,
    pass_rate      REAL,                -- NULL 表示无数据
    scorer_version TEXT    NOT NULL,
    UNIQUE (run_id, dimension, group_key, scorer_version)
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """SQLite 存储。所有方法都不依赖 NPU。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    # ---------- 建表 ----------

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- corpus ----------

    def create_schema(self) -> None:
        # 分两次执行：executescript 会隐式提交当前事务，
        # 连续调用两次会在某些驱动下触发 "cannot start a transaction
        # within a transaction"。
        self.conn.executescript(SCHEMA)
        self.conn.executescript(SUBMISSION_RECORD_DDL)
        self.conn.executescript(GROUP_SCORE_DDL)
        # 老库没有这两列，CREATE TABLE IF NOT EXISTS 不会补列，故显式迁移
        self._ensure_column("eval_run", "submission_record_id", "INTEGER")
        # zip 上传闭环：提交的身份链要能从「包」追溯到「目录」——
        # 包名 + 包指纹入库，同一份包无论解包几次、谁解包，可归并
        self._ensure_column("submission_record", "package_sha256", "TEXT")
        self._ensure_column("submission_record", "package_name", "TEXT")
        # schema 不兼容要在写任何数据之前报出，而不是 import_corpus 时
        # 抛 IntegrityError。_ensure_column 只会加列，删不掉旧表的
        # NOT NULL 列——旧库没法原地迁移，只能重建（jsonl 是真相源）。
        self._reject_incompatible_db()
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def insert_corpus(self, manifest: CorpusManifest) -> int:
        """写入 corpus 版本，已存在则更新而非替换。

        不能用 `INSERT OR REPLACE`：REPLACE 的语义是删除旧行再插入新行，
        而 operator / eval_case 都有指向 corpus_version 的外键，
        删除父行会触发 `FOREIGN KEY constraint failed`。
        重复导入同一版本（例如重新生成清单后再跑一次）必须保留原 id。
        """
        row = self.conn.execute(
            "SELECT id FROM corpus_version WHERE version = ?", (manifest.version,)
        ).fetchone()
        if row is not None:
            self.conn.execute(
                "UPDATE corpus_version SET manifest_hash=?, case_count=?, frozen_at=? "
                "WHERE id=?",
                (manifest.manifest_hash, len(manifest.cases),
                 manifest.frozen_at, row["id"]),
            )
            self.conn.commit()
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO corpus_version (version, manifest_hash, case_count, frozen_at) "
            "VALUES (?,?,?,?)",
            (manifest.version, manifest.manifest_hash,
             len(manifest.cases), manifest.frozen_at),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def insert_operator(self, corpus_version_id: int, op: OperatorSpec) -> int:
        """同上：已存在则 UPDATE，避免 REPLACE 破坏 eval_case 的外键。"""
        row = self.conn.execute(
            "SELECT id FROM operator WHERE corpus_version_id=? AND name=?",
            (corpus_version_id, op.name),
        ).fetchone()
        if row is not None:
            self.conn.execute(
                "UPDATE operator SET source=?, weight=?, ref_path=? WHERE id=?",
                (op.source.value, op.weight, op.ref_path, row["id"]),
            )
            self.conn.commit()
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO operator (corpus_version_id, name, source, weight, ref_path) "
            "VALUES (?,?,?,?,?)",
            (corpus_version_id, op.name, op.source.value, op.weight, op.ref_path),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def insert_case(self, operator_id: int, case: CaseSpec) -> int:
        """同上：已存在则 UPDATE，避免 REPLACE 破坏 case_result 的外键。"""
        row = self.conn.execute(
            "SELECT id FROM eval_case WHERE code = ?", (case.code,)
        ).fetchone()
        if row is not None:
            self.conn.execute(
                "UPDATE eval_case SET operator_id=?, dtype=?, "
                "shape_json=?, seed=?, launch_json=?, timeout_s=?, "
                "feature_atoms_json=?, content_hash=? WHERE id=?",
                (
                    operator_id, case.dtype,
                    json.dumps(case.shape), case.seed, json.dumps(case.launch),
                    case.timeout_s, json.dumps(case.feature_atoms),
                    case.content_hash, row["id"],
                ),
            )
            self.conn.commit()
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO eval_case "
            "(operator_id, code, dtype, shape_json, seed, "
            " launch_json, timeout_s, feature_atoms_json, content_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                operator_id, case.code, case.dtype,
                json.dumps(case.shape), case.seed, json.dumps(case.launch),
                case.timeout_s, json.dumps(case.feature_atoms), case.content_hash,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def import_corpus(self, manifest: CorpusManifest) -> int:
        """把冻结清单整体写入库，返回 corpus_version_id。"""
        corpus_id = self.insert_corpus(manifest)
        op_ids = {
            op.name: self.insert_operator(corpus_id, op) for op in manifest.operators
        }
        for case in manifest.cases:
            if case.operator not in op_ids:
                raise KeyError(f"用例 {case.code} 引用了未定义的算子 {case.operator}")
            self.insert_case(op_ids[case.operator], case)
        return corpus_id

    def case_id_by_code(self, code: str) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM eval_case WHERE code = ?", (code,)
        ).fetchone()
        return int(row["id"]) if row else None

    # ---------- submission ----------

    def insert_submission_record(
        self,
        submit_id: str,
        root_path: str,
        operators: list[str],
        snapshot_fp: str,
        validation_ok: bool,
        validation: list[dict[str, Any]],
        load_failures: dict[str, str] | None = None,
        install_mode: str = "source",
        package_sha256: str | None = None,
        package_name: str | None = None,
    ) -> int:
        """记录一份提交及其校验结论。每次评测都新插一行。

        不 upsert：同一份目录在不同时间内容可能不同（snapshot_fp 会变），
        把它们折叠成一行就丢掉了「提交内容本身也是变量」这个事实。
        package_sha256 / package_name：来自 zip 上传时填写——身份链
        「包 → 目录 → 快照指纹」的起点。
        """
        cur = self.conn.execute(
            "INSERT INTO submission_record (submit_id, root_path, install_mode,"
            " operators_json, snapshot_fp, validation_ok, validation_json,"
            " load_failures_json, package_sha256, package_name)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                submit_id, root_path, install_mode,
                json.dumps(operators), snapshot_fp,
                1 if validation_ok else 0, json.dumps(validation),
                json.dumps(load_failures or {}),
                package_sha256, package_name,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def submission_record(self, record_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM submission_record WHERE id=?", (record_id,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_submission(
        self, name: str, version: str, commit: str | None, install_mode: str
    ) -> int:
        self.conn.execute(
            "INSERT OR IGNORE INTO compiler_submission "
            "(name, version, git_commit, install_mode) VALUES (?,?,?,?)",
            (name, version, commit, install_mode),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM compiler_submission WHERE name=? AND version=? "
            "AND git_commit IS ?",
            (name, version, commit),
        ).fetchone()
        return int(row["id"])

    # ---------- run ----------

    def create_run(
        self,
        corpus_version_id: int,
        submission_id: int,
        profile: str,
        env_json: dict[str, Any],
        env_fp: str,
        probe_status: str,
        run_dir: str | None = None,
        submission_record_id: int | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO eval_run (corpus_version_id, submission_id, profile, "
            "started_at, status, env_json, env_fp, probe_status, run_dir, "
            "submission_record_id) "
            "VALUES (?,?,?,?,'RUNNING',?,?,?,?,?)",
            (corpus_version_id, submission_id, profile, utcnow(),
             json.dumps(env_json, sort_keys=True), env_fp, probe_status, run_dir,
             submission_record_id),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def finish_run(
        self, run_id: int, status: str, invalid_reason: str | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE eval_run SET finished_at=?, status=?, invalid_reason=? WHERE id=?",
            (utcnow(), status, invalid_reason, run_id),
        )
        self.conn.commit()

    def set_run_probe(self, run_id: int, probe_status: str) -> None:
        """补记健康探针结论。run 行先建（probe=PENDING），子进程探针事件到达后回填。"""
        self.conn.execute(
            "UPDATE eval_run SET probe_status=? WHERE id=?", (probe_status, run_id)
        )
        self.conn.commit()

    def set_run_submission_record(self, run_id: int, record_id: int | None) -> None:
        """把 run 行关联到提交记录。记录要等子进程的校验事件才能插入，
        而 run 行必须先建（子进程挂掉也要留痕），所以只能事后回填。
        """
        self.conn.execute(
            "UPDATE eval_run SET submission_record_id=? WHERE id=?",
            (record_id, run_id),
        )
        self.conn.commit()

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM eval_run WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # ---------- case_result ----------

    def insert_case_result(self, rec: CaseResultRecord) -> int:
        """写入一条用例结果。同一 run 的同一 case 只能有一条，重复写入会报错。"""
        case_id = self.case_id_by_code(rec.case_code)
        if case_id is None:
            raise KeyError(f"未登记的用例 {rec.case_code}；请先 import_corpus")
        cur = self.conn.execute(
            "INSERT INTO case_result (run_id, case_id, status, duration_ms, "
            "compile_err, max_rel_err, max_abs_err, samples_ms_json, perf_median_ms, "
            "perf_cv, baseline_median_ms, speedup, log_path) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec.run_id, case_id, rec.status.value, rec.duration_ms, rec.compile_err,
                rec.max_rel_err, rec.max_abs_err, json.dumps(rec.samples_ms),
                rec.perf_median_ms, rec.perf_cv, rec.baseline_median_ms,
                rec.speedup, rec.log_path,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def case_results(self, run_id: int) -> list[CaseResultRecord]:
        rows = self.conn.execute(
            "SELECT r.*, c.code AS case_code, o.name AS operator "
            "FROM case_result r "
            "JOIN eval_case c ON c.id = r.case_id "
            "JOIN operator o ON o.id = c.operator_id "
            "WHERE r.run_id = ? ORDER BY r.id",
            (run_id,),
        ).fetchall()
        out: list[CaseResultRecord] = []
        for row in rows:
            out.append(
                CaseResultRecord(
                    run_id=row["run_id"],
                    case_code=row["case_code"],
                    operator=row["operator"],
                    status=row["status"],
                    duration_ms=row["duration_ms"],
                    compile_err=row["compile_err"],
                    max_rel_err=row["max_rel_err"],
                    max_abs_err=row["max_abs_err"],
                    samples_ms=json.loads(row["samples_ms_json"] or "[]"),
                    perf_median_ms=row["perf_median_ms"],
                    perf_cv=row["perf_cv"],
                    baseline_median_ms=row["baseline_median_ms"],
                    speedup=row["speedup"],
                    log_path=row["log_path"],
                )
            )
        return out

    # ---------- run_score ----------

    def insert_run_score(self, score: RunScoreRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO run_score (run_id, coverage_ops_core, coverage_strict, "
            "severe_count, perf_geo, perf_score, fast_1, fast_1_2, fast_2, "
            "unstable_rate, baseline_anchor_ref, scorer_version, computed_at, "
            "by_category_json, by_level_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                score.run_id, score.coverage_ops_core, score.coverage_strict,
                score.severe_count, score.perf_geo, score.perf_score,
                score.fast_1, score.fast_1_2, score.fast_2, score.unstable_rate,
                score.baseline_anchor_ref, score.scorer_version, score.computed_at,
                json.dumps(getattr(score, "by_category", None)),
                json.dumps(getattr(score, "by_level", None)),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def run_scores(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM run_score WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 从 jsonl 重建（真相源 → 数据库） ----------

    def rebuild_from_jsonl(self, run_id: int, jsonl_path: str | Path) -> int:
        """从 results.jsonl 重建该 run 的用例结果。

        数据库是可丢弃的查询加速层；只要 jsonl 还在，记录就能完整恢复。
        """
        path = Path(jsonl_path)
        self.conn.execute("DELETE FROM case_result WHERE run_id=?", (run_id,))
        self.conn.commit()
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = CaseResultRecord.model_validate_json(line)
            rec.run_id = run_id
            self.insert_case_result(rec)
            count += 1
        return count

    def _reject_incompatible_db(self) -> None:
        """旧版 schema（eval_case.kernel_path 已废弃）的库必须拒用。

        曾经的表现：建表静默通过（IF NOT EXISTS 不动旧表），
        import_corpus 插到一半才抛
        `IntegrityError: NOT NULL constraint failed: eval_case.kernel_path`，
        看起来像数据坏了，实际是库旧了。
        """
        cols = [r["name"] for r in self.conn.execute("PRAGMA table_info(eval_case)")]
        if "kernel_path" in cols:
            raise RuntimeError(
                "数据库是旧版 schema（eval_case.kernel_path 已废弃，"
                "用例不再携带实现路径）。数据库只是查询加速层："
                "删除该 .db 文件后重跑即可，历史结果可从各 run 目录的 "
                "results.jsonl 用 `import` 子命令重建；或换一个 --db 路径。"
            )

    def _ensure_column(self, table: str, column: str, decl: str) -> None:
        """幂等地补列。SQLite 没有 ADD COLUMN IF NOT EXISTS。"""
        cols = [r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")]
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
