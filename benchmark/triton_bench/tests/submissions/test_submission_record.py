# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""提交记录与覆盖率披露的测试（L0，无需 NPU）。

守护的是「上传算子闭环」的可追溯性：
    一轮评测必须能回答三个问题——测的是哪份提交、它合规吗、覆盖了几个算子。
    曾经只有 compiler_submission（编译器身份），提交方身份完全缺失：
    报告能说清编译器版本，却说不清这份结果属于谁提交的什么。
"""

from __future__ import annotations

import json

import pytest

from triton_eval.report import build_context
from triton_eval.store import Store

pytestmark = pytest.mark.torch


def _db(tmp_path) -> Store:
    s = Store(tmp_path / "t.db")
    s.create_schema()
    return s


def test_submission_record_roundtrip(tmp_path) -> None:
    s = _db(tmp_path)
    rid = s.insert_submission_record(
        submit_id="demo-001",
        root_path="/tmp/sub",
        operators=["Exp", "Softmax"],
        snapshot_fp="abc123",
        validation_ok=False,
        validation=[{"rule_id": "SUB-STR-001", "stage": "static",
                     "operator": "Softmax", "detail": "代算", "lineno": 11}],
        load_failures={"Gelu": "ImportError: ..."},
    )
    rec = s.submission_record(rid)
    assert rec is not None
    assert rec["submit_id"] == "demo-001"
    assert rec["install_mode"] == "source", "目录式提交没有打包环节"
    assert rec["validation_ok"] == 0
    assert json.loads(rec["operators_json"]) == ["Exp", "Softmax"]
    assert json.loads(rec["load_failures_json"]) == {"Gelu": "ImportError: ..."}
    s.close()


def test_each_run_gets_its_own_record(tmp_path) -> None:
    """不 upsert：同一目录在不同时间内容可能不同，折叠成一行会丢掉这个事实。"""
    s = _db(tmp_path)
    a = s.insert_submission_record("demo-001", "/tmp/sub", ["Exp"], "fp1", True, [])
    b = s.insert_submission_record("demo-001", "/tmp/sub", ["Exp"], "fp2", True, [])
    assert a != b
    s.close()


def test_run_links_submission_record(tmp_path) -> None:
    s = _db(tmp_path)
    cid = s.conn.execute(
        "INSERT INTO corpus_version (version, manifest_hash, case_count, frozen_at)"
        " VALUES ('v','h',0,'t')").lastrowid
    sid = s.upsert_submission("triton-ascend", "3.2.0", None, "source")
    rid = s.insert_submission_record("demo-001", "/tmp/sub", ["Exp"], "fp", True, [])
    run_id = s.create_run(cid, sid, "smoke", {}, "fp", "PASS",
                          submission_record_id=rid)
    run = s.get_run(run_id)
    assert run["submission_record_id"] == rid
    s.close()


def test_schema_migration_adds_column(tmp_path) -> None:
    """老库没有 submission_record_id，CREATE TABLE IF NOT EXISTS 不会补列。"""
    s = _db(tmp_path)
    cols = [r["name"] for r in s.conn.execute("PRAGMA table_info(eval_run)")]
    assert "submission_record_id" in cols
    # 重复建表必须幂等
    s.create_schema()
    cols = [r["name"] for r in s.conn.execute("PRAGMA table_info(eval_run)")]
    assert cols.count("submission_record_id") == 1
    s.close()


def _result(operator: str, status: str):
    """构造一个最小结果对象：status 用真实枚举，避免测试自己造出非法值。"""
    from triton_eval.models import CaseStatus, CaseResultRecord

    return CaseResultRecord(run_id=1, case_code=f"{operator}-x", operator=operator,
                            status=CaseStatus(status))


def test_coverage_discloses_partial_submission() -> None:
    """只交一部分算子也能满分——覆盖率必须与分数同时出现。"""
    results = [_result("Exp", "PASS"), _result("Exp", "PASS"),
               _result("Gelu", "SUBMISSION_INVALID")]
    ctx = build_context(
        run={"id": 1, "env_fp": "fp"}, env={}, submission={},
        results=results, score=None, corpus_version="v", spec_version="s",
        scorer_version="sc", run_dir="/r",
        submission_record={"submit_id": "demo", "root_path": "/s",
                           "install_mode": "source", "snapshot_fp": "fp",
                           "validation_ok": 1,
                           "operators_json": json.dumps(["Exp", "Gelu"])},
        total_operators=14,
    )
    cov = ctx["coverage"]
    assert cov["covered"] == ["Exp"], "未提交实现的算子不得计入覆盖"
    assert cov["covered_count"] == 1
    assert cov["total_operators"] == 14
    assert cov["missing"] == ["Gelu"], "声明了却没被评测的算子必须列出来"


def test_coverage_without_record_still_counts() -> None:
    ctx = build_context(
        run={"id": 1}, env={}, submission={}, results=[_result("Exp", "PASS")],
        score=None, corpus_version="v", spec_version="s", scorer_version="sc",
        run_dir="/r", total_operators=14,
    )
    assert ctx["coverage"]["covered"] == ["Exp"]
    assert ctx["submission_record"] is None
