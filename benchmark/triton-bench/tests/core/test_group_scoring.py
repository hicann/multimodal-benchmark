# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按 category / level 分组评分测试（L0，无 NPU 依赖）。"""

from __future__ import annotations

import pytest

from triton_eval.models import CaseResultRecord, CaseStatus, RunScoreRecord
from triton_eval.scorer import (
    coverage_by_category,
    coverage_by_level,
    coverage_by_operator,
)
from triton_eval.store import Store, utcnow

# 类别与等级映射（对应真实注册表）
CAT = {
    "Exp": "Elementwise", "Gelu": "Elementwise",
    "Cummin": "Reduction", "Maximum": "Broadcast",
    "Softmax": "Normalization", "RmsNorm": "Normalization",
}
LVL = {
    "Exp": "L1", "Gelu": "L1", "Cummin": "L2",
    "Maximum": "L2", "Softmax": "L2", "RmsNorm": "L2",
}


def _r(op: str, code: str, status: CaseStatus) -> CaseResultRecord:
    return CaseResultRecord(run_id=1, case_code=code, operator=op, status=status)


def _cat_of(name: str) -> str | None:
    return CAT.get(name)


def _lvl_of(name: str) -> str | None:
    return LVL.get(name)


def _store_with_run() -> Store:
    """建库并插入一条 run（run_score 有指向 eval_run 的外键）。"""
    from triton_eval.models import CaseSource, CaseSpec, CorpusManifest, OperatorSpec

    s = Store(":memory:")
    s.create_schema()
    manifest = CorpusManifest(
        version="v0.3-multi-input-frozen", spec_version="0.1-mvp",
        frozen_at=utcnow(),
        operators=[OperatorSpec(name="Exp", source=CaseSource.VENDOR_SELFTEST,
                                weight=1.0, ref_path="operators/exp/ref.py")],
        cases=[CaseSpec(code="exp-f32-00", operator="Exp",
                        source=CaseSource.VENDOR_SELFTEST, dtype="float32",
                        shape=[1024], seed=1)],
        manifest_hash="x",
    )
    corpus_id = s.import_corpus(manifest)
    sub_id = s.upsert_submission("triton-ascend", "3.2.0", None, "wheel")
    s.create_run(corpus_id, sub_id, "standard", {}, "fp", "PASS")
    return s


# ---------- 任务 8.1 ----------


def test_coverage_by_category() -> None:
    results = [
        _r("Exp", "e1", CaseStatus.PASS),
        _r("Exp", "e2", CaseStatus.COMPILE_FAIL),   # Elementwise 1/2
        _r("Cummin", "c1", CaseStatus.PASS),        # Reduction 1/1
    ]
    out = coverage_by_category(results, {}, _cat_of)
    assert out["Elementwise"] == pytest.approx(50.0)
    assert out["Reduction"] == pytest.approx(100.0)


def test_category_without_data_is_none_not_zero() -> None:
    """无数据的类别必须是 None，不能是 0——0 会被误读为全部失败。"""
    results = [_r("Exp", "e1", CaseStatus.PASS)]
    out = coverage_by_category(results, {}, _cat_of)
    assert "Broadcast" not in out or out.get("Broadcast") is None
    # 有结果的类别仍是正常数值
    assert out["Elementwise"] == pytest.approx(100.0)


def test_category_excludes_non_counted() -> None:
    """提交不合规 / 参考失效不计入分组分母。"""
    results = [
        _r("Exp", "e1", CaseStatus.PASS),
        _r("Exp", "e2", CaseStatus.SUBMISSION_INVALID),
        _r("Exp", "e3", CaseStatus.ORACLE_FAIL),
    ]
    out = coverage_by_category(results, {}, _cat_of)
    assert out["Elementwise"] == pytest.approx(100.0), "非计入项被排除后应为 100"


def test_unknown_operator_category_skipped() -> None:
    """无法映射类别的算子不应污染分组结果。"""
    results = [
        _r("Exp", "e1", CaseStatus.PASS),
        _r("UnknownOp", "u1", CaseStatus.COMPILE_FAIL),
    ]
    out = coverage_by_category(results, {}, _cat_of)
    assert out["Elementwise"] == pytest.approx(100.0)


# ---------- 任务 8.2 ----------


def test_coverage_by_level() -> None:
    results = [
        _r("Exp", "e1", CaseStatus.PASS),          # L1 通过
        _r("Gelu", "g1", CaseStatus.COMPILE_FAIL),  # L1 失败
        _r("Cummin", "c1", CaseStatus.PASS),        # L2 通过
    ]
    out = coverage_by_level(results, {}, _lvl_of)
    assert out["L1"] == pytest.approx(50.0)
    assert out["L2"] == pytest.approx(100.0)


def test_level_and_category_are_independent() -> None:
    """同一 category 可跨 level，同一 level 可含多 category。"""
    results = [
        _r("Exp", "e1", CaseStatus.PASS),      # L1 + Elementwise
        _r("Cummin", "c1", CaseStatus.PASS),   # L2 + Reduction
        _r("Softmax", "s1", CaseStatus.PASS),  # L2 + Normalization
    ]
    by_cat = coverage_by_category(results, {}, _cat_of)
    by_lvl = coverage_by_level(results, {}, _lvl_of)
    assert set(by_cat) == {"Elementwise", "Reduction", "Normalization"}
    assert set(by_lvl) == {"L1", "L2"}


# ---------- 任务 8.3：不计入项排除 ----------


def test_non_counted_excluded_from_both_dimensions() -> None:
    results = [
        _r("Exp", "e1", CaseStatus.PASS),
        _r("Exp", "e2", CaseStatus.SUBMISSION_INVALID),
        _r("Cummin", "c1", CaseStatus.PASS),
        _r("Cummin", "c2", CaseStatus.ORACLE_FAIL),
    ]
    by_cat = coverage_by_category(results, {}, _cat_of)
    by_lvl = coverage_by_level(results, {}, _lvl_of)
    assert by_cat["Elementwise"] == pytest.approx(100.0)
    assert by_cat["Reduction"] == pytest.approx(100.0)
    assert by_lvl["L1"] == pytest.approx(100.0)
    assert by_lvl["L2"] == pytest.approx(100.0)


# ---------- 任务 8.5：随 scorer_version 记录 ----------


def test_group_results_persisted_with_scorer_version() -> None:
    store = _store_with_run()
    score = RunScoreRecord(
        run_id=1, coverage_ops_core=80.0, coverage_strict=60.0, severe_count=0,
        baseline_anchor_ref="BASELINE", scorer_version="0.1.0", computed_at=utcnow(),
        by_category={"Elementwise": 50.0, "Reduction": None},
        by_level={"L1": 50.0},
    )
    store.insert_run_score(score)
    rows = store.run_scores(1)
    assert len(rows) == 1
    assert rows[0]["by_category_json"] is not None
    assert "Elementwise" in rows[0]["by_category_json"]
    assert rows[0]["by_level_json"] is not None
    store.close()


def test_rescore_keeps_grouping() -> None:
    store = _store_with_run()
    for ver in ("0.1.0", "0.2.0"):
        store.insert_run_score(RunScoreRecord(
            run_id=1, coverage_ops_core=80.0, coverage_strict=60.0, severe_count=0,
            baseline_anchor_ref="BASELINE", scorer_version=ver, computed_at=utcnow(),
            by_category={"Elementwise": 50.0},
        ))
    rows = store.run_scores(1)
    assert len(rows) == 2, "重算应新增记录而非覆盖"
    for r in rows:
        assert r["by_category_json"] is not None
    store.close()


# ---------- 任务 8.4 / 8.6：报告披露 ----------


def test_report_includes_group_sections() -> None:
    from triton_eval.report import build_context

    score = RunScoreRecord(
        run_id=1, coverage_ops_core=80.0, coverage_strict=60.0, severe_count=1,
        baseline_anchor_ref="BASELINE", scorer_version="0.1.0", computed_at=utcnow(),
        by_category={"Elementwise": 50.0, "Broadcast": None},
        by_level={"L1": 50.0},
    ).model_dump()
    ctx = build_context(
        run={"id": 1, "status": "OK", "profile": "standard", "env_fp": "x"},
        env={"clock_locked": False},
        submission={}, results=[], score=score,
        corpus_version="v0.2", spec_version="0.1", scorer_version="0.1.0",
        run_dir="/tmp/run-1",
    )
    from triton_eval.report import render_markdown

    md = render_markdown(ctx)
    assert "分类分级结果" in md
    assert "Elementwise" in md
    assert "无数据" in md, "无数据的类别必须显示为「无数据」而非 0"
    assert "不做加权合成或排名" in md


def test_no_cross_category_total() -> None:
    results = [
        _r("Exp", "e1", CaseStatus.PASS),
        _r("Cummin", "c1", CaseStatus.COMPILE_FAIL),
    ]
    by_cat = coverage_by_category(results, {}, _cat_of)
    # 分组结果只是各组的键值，不含任何合成字段
    assert "total" not in by_cat
    assert "overall" not in by_cat


def test_coverage_by_operator() -> None:
    results = [
        _r("Exp", "e1", CaseStatus.PASS),
        _r("Exp", "e2", CaseStatus.WRONG_RESULT),
        _r("Cummin", "c1", CaseStatus.PASS),
    ]
    out = coverage_by_operator(results)
    assert out["Exp"] == pytest.approx(50.0)
    assert out["Cummin"] == pytest.approx(100.0)
