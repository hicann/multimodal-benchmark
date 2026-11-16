# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""评分层单元测试（L0，无需 NPU）。"""

from __future__ import annotations

import math

import pytest

from triton_eval.models import CaseResultRecord, CaseStatus
from triton_eval.scorer import (
    compute_run_score,
    coverage_scores,
    operator_pass_rate,
    perf_aggregate,
    perf_score_against_anchor,
)


def _r(op: str, code: str, status: CaseStatus, speedup: float | None = None,
       cv: float | None = None) -> CaseResultRecord:
    return CaseResultRecord(
        run_id=1, case_code=code, operator=op, status=status,
        speedup=speedup, perf_cv=cv, perf_median_ms=(2.0 if speedup else None),
    )


# ---------- 功能得分（任务 7.1） ----------


def test_operator_pass_rate() -> None:
    results = [
        _r("add", "a1", CaseStatus.PASS),
        _r("add", "a2", CaseStatus.PASS),
        _r("add", "a3", CaseStatus.COMPILE_FAIL),
        _r("norm", "n1", CaseStatus.PASS),
        _r("norm", "n2", CaseStatus.PASS),
    ]
    rates = operator_pass_rate(results)
    assert rates["add"] == (2 / 3, 0.0), "add 有失败变体 → 严格通过率为 0"
    assert rates["norm"] == (1.0, 1.0)


def test_non_counted_results_excluded_from_coverage() -> None:
    """corpus / 评测方的 bug 不参与算分。"""
    results = [
        _r("add", "a1", CaseStatus.PASS),
        _r("add", "a2", CaseStatus.ORACLE_FAIL),
    ]
    rates = operator_pass_rate(results)
    assert rates["add"] == (1.0, 1.0), "ORACLE_FAIL 被排除在分母外"


def test_coverage_scores_weighted() -> None:
    results = [
        _r("add", "a1", CaseStatus.PASS),
        _r("add", "a2", CaseStatus.COMPILE_FAIL),
        _r("norm", "n1", CaseStatus.PASS),
        _r("norm", "n2", CaseStatus.PASS),
    ]
    # add: r=0.5 q=0 ; norm: r=1 q=1 ；等权
    core, strict = coverage_scores(results, {"add": 1.0, "norm": 1.0})
    assert core == pytest.approx(75.0)
    assert strict == pytest.approx(50.0)


def test_coverage_scores_respect_weights() -> None:
    results = [
        _r("add", "a1", CaseStatus.COMPILE_FAIL),
        _r("norm", "n1", CaseStatus.PASS),
    ]
    core, _ = coverage_scores(results, {"add": 0.1, "norm": 0.9})
    assert core == pytest.approx(90.0)


def test_coverage_scores_empty() -> None:
    assert coverage_scores([], {}) == (0.0, 0.0)


# ---------- 性能聚合（任务 7.2） ----------


def test_perf_geometric_mean() -> None:
    results = [
        _r("add", "a1", CaseStatus.PASS, speedup=2.0, cv=0.01),
        _r("add", "a2", CaseStatus.PASS, speedup=8.0, cv=0.01),
    ]
    geo, fast, unstable = perf_aggregate(results, [1.0, 1.2, 2.0], cv_gate=0.03)
    assert geo == pytest.approx(4.0), "几何平均 sqrt(2*8)=4"
    assert fast == [1.0, 1.0, 1.0]
    assert math.isclose(unstable, 0.0, rel_tol=0.0, abs_tol=0.0)


def test_fast_ratios_require_correctness() -> None:
    """fast_p 是「既正确又达标」的联合口径。"""
    results = [
        _r("a", "a1", CaseStatus.PASS, speedup=3.0, cv=0.01),
        _r("a", "a2", CaseStatus.WRONG_RESULT, speedup=5.0, cv=0.01),
    ]
    _, fast, _ = perf_aggregate(results, [1.0, 2.0], cv_gate=0.03)
    assert fast == [0.5, 0.5], "结果错误的用例不计入达标"


def test_unstable_rate_counts_cv_violations() -> None:
    results = [
        _r("a", "a1", CaseStatus.PASS, speedup=2.0, cv=0.01),
        _r("a", "a2", CaseStatus.PASS, speedup=2.0, cv=0.50),
    ]
    _, _, unstable = perf_aggregate(results, [1.0], cv_gate=0.03)
    assert unstable == pytest.approx(0.5)


def test_perf_aggregate_no_data() -> None:
    geo, fast, unstable = perf_aggregate([_r("a", "a1", CaseStatus.PASS)], [1.0], 0.03)
    assert geo is None
    assert fast == [0.0]
    assert math.isclose(unstable, 0.0, rel_tol=0.0, abs_tol=0.0)


# ---------- 锚点（任务 7.3；按算子口径） ----------


def test_self_anchor_scores_100() -> None:
    """首全对轮自锚：ln(x)/ln(x) = 1，自然得 100 分，无需任何特判。

    取代旧「首轮锚点 None → 100」的写死逻辑：按算子锚点下，
    首个全对轮的锚点值就是它自己的几何平均。
    """
    assert perf_score_against_anchor(2.0, 2.0) == pytest.approx(100.0)
    assert perf_score_against_anchor(0.3, 0.3) == pytest.approx(100.0)


def test_sublinear_anchor_direction_not_inverted() -> None:
    """亚线性锚点（<1）的得分方向必须正确（0.2.2 修复的回归钉）。

    2026-10-12 真机首冻亚线性锚点实测暴露：demo transpose 内核比官方
    框架慢约 7 倍（锚点 0.14x）。旧公式 100·ln(geo)/ln(anchor) 在
    anchor<1 时两个对数均为负——把内核优化快 2 倍（0.14→0.28x）算成
    64 分、慢 2 倍（0.07x）算成 135 分，快慢颠倒。统一公式
    100 + 100·ln(geo/anchor)/|ln(anchor)| 下方向正确，且锚点>1 时
    与旧公式代数恒等（上面的既有断言不改即过）。
    """
    anchor = 0.14  # 真机实测锚点（2026-10-12，transpose）
    # 自锚 100
    assert perf_score_against_anchor(anchor, anchor) == pytest.approx(100.0)
    # 快 2 倍 → >100；慢 2 倍 → <100（旧公式正好相反）
    assert perf_score_against_anchor(0.28, anchor) == pytest.approx(135.2, abs=0.2)
    assert perf_score_against_anchor(0.07, anchor) == pytest.approx(64.8, abs=0.2)
    # 追平官方框架（1.0x）= 200：从锚点 deficit 到框架 parity 的整段路
    assert perf_score_against_anchor(1.0, anchor) == pytest.approx(200.0)
    # 单调性：越快分越高
    assert perf_score_against_anchor(0.5, anchor) > perf_score_against_anchor(0.28, anchor) \
        > perf_score_against_anchor(anchor, anchor) > perf_score_against_anchor(0.07, anchor)
    # 远慢于锚点（10 倍）→ 负分 → 数值护栏判不可用，不编数
    assert perf_score_against_anchor(0.014, anchor) is None


def test_later_run_relative_to_anchor() -> None:
    # 锚点 4.0，本轮 2.0 → ln2/ln4 = 0.5 → 50 分
    assert perf_score_against_anchor(2.0, 4.0) == pytest.approx(50.0)


def test_no_perf_data_gives_no_score() -> None:
    assert perf_score_against_anchor(None, 2.0) is None
    assert perf_score_against_anchor(2.0, None) is None, "无锚点 → 不可用，不编数"


def test_perf_geo_excludes_wrong_results() -> None:
    """P_geo 只统计判对用例：「错得快」不给性能分充值。

    此前 WRONG_RESULT 用例的加速比也进几何平均——实现返回错误结果
    但跑得越快，性能分越高，是真缺陷。
    """
    results = [
        _r("a", "a1", CaseStatus.PASS, speedup=2.0, cv=0.01),
        _r("a", "a2", CaseStatus.WRONG_RESULT, speedup=8.0, cv=0.01),
    ]
    geo, fast, _ = perf_aggregate(results, [1.0], cv_gate=0.03)
    assert geo == pytest.approx(2.0), "8 倍加速比的错误用例不得进几何平均"
    assert fast == [0.5], "fast 达标率分母仍含判错用例（又对又快的联合口径）"


def test_anchor_near_one_makes_score_unusable() -> None:
    """锚点接近 1 时对数归一化会数值爆炸，必须声明不可用而非报错。

    实测：P_ref=1.05 且 P_geo=1.5 会算出 831 分，P_geo<1 还会出负分。
    """
    assert perf_score_against_anchor(1.5, 1.05) is None, "锚点太靠近 1，口径不可用"
    assert perf_score_against_anchor(0.9, 1.02) is None
    # 显著偏离 1 时正常可用
    assert perf_score_against_anchor(4.0, 2.0) == pytest.approx(200.0)


def test_score_out_of_range_is_unusable() -> None:
    """失真到荒谬的分数必须声明不可用，不能原样报出。"""
    assert perf_score_against_anchor(50.0, 1.2) is None


def test_unusable_is_none_not_zero() -> None:
    """「无法计算」必须是 None，不能补 0 —— 补 0 会伪装成性能极差。"""
    for geo, anchor in [(None, None), (1.5, 1.05), (0.0, 2.0)]:
        assert perf_score_against_anchor(geo, anchor) is None


# ---------- 重算（任务 7.4） ----------


def test_compute_run_score_fields() -> None:
    results = [
        _r("add", "a1", CaseStatus.PASS, speedup=2.0, cv=0.01),
        _r("add", "a2", CaseStatus.WRONG_RESULT, speedup=2.0, cv=0.01),
        _r("add", "a3", CaseStatus.ORACLE_FAIL),
    ]
    score = compute_run_score(
        run_id=1, results=results, weights={"add": 1.0},
        thresholds=[1.0, 1.2, 2.0], cv_gate=0.03, scorer_version="0.2.0",
    )
    assert score.severe_count == 1
    assert score.baseline_anchor_ref == "PER-OP"
    assert score.perf_score is None, "未传算子分数（离线重算无锚点）→ 不可用，不编 100"
    assert score.perf_geo == pytest.approx(2.0), "P_geo 只统计判对用例"
    assert score.scorer_version == "0.2.0"
    assert not hasattr(score, "total"), "不得产出跨赛道总分"


def test_compute_run_score_aggregates_op_scores() -> None:
    """轮级性能分 = 有分算子的加权平均；无分算子不进分母。"""
    score = compute_run_score(
        run_id=1, results=[], weights={"add": 1.0, "norm": 1.0, "mul": 3.0},
        thresholds=[1.0], cv_gate=0.03, scorer_version="0.2.0",
        op_scores={"add": 100.0, "norm": 50.0, "mul": None},
    )
    # mul 无分不进分母：(1×100 + 1×50) / 2 = 75
    assert score.perf_score == pytest.approx(75.0)


def test_compute_run_score_all_ops_unscored_is_none() -> None:
    score = compute_run_score(
        run_id=1, results=[], weights={"add": 1.0},
        thresholds=[1.0], cv_gate=0.03, scorer_version="0.2.0",
        op_scores={"add": None},
    )
    assert score.perf_score is None, "全部算子无分 → 不可用，不编数"


def test_rescore_with_new_version_is_a_new_record() -> None:
    results = [_r("add", "a1", CaseStatus.PASS, speedup=2.0, cv=0.01)]
    s1 = compute_run_score(1, results, {"add": 1.0}, [1.0], 0.03, "0.1.0")
    s2 = compute_run_score(1, results, {"add": 1.0}, [1.0], 0.03, "0.2.0")
    assert s1.scorer_version != s2.scorer_version
    assert s1.computed_at and s2.computed_at
