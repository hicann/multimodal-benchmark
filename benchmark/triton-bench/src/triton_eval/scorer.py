# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""评分（L0，纯计算，无 NPU 依赖）。

规范约束：
  - 功能主得分 = 按算子权重加权的变体通过率 OpScore
  - 严格通过率 OpStrictPass 单独披露
  - 性能得分按算子锚点计算（见 anchors.py）：轮级分数 = 有分算子的
    加权平均，每个算子相对自己「首次全对轮次」的 100 分尺子
  - 不合成跨赛道总分（规范 §3.3 禁止两赛道隐式换算）
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Callable

from .bench.stats import geometric_mean
from .models import CaseResultRecord, CaseStatus, RunScoreRecord, is_counted, is_severe


def _counted(results: list[CaseResultRecord]) -> list[CaseResultRecord]:
    return [r for r in results if is_counted(r.status)]


def operator_pass_rate(
    results: list[CaseResultRecord],
) -> dict[str, tuple[float, float]]:
    """按算子聚合，返回 {算子名: (变体通过率 r_o, 严格通过率 q_o)}。"""
    buckets: dict[str, list[CaseResultRecord]] = {}
    for r in _counted(results):
        buckets.setdefault(r.operator, []).append(r)

    out: dict[str, tuple[float, float]] = {}
    for op, items in buckets.items():
        if not items:
            continue
        passed = sum(1 for r in items if r.status is CaseStatus.PASS)
        r_o = passed / len(items)
        q_o = 1.0 if passed == len(items) else 0.0
        out[op] = (r_o, q_o)
    return out


def coverage_scores(
    results: list[CaseResultRecord], weights: dict[str, float]
) -> tuple[float, float]:
    """返回 (OpScore, OpStrictPass)，均为百分制。

    分母只统计 counted 的用例；corpus 自身或评测方的 bug 不参与。
    """
    rates = operator_pass_rate(results)
    if not rates:
        return 0.0, 0.0

    total_w = 0.0
    acc_r = 0.0
    acc_q = 0.0
    for op, (r_o, q_o) in rates.items():
        w = weights.get(op, 1.0)
        total_w += w
        acc_r += w * r_o
        acc_q += w * q_o
    if total_w == 0:
        return 0.0, 0.0
    return 100.0 * acc_r / total_w, 100.0 * acc_q / total_w


def _perf_cases(results: list[CaseResultRecord]) -> list[CaseResultRecord]:
    """测到了性能的用例：计入分母、且拿到了加速比。

    fast 档达标率是「又对又快」的联合口径：分母含判错用例（错结果会
    稀释达标率），分子要求既正确又达标。
    """
    return [r for r in _counted(results) if r.speedup and r.speedup > 0]


def perf_aggregate(
    results: list[CaseResultRecord],
    thresholds: list[float],
    cv_gate: float,
) -> tuple[float | None, list[float], float]:
    """返回 (几何平均加速比, 各阈值达标占比, 不稳定比例)。

    P_geo 只统计**判对**用例的加速比——「算错但跑得快」的用例此前也进
    几何平均，等于给性能分充值（错得越快越沾光）。fast 达标率与不稳定
    比例仍以全部测到性能的用例为分母：前者是联合口径（判错稀释达标率），
    后者是测量质量指标（判错用例的计时不稳定同样是问题）。
    """
    measured = _perf_cases(results)
    if not measured:
        return None, [0.0] * len(thresholds), 0.0

    correct = [r for r in measured if r.status is CaseStatus.PASS]
    geo = geometric_mean([r.speedup for r in correct]) if correct else None

    fast = [
        sum(1 for r in measured
            if r.status is CaseStatus.PASS and r.speedup >= t) / len(measured)
        for t in thresholds
    ]

    with_cv = [r for r in measured if r.perf_cv is not None]
    unstable = (
        sum(1 for r in with_cv if r.perf_cv > cv_gate) / len(with_cv) if with_cv else 0.0
    )
    return geo, fast, unstable


# 锚点接近 1 时，ln(anchor) 趋近 0，对数归一化会数值爆炸。
# 实测：P_ref=1.05 时，P_geo=1.5 会算出 831 分；P_geo<1 还会出负分。
# 因此锚点必须显著偏离 1，否则该口径不可用。
MIN_ANCHOR_DEVIATION = 0.10


def perf_score_against_anchor(geo: float | None, anchor_geo: float | None) -> float | None:
    """相对锚点的性能得分。

    任一值为 None（本轮无性能数据 / 无锚点）→ None（不可用）。
    「首全对轮次 = 100 分」不需要特判：锚点取该轮自己的几何平均，
    ln(geo/anchor)=0，自然得 100。

    统一公式（0.2.2）：score = 100 + 100·ln(geo/anchor)/|ln(anchor)|。

    - 锚点 > 1 时与旧公式 100·ln(geo)/ln(anchor) 代数恒等（行为不变）；
    - 锚点 < 1 时修复**方向反转**：2026-10-12 真机首冻亚线性锚点
      （transpose 0.14x——demo 内核比官方框架慢约 7 倍）实测暴露，
      旧公式两个对数均为负，把内核优化快 2 倍（0.14x→0.28x）算成
      64 分、慢 2 倍算成 135 分——快慢颠倒。统一公式下：2 倍快
      ≈135 分、2 倍慢 ≈65 分、追平框架（1.0x）=200 分，单调正确。

    返回 None 表示「该口径不可用」，调用方必须原样透传而不是补 0——
    补 0 会把「无法计算」伪装成「性能极差」。
    """
    if geo is None or geo <= 0:
        return None
    if anchor_geo is None or anchor_geo <= 0:
        return None
    if abs(anchor_geo - 1.0) < MIN_ANCHOR_DEVIATION:
        # 锚点太靠近 1，ln(anchor)≈0，任何微小波动都会被放大成巨额分数
        return None
    score = 100.0 + 100.0 * math.log(geo / anchor_geo) / abs(math.log(anchor_geo))
    # 得分超过上限说明口径已失真（例如锚点与本轮的算子集合不同），
    # 与其报一个荒谬的 831 分，不如明确声明不可用
    if not (0.0 <= score <= 500.0):
        return None
    return score


def compute_run_score(
    run_id: int,
    results: list[CaseResultRecord],
    weights: dict[str, float],
    thresholds: list[float],
    cv_gate: float,
    scorer_version: str,
    op_scores: dict[str, float | None] | None = None,
    baseline_anchor_ref: str = "PER-OP",
) -> RunScoreRecord:
    """计算一次运行的分数。

    op_scores 是 anchors.score_per_operator 的输出（逐算子性能分）；
    轮级性能分 = 有分算子按权重加权平均，分数旁必须披露哪些算子没出分。
    传 None（如离线 rescore 拿不到锚点）→ 性能分记不可用，
    而不是编一个 100 充数。
    """
    # 函数内导入：anchors 依赖本模块的锚点护栏函数，顶层导入会成环
    from .anchors import weighted_perf_score

    ops_core, ops_strict = coverage_scores(results, weights)
    geo, fast, unstable = perf_aggregate(results, thresholds, cv_gate)
    score = weighted_perf_score(op_scores or {}, weights)

    return RunScoreRecord(
        run_id=run_id,
        coverage_ops_core=round(ops_core, 4),
        coverage_strict=round(ops_strict, 4),
        severe_count=sum(1 for r in results if is_severe(r.status)),
        perf_geo=None if geo is None else round(geo, 6),
        perf_score=None if score is None else round(score, 4),
        fast_1=None if not fast else round(fast[0], 4),
        fast_1_2=round(fast[1], 4) if len(fast) > 1 else None,
        fast_2=round(fast[2], 4) if len(fast) > 2 else None,
        unstable_rate=round(unstable, 4),
        baseline_anchor_ref=baseline_anchor_ref,
        scorer_version=scorer_version,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def mean_cv(results: list[CaseResultRecord]) -> float | None:
    cvs = [r.perf_cv for r in results if r.perf_cv is not None]
    return statistics.fmean(cvs) if cvs else None


# ---------------- 按 category / level 分组聚合 ----------------


def _group_weighted(
    results: list[CaseResultRecord],
    weights: dict[str, float],
    group_of: Callable[[CaseResultRecord], str | None],
) -> dict[str, float | None]:
    """按给定分组函数聚合加权通过率。

    无数据的组返回 None 而非 0：
    0 分会被误读为「该类别全部失败」，而实际是「无人提交 / 无结果」。
    """
    buckets: dict[str, list[CaseResultRecord]] = {}
    for r in _counted(results):
        g = group_of(r)
        if g is None:
            continue
        buckets.setdefault(g, []).append(r)

    out: dict[str, float | None] = {}
    for g, items in buckets.items():
        passed = sum(1 for r in items if r.status is CaseStatus.PASS)
        w = sum(weights.get(r.operator, 1.0) for r in items)
        out[g] = 100.0 * (passed / len(items)) if w else None
    return out


def coverage_by_category(
    results: list[CaseResultRecord],
    weights: dict[str, float],
    category_of: Callable[[str], str | None],
) -> dict[str, float | None]:
    """按 category 聚合功能通过率。"""
    return _group_weighted(
        results, weights, lambda r: category_of(r.operator)
    )


def coverage_by_level(
    results: list[CaseResultRecord],
    weights: dict[str, float],
    level_of: Callable[[str], str | None],
) -> dict[str, float | None]:
    """按 level 聚合功能通过率。"""
    return _group_weighted(results, weights, lambda r: level_of(r.operator))


def coverage_by_operator(results: list[CaseResultRecord]) -> dict[str, float | None]:
    """按算子聚合通过率。便于定位到具体算子。"""
    return _group_weighted(results, {}, lambda r: r.operator)
