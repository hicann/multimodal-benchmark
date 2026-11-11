# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按算子的性能锚点（100 分尺子）。

设计要点（相对旧「轮级首轮锚点」的变更）：

  旧的轮级锚点把「性能尺子是否存在」耦合到「有一轮全部计入用例都判对」上：
  题库里只要有一道坏题（无论谁交都过不了），任何一轮都不干净，锚点永远
  冻结不了，性能赛道整体死锁。按算子锚点把死锁半径缩小到单个算子——
  每个算子有自己的尺子，坏题只废掉它所在的算子，其余算子照常出分，
  且坏算子会被大声暴露（永远出不了性能分）而不是悄悄污染所有分数。

  规则：
  1. 某算子第一次「题库内该算子**全部**用例都 PASS 且都有加速比」的轮次，
     冻结为该算子的锚点：该轮该算子全部用例加速比的几何平均 = 100 分。
     --limit 跑过的轮次因用例不全而天然不满足；冒烟档不测性能、
     同样天然不满足。资格由数据本身说话，不依赖运行状态字段。
  2. 之后每轮逐算子出分：该算子本轮全对才比（判错用例的加速比不进
     性能统计——「错得快」不该给性能分充值）；没全对 → 该算子本轮
     无性能分，不影响其他算子。
  3. 尺子毕竟是过去某轮的测量，环境升级后可能变旧——但这只影响
     100 分制口径；逐用例加速比是每轮同会话现场实测的裸值，不受锚点
     变旧影响（两轮对比命令直接比裸值）。

  本模块只依赖 Store 与纯计算（无 NPU），可在 macOS 上完整测试。
"""

from __future__ import annotations

from collections import defaultdict

from .bench.stats import geometric_mean
from .models import CaseResultRecord, CaseStatus, is_counted
from .scorer import MIN_ANCHOR_DEVIATION, perf_score_against_anchor
from .store import Store


def op_corpus_cases(s: Store, corpus_version_id: int) -> dict[str, set[str]]:
    """题库中每个算子的全部用例编号。

    「全对」的判定基准：锚点/出分都要求该算子**每一道题库用例**都有
    PASS 结果——跑了一半（--limit）或漏题的轮次不算全对。
    """
    rows = s.conn.execute(
        "SELECT o.name AS op, c.code AS code FROM eval_case c "
        "JOIN operator o ON o.id = c.operator_id "
        "WHERE o.corpus_version_id = ?",
        (corpus_version_id,),
    ).fetchall()
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        out[row["op"]].add(row["code"])
    return dict(out)


def resolve_op_anchors(
    s: Store, corpus_version_id: int
) -> dict[str, tuple[int, float]]:
    """解析按算子的锚点：{算子名: (锚点轮次编号, 锚点几何平均加速比)}。

    每个算子取「该算子全部题库用例都 PASS 且都有加速比」的**最早**轮次。
    锚点跨题库版本不通用（题都不一样），必须按 corpus_version_id 圈定。
    """
    op_cases = op_corpus_cases(s, corpus_version_id)
    run_ids = _corpus_run_ids(s, corpus_version_id)
    anchors: dict[str, tuple[int, float]] = {}
    for run_id in run_ids:
        pending = {op for op in op_cases if op not in anchors}
        if not pending:
            break
        results = s.case_results(run_id)
        if not results:
            continue
        by_op: dict[str, list[CaseResultRecord]] = defaultdict(list)
        for r in results:
            by_op[r.operator].append(r)
        for op in pending:
            cases_needed = op_cases[op]
            if not cases_needed:
                continue
            pass_rows = [r for r in by_op.get(op, []) if r.status is CaseStatus.PASS]
            # 全对：PASS 的用例集合必须恰好等于题库用例集合
            if {r.case_code for r in pass_rows} != cases_needed:
                continue
            # 尺子必须量过该算子的每一道题（缺性能数据的轮次不配当尺子）
            speedups = [r.speedup for r in pass_rows if r.speedup and r.speedup > 0]
            if len(speedups) != len(cases_needed):
                continue
            anchors[op] = (run_id, geometric_mean(speedups))
    return anchors


def score_per_operator(
    results: list[CaseResultRecord],
    op_anchors: dict[str, tuple[int, float]],
    op_cases: dict[str, set[str]],
    weights: dict[str, float] | None = None,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """逐算子计算性能分。

    返回 ({算子: 分数或 None}, {算子: 一句话说明})。
    None 的三种原因都写进说明：本轮未全对 / 本轮无性能数据 / 无锚点
    （或锚点不过数值护栏）——「为什么没分」必须可读，不能只留空。
    """
    by_op: dict[str, list[CaseResultRecord]] = defaultdict(list)
    for r in results:
        by_op[r.operator].append(r)

    scores: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    for op in sorted(op_cases):
        rows = by_op.get(op, [])
        if not rows:
            scores[op] = None
            reasons[op] = "本轮无该算子结果（未提交或未跑到）"
            continue
        cases_needed = op_cases[op]
        pass_rows = [r for r in rows if r.status is CaseStatus.PASS]
        if not cases_needed or {r.case_code for r in pass_rows} != cases_needed:
            failed = sum(1 for r in rows if is_counted(r.status) and r.status is not CaseStatus.PASS)
            missing = max(len(cases_needed) - len(rows), 0)
            scores[op] = None
            reasons[op] = f"本轮未全对（{failed} 个判错，{max(missing, 0)} 个无结果）"
            continue
        speedups = [r.speedup for r in pass_rows if r.speedup and r.speedup > 0]
        if not speedups:
            scores[op] = None
            reasons[op] = "本轮无性能数据（冒烟档不测性能）"
            continue
        anchor = op_anchors.get(op)
        if anchor is None:
            scores[op] = None
            reasons[op] = "尚无锚点（该算子还没有过全对轮次）"
            continue
        anchor_run, anchor_geo = anchor
        value = perf_score_against_anchor(geometric_mean(speedups), anchor_geo)
        if value is None:
            scores[op] = None
            reasons[op] = (
                f"锚点（run {anchor_run}）不过数值护栏"
                f"（偏离 1 不足 {MIN_ANCHOR_DEVIATION} 或得分越界）"
            )
            continue
        scores[op] = round(value, 4)
        reasons[op] = f"锚点 run {anchor_run}"
    return scores, reasons


def weighted_perf_score(
    op_scores: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> float | None:
    """轮级性能分 = 有分算子按权重加权平均。

    没有任何算子有分 → None（不可用，不编数）。
    权重分母只含有分的算子——分数旁必须披露哪些算子没出分（见 CLI 输出
    与报告模板），否则加权平均会被误读成「全量尺子的分」。
    """
    weights = weights or {}
    total_w = 0.0
    acc = 0.0
    for op, value in op_scores.items():
        if value is None:
            continue
        w = weights.get(op, 1.0)
        total_w += w
        acc += w * value
    if total_w == 0:
        return None
    return round(acc / total_w, 4)


def anchor_disclaimer() -> str:
    """报告里的固定声明：100 分口径相对什么、不是什么。"""
    return (
        "性能得分相对本评测体系**各算子首次全对轮次**（按算子锚点）计算，"
        "**不是跨编译器对照**——昇腾平台上不存在可安装的上游 Triton，"
        "无法构造跨编译器基线。"
    )


def _corpus_run_ids(store, corpus_version_id):
    rows = store.conn.execute(
        "SELECT id FROM eval_run WHERE corpus_version_id = ? ORDER BY id", (corpus_version_id,)).fetchall()
    return [int(row["id"]) for row in rows]
