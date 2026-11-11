# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按算子锚点测试（L0，无需 NPU）。

核心回归命题（死锁论证）：轮级「首轮锚点」把性能尺子的存在耦合到
「有一轮全部计入用例都判对」——题库混进一道坏题（无论谁交都过不了）
则任何一轮都不干净，锚点永远冻结不了，性能赛道整体死锁。
按算子锚点把死锁半径缩小到单个算子：坏题只废它所在的算子，
其余算子照常出分。
"""

from __future__ import annotations

import pytest

from triton_eval.anchors import (
    op_corpus_cases,
    resolve_op_anchors,
    score_per_operator,
    weighted_perf_score,
)
from triton_eval.models import (
    CaseResultRecord,
    CaseSource,
    CaseSpec,
    CaseStatus,
    CorpusManifest,
    OperatorSpec,
)
from triton_eval.store import Store, utcnow

N = 2  # 每算子题数


@pytest.fixture()
def shop(tmp_path):
    """两个算子（add / mul）、每算子 2 道题的题库。"""
    store = Store(tmp_path / "anchors.db")
    store.create_schema()
    ops = [
        OperatorSpec(name=name, source=CaseSource.VENDOR_SELFTEST, weight=1.0,
                     ref_path=f"operators/{name}/ref.py")
        for name in ("add", "mul")
    ]
    cases = [
        CaseSpec(code=f"{op}-f32-{i:04d}", operator=op,
                 source=CaseSource.VENDOR_SELFTEST, dtype="float32",
                 shape=[1024], seed=i, launch={}, feature_atoms=[])
        for op in ("add", "mul") for i in range(N)
    ]
    manifest = CorpusManifest(version="v-anchor-test", spec_version="0.1-mvp",
                              frozen_at=utcnow(), operators=ops, cases=cases,
                              manifest_hash="x")
    corpus_id = store.import_corpus(manifest)
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    yield store, corpus_id, sub_id
    store.close()


def _run(store, corpus_id, sub_id, spec: dict[str, list[tuple[str, float | None]]],
         env_fp: str = "fp") -> int:
    """按算子生成一轮结果。spec = {算子: [(终态, 加速比), ...]}。"""
    run_id = store.create_run(corpus_id, sub_id, "standard", {}, env_fp, "PASS")
    for op, rows in spec.items():
        for i, (status, speedup) in enumerate(rows):
            store.insert_case_result(CaseResultRecord(
                run_id=run_id, case_code=f"{op}-f32-{i:04d}", operator=op,
                status=CaseStatus(status), speedup=speedup,
                perf_cv=(0.01 if speedup else None),
                perf_median_ms=(0.4 if speedup else None),
            ))
    store.finish_run(run_id, "OK")
    return run_id


def _all(status: str = "PASS", speedup: float | None = 2.0, n: int = N):
    return [(status, speedup)] * n


# ---------- 锚点解析 ----------


def test_first_all_pass_round_self_anchors(shop) -> None:
    """首次全对的轮次自锚：锚点 = 该轮该算子的几何平均加速比。"""
    store, corpus_id, sub_id = shop
    run1 = _run(store, corpus_id, sub_id,
                {"add": _all(speedup=4.0), "mul": _all(speedup=2.0)})
    anchors = resolve_op_anchors(store, corpus_id)
    assert anchors["add"] == (run1, pytest.approx(4.0))
    assert anchors["mul"] == (run1, pytest.approx(2.0))

    results = store.case_results(run1)
    scores, reasons = score_per_operator(
        results, anchors, op_corpus_cases(store, corpus_id), {})
    assert scores["add"] == pytest.approx(100.0), "自锚 → ln(x)/ln(x) = 100"
    assert "锚点" in reasons["add"]


def test_dirty_operator_does_not_block_others(shop) -> None:
    """死锁半径：坏算子不配当尺子，但只废它自己——其余算子照常出分。

    这是对旧轮级门槛的替代设计：旧口径下「一轮里任何一个算子错」
    会废掉整轮的锚点资格；按算子口径下 mul 错不影响 add。
    """
    store, corpus_id, sub_id = shop
    run1 = _run(store, corpus_id, sub_id, {
        "add": _all(speedup=2.0),
        "mul": [("PASS", 2.0)] + [("WRONG_RESULT", 2.0)] * (N - 1),
    })
    anchors = resolve_op_anchors(store, corpus_id)
    assert "add" in anchors, "干净算子照常立尺"
    assert "mul" not in anchors, "含判错的算子不配当尺子"

    results = store.case_results(run1)
    scores, reasons = score_per_operator(
        results, anchors, op_corpus_cases(store, corpus_id), {})
    assert scores["add"] == pytest.approx(100.0)
    assert scores["mul"] is None
    assert "本轮未全对" in reasons["mul"], "为什么没分必须可读"


def test_broken_case_only_kills_its_operator(shop) -> None:
    """题库坏题的爆炸半径 = 单个算子：mul 永远全不了对，add 不受影响。

    多轮都不干净也不死锁——这正是轮级「全有或全无」门槛做不到的。
    """
    store, corpus_id, sub_id = shop
    bad_mul = [("WRONG_RESULT", 2.0)] * N
    _run(store, corpus_id, sub_id, {"add": _all(speedup=2.0), "mul": bad_mul})
    _run(store, corpus_id, sub_id, {"add": _all(speedup=4.0), "mul": bad_mul})

    anchors = resolve_op_anchors(store, corpus_id)
    assert "add" in anchors
    assert "mul" not in anchors

    # 第三轮 mul 修好了（比如题被修正后重跑）→ mul 立刻有分，
    # 且 add 的锚点仍是它自己的首次全对轮（最早原则）
    run3 = _run(store, corpus_id, sub_id, {"add": _all(speedup=8.0),
                                           "mul": _all(speedup=2.0)})
    anchors = resolve_op_anchors(store, corpus_id)
    assert anchors["mul"][0] == run3, "锚点取最早全对轮"
    assert anchors["add"][0] < run3

    scores, reasons = score_per_operator(
        store.case_results(run3), anchors, op_corpus_cases(store, corpus_id), {})
    assert scores["mul"] == pytest.approx(100.0), "自锚得 100"
    # add：锚点 geo=2.0，本轮 geo=8.0 → 100·ln8/ln2 = 300，护栏 [0,500] 内可用
    assert scores["add"] == pytest.approx(300.0)


def test_incomplete_run_cannot_anchor(shop) -> None:
    """--limit 跑了一半的轮次不配当尺子：没跑的题没有 PASS 结果。"""
    store, corpus_id, sub_id = shop
    run_id = store.create_run(corpus_id, sub_id, "standard", {}, "fp", "PASS")
    store.insert_case_result(CaseResultRecord(
        run_id=run_id, case_code="add-f32-0000", operator="add",
        status=CaseStatus.PASS, speedup=2.0,
    ))
    store.finish_run(run_id, "OK")

    assert resolve_op_anchors(store, corpus_id) == {}, \
        "只跑了 add 一半的题，不能当 add 的尺子"


def test_smoke_run_cannot_anchor(shop) -> None:
    """冒烟档没有性能数据，全对也立不了尺子（尺子必须量过每道题）。"""
    store, corpus_id, sub_id = shop
    run_id = store.create_run(corpus_id, sub_id, "smoke", {}, "fp", "PASS")
    for op in ("add", "mul"):
        for i in range(N):
            store.insert_case_result(CaseResultRecord(
                run_id=run_id, case_code=f"{op}-f32-{i:04d}", operator=op,
                status=CaseStatus.PASS,
            ))
    store.finish_run(run_id, "OK")

    assert resolve_op_anchors(store, corpus_id) == {}


def test_anchors_do_not_cross_corpus_versions(tmp_path) -> None:
    """锚点跨题库版本不通用：题不一样，尺子不能借。"""
    store = Store(tmp_path / "cross.db")
    store.create_schema()

    def _manifest(version: str) -> CorpusManifest:
        ops = [OperatorSpec(name="add", source=CaseSource.VENDOR_SELFTEST,
                            weight=1.0, ref_path="operators/add/ref.py")]
        cases = [
            CaseSpec(code=f"add-{version}-{i}", operator="add",
                     source=CaseSource.VENDOR_SELFTEST, dtype="float32",
                     shape=[1024], seed=i, launch={}, feature_atoms=[])
            for i in range(N)
        ]
        return CorpusManifest(version=version, spec_version="0.1-mvp",
                              frozen_at=utcnow(), operators=ops, cases=cases,
                              manifest_hash=version)

    corpus_a = store.import_corpus(_manifest("v-a"))
    corpus_b = store.import_corpus(_manifest("v-b"))
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")

    # v-b 的题库上全对了一轮
    run_b = store.create_run(corpus_b, sub_id, "standard", {}, "fp", "PASS")
    for i in range(N):
        store.insert_case_result(CaseResultRecord(
            run_id=run_b, case_code=f"add-v-b-{i}", operator="add",
            status=CaseStatus.PASS, speedup=2.0,
        ))
    store.finish_run(run_b, "OK")

    assert resolve_op_anchors(store, corpus_a) == {}, "v-b 的全对轮不能给 v-a 立尺"
    assert "add" in resolve_op_anchors(store, corpus_b)
    store.close()


# ---------- 逐算子出分 ----------


def test_score_reasons_are_readable(shop) -> None:
    """没分的算子必须给出可读原因，不能只留空。"""
    store, corpus_id, sub_id = shop
    run1 = _run(store, corpus_id, sub_id, {
        "add": _all(speedup=2.0),
        "mul": [("SUBMISSION_INVALID", None)] * N,  # 未提交
    })
    anchors = resolve_op_anchors(store, corpus_id)
    scores, reasons = score_per_operator(
        store.case_results(run1), anchors, op_corpus_cases(store, corpus_id), {})

    assert scores["add"] == pytest.approx(100.0)
    assert scores["mul"] is None
    # 未提交/未全对/无性能数据/无锚点——四种原因之一，必须落在纸上
    assert reasons["mul"], "无分原因不得为空"

    # 全对但无锚点（直接喂空锚点表，模拟「历史轮次还没解析进来」）
    scores2, reasons2 = score_per_operator(
        store.case_results(run1), {}, op_corpus_cases(store, corpus_id), {})
    assert scores2["add"] is None
    assert "尚无锚点" in reasons2["add"]


def test_score_per_operator_smoke_no_perf_data(shop) -> None:
    """全对但没有加速比（冒烟档）：无分，原因写明「无性能数据」。"""
    store, corpus_id, sub_id = shop
    run_id = store.create_run(corpus_id, sub_id, "smoke", {}, "fp", "PASS")
    for op in ("add", "mul"):
        for i in range(N):
            store.insert_case_result(CaseResultRecord(
                run_id=run_id, case_code=f"{op}-f32-{i:04d}", operator=op,
                status=CaseStatus.PASS,
            ))
    store.finish_run(run_id, "OK")

    scores, reasons = score_per_operator(
        store.case_results(run_id), {}, op_corpus_cases(store, corpus_id), {})
    assert scores["add"] is None
    assert "无性能数据" in reasons["add"]


# ---------- 轮级加权 ----------


def test_weighted_perf_score() -> None:
    assert weighted_perf_score({"a": 100.0, "b": 50.0}, {"a": 1.0, "b": 1.0}) \
        == pytest.approx(75.0)
    # 无分算子不进分母
    assert weighted_perf_score({"a": 100.0, "b": None}, {"a": 1.0, "b": 3.0}) \
        == pytest.approx(100.0)
    # 权重生效
    assert weighted_perf_score({"a": 100.0, "b": 50.0}, {"a": 3.0, "b": 1.0}) \
        == pytest.approx(87.5)
    # 全部无分 → None（不可用，不编数）
    assert weighted_perf_score({"a": None}, {"a": 1.0}) is None
    assert weighted_perf_score({}, {}) is None
