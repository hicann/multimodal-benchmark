# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""报告生成测试（L0，无需 NPU）。

报告只依赖数据库记录，因此可以在本地用构造数据完整验证——
这正是 results.jsonl 契约带来的能力。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton_eval.models import CaseResultRecord, CaseStatus, RunScoreRecord
from triton_eval.report import (
    build_context,
    dump_env,
    render_html,
    render_markdown,
    write_report,
)
from triton_eval.store import Store, utcnow

ENV = {
    "arch": "aarch64", "os": "Linux 5.10.0-hce2", "kernel": "5.10.0-hce2",
    "npu_model": "Ascend910", "npu_count": 2, "npu_id": "0",
    "ascend_visible_devices": "0", "driver_version": "25.5.5",
    "cann_version": "9.0.0", "python_version": "3.11.4",
    "torch_version": "2.7.1+cpu", "torch_npu_version": "2.7.1.post4",
    "compiler_name": "triton-ascend", "compiler_version": "3.2.1",
    "triton_import_version": "3.2.0", "venv_path": "/home/x/venv/te-compiler",
    "venv_lock_hash": "abc123", "clock_locked": False,
    "env_health": "ok", "env_health_notes": "",
}

SUBMISSION = {"name": "triton-ascend", "version": "3.2.1", "git_commit": None,
              "install_mode": "wheel"}


@pytest.fixture()
def run_ctx(tmp_path: Path):
    """构造一次完整运行：含通过、静默错误、编译失败、参考失效四种终态。"""
    store = Store(tmp_path / "t.db")
    store.create_schema()
    corpus_id = store.import_corpus(_manifest())
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(corpus_id, sub_id, "standard", ENV, "fp-123", "PASS",
                              run_dir=str(tmp_path / "run-1"))

    results = []
    for code, st, sp in [
        ("add-f32-1024", CaseStatus.PASS, 2.0),
        ("add-f32-4096", CaseStatus.PASS, 2.5),
        ("add-f16-2048", CaseStatus.WRONG_RESULT, 2.1),
        ("add-i32-1024", CaseStatus.COMPILE_FAIL, None),
        ("mul-f32-1024", CaseStatus.ORACLE_FAIL, None),
    ]:
        rec = CaseResultRecord(
            run_id=run_id, case_code=code, operator=code.split("-")[0], status=st,
            max_rel_err=(1e-2 if st is CaseStatus.WRONG_RESULT else 1e-7),
            max_abs_err=(1e-3 if st is CaseStatus.WRONG_RESULT else 1e-9),
            speedup=sp, perf_cv=0.01, perf_median_ms=0.4 if sp else None,
            samples_ms=[0.4] if sp else [],
            compile_err="triton.CompilationError: ..." if st is CaseStatus.COMPILE_FAIL else None,
        )
        store.insert_case_result(rec)
        results.append(rec)
    store.finish_run(run_id, "OK")

    score = RunScoreRecord(
        run_id=run_id, coverage_ops_core=66.67, coverage_strict=33.33, severe_count=1,
        perf_geo=2.16, perf_score=100.0, fast_1=1.0, fast_1_2=1.0, fast_2=0.3333,
        unstable_rate=0.0, baseline_anchor_ref="PER-OP", scorer_version="0.1.0",
        computed_at=utcnow(),
    )
    store.insert_run_score(score)

    ctx = build_context(
        run=store.get_run(run_id), env=ENV, submission=SUBMISSION,
        results=results, score=score.model_dump(),
        corpus_version="v0.3-multi-input-frozen", spec_version="0.1-mvp",
        scorer_version="0.1.0", run_dir=str(tmp_path / "run-1"),
        op_perf={
            "add": {"score": 100.0, "reason": "锚点 run 1"},
            "mul": {"score": None, "reason": "本轮未全对（1 个判错）"},
        },
    )
    yield ctx, tmp_path
    store.close()


def _manifest():
    from triton_eval.models import CaseSource, CaseSpec, CorpusManifest, OperatorSpec

    ops = []
    for name in ['add', 'mul', 'sum', 'softmax', 'layer_norm']:
        ops.append(OperatorSpec(
            name=name,
            source=CaseSource.VENDOR_SELFTEST,
            weight=1.0,
            ref_path=f'operators/{name}/ref.py',
        ))
    cases = []
    for code in ['add-f32-1024', 'add-f32-4096', 'add-f16-2048', 'add-i32-1024', 'mul-f32-1024']:
        cases.append(CaseSpec(
            code=code,
            operator=code.split('-')[0],
            source=CaseSource.VENDOR_SELFTEST,
            dtype='float32',
            shape=[1024],
            seed=1,
            launch={},
            feature_atoms=[],
        ))
    return CorpusManifest(version="v0.3-multi-input-frozen", spec_version="0.1-mvp",
                          frozen_at=utcnow(), operators=ops, cases=cases,
                          manifest_hash="x")


# ---------- 任务 8.4 ----------


def test_markdown_report_has_all_mandatory_sections(run_ctx) -> None:
    ctx, _ = run_ctx
    md = render_markdown(ctx)
    for section in ["## 1. 元信息", "## 2. 环境声明", "## 3. 被测对象",
                    "## 4. 功能完备性结果", "## 5. 综合性能结果",
                    "## 6. 稳定性结果", "## 8. 可追溯附录"]:
        assert section in md, f"报告缺少章节：{section}"


def test_report_contains_traceability_fields(run_ctx) -> None:
    ctx, _ = run_ctx
    md = render_markdown(ctx)
    assert "fp-123" in md, "必须披露环境指纹"
    assert "0.1-mvp" in md, "必须披露规范版本"
    assert "v0.3-multi-input-frozen" in md, "必须披露测试集版本"
    assert "0.1.0" in md, "必须披露评分器版本"
    assert "results.jsonl" in md, "必须给出真相源路径"


def test_write_report_produces_md_and_html(run_ctx) -> None:
    ctx, tmp_path = run_ctx
    out = write_report(ctx, tmp_path / "reports", "report-run-1")
    assert out["markdown"].is_file()
    assert out["html"].is_file()
    assert "<html" in out["html"].read_text(encoding="utf-8")
    assert out["markdown"].read_text(encoding="utf-8").startswith("# Triton")


# ---------- 任务 8.5 ----------


def test_severe_cases_get_own_highlighted_section(run_ctx) -> None:
    ctx, _ = run_ctx
    md = render_markdown(ctx)
    assert "## 7. 静默错误清单" in md
    assert "add-f16-2048" in md, "静默错误的用例必须列出"
    assert len(ctx["severe_cases"]) == 1


def test_interpretation_boundary_always_present(run_ctx) -> None:
    ctx, _ = run_ctx
    md = render_markdown(ctx)
    assert "## 9. 结果解释边界" in md
    assert "不跨硬件厂商做单一总分排名" in md
    assert "不给出合成总分" in md


def test_no_total_score_in_report(run_ctx) -> None:
    """规范 §3.3 禁止跨赛道换算，报告不得出现合成总分。"""
    ctx, _ = run_ctx
    md = render_markdown(ctx)
    assert "Total" not in md


# ---------- 任务 8.6 ----------


def test_unlocked_clock_marks_perf_unverified(run_ctx) -> None:
    ctx, _ = run_ctx
    assert ENV["clock_locked"] is False
    md = render_markdown(ctx)
    assert "UNVERIFIED" in md, "未锁频必须标注性能数据不可信"
    assert ctx["unverified_perf"] is True


def test_locked_clock_no_unverified_marker(run_ctx) -> None:
    ctx, _ = run_ctx
    locked = dict(ENV)
    locked["clock_locked"] = True
    ctx2 = dict(ctx)
    ctx2["env"] = locked
    ctx2["unverified_perf"] = False
    md = render_markdown(ctx2)
    assert "UNVERIFIED" not in md


def test_html_also_marks_unverified(run_ctx) -> None:
    ctx, _ = run_ctx
    html = render_html(ctx)
    assert "UNVERIFIED" in html


# ---------- 无效运行 ----------


def test_invalid_run_is_prominent(run_ctx) -> None:
    ctx, _ = run_ctx
    ctx["run"]["status"] = "INVALID"
    ctx["run"]["invalid_reason"] = "环境健康探针失败"
    md = render_markdown(ctx)
    assert "INVALID" in md
    assert "环境健康探针失败" in md


def test_perf_anchor_disclaimer_on_every_scored_report(run_ctx) -> None:
    """有分数的报告必须写明按算子锚点口径、声明不是跨编译器对照。

    旧版声明只在「首轮基线」自己的报告里出现，后续轮次全缺；
    按算子锚点口径下统一声明（1.7 的缺口就此关闭）。
    """
    ctx, _ = run_ctx
    md = render_markdown(ctx)
    assert "按算子锚点" in md
    assert "各算子首次全对轮次" in md
    assert "不是跨编译器对照" in md
    html = render_html(ctx)
    assert "不是跨编译器对照" in html, "网页版声明不得缺席"


def test_op_perf_table_discloses_unscored_operators(run_ctx) -> None:
    """逐算子分数表：没分的算子点名给原因，防加权分被读成全量尺子的分。"""
    ctx, _ = run_ctx
    md = render_markdown(ctx)
    assert "按算子性能分" in md
    assert "锚点 run 1" in md
    assert "本轮未全对" in md, "无分原因必须同屏"
    assert "有分算子" in md, "分母口径必须写明"


def test_dump_env_writes_sorted_json(tmp_path: Path) -> None:
    p = dump_env(ENV, tmp_path / "run-9")
    raw = p.read_text(encoding="utf-8")
    loaded = json.loads(raw)
    assert loaded["arch"] == "aarch64"
    assert list(loaded) == sorted(loaded), "环境快照必须按键排序，保证可比"
