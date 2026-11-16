# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""演化对照与生成日志的单元测试（L0，无需 NPU）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from triton_eval.buildlog import LOG_FILENAME, GenerationLog, read_log
from triton_eval.corpusmod import build_manifest, load_manifest
from triton_eval.evolution import diff_manifests, render_markdown
from triton_eval.models import CaseSource, CaseSpec, OperatorSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _case(code: str, operator: str, *, dtype: str = "float32",
          shape: tuple[int, ...] = (1024,), seed: int = 1,
          atoms: tuple[str, ...] = ("FA-MATH-ADD",)) -> CaseSpec:
    return CaseSpec(
        code=code, operator=operator, source=CaseSource.VENDOR_SELFTEST,
        dtype=dtype, shape=list(shape), seed=seed, launch={},
        feature_atoms=list(atoms),
    )


def _operators(*names: str) -> list[OperatorSpec]:
    return [
        OperatorSpec(name=n, source=CaseSource.VENDOR_SELFTEST, weight=1.0,
                     ref_path=f"operators/{n}/ref.py")
        for n in names
    ]


def _build(tmp: Path, version: str, operators, cases, excluded=None):
    return build_manifest(tmp, version, "0.1-mvp", operators, cases,
                          excluded=excluded)


@pytest.fixture()
def pair(tmp_path: Path):
    """构造一对小清单：覆盖增删改全部分支。"""
    old = _build(
        tmp_path / "old", "v0-old",
        _operators("Add", "Exp"),
        [
            _case("add-f32-00", "Add", seed=11),
            # 弱键演示：同编号，新版本 dtype 变了
            _case("add-f32-01", "Add", dtype="float32", seed=12),
            _case("exp-f32-00", "Exp", atoms=("FA-MATH-EXP",)),
        ],
        excluded=[{"code": "exp-i8-00", "reason": "参考实现自检：int8 溢出"}],
    )
    new = _build(
        tmp_path / "new", "v0-new",
        _operators("Add", "Cummin"),
        [
            _case("add-f32-00", "Add", seed=11),           # 内容未变
            _case("add-f32-01", "Add", dtype="bfloat16", seed=12),  # 同编号改 dtype
            _case("cummin-f32-00", "Cummin", atoms=("FA-CTRL-SCAN",)),
        ],
        excluded=[{"code": "add-i8-00", "reason": "参考实现自检：输出含 NaN"}],
    )
    return old, new


def test_diff_detects_operator_add_remove(pair) -> None:
    old, new = pair
    d = diff_manifests(old, new)
    assert [o["name"] for o in d["operators_added"]] == ["Cummin"]
    assert [o["name"] for o in d["operators_removed"]] == ["Exp"]


def test_diff_counts_per_operator_cases(pair) -> None:
    old, new = pair
    d = diff_manifests(old, new)
    rows = {r["operator"]: r for r in d["op_case_delta"]}
    # Add 数量没变（2→2，只是其中一条换了 dtype）——不该出现在变化表里
    assert "Add" not in rows
    assert d["op_case_unchanged"] == 1
    assert rows["Cummin"] == {"operator": "Cummin", "old": 0, "new": 1, "delta": 1}
    assert rows["Exp"] == {"operator": "Exp", "old": 1, "new": 0, "delta": -1}


def test_diff_case_level_add_remove_change_keep(pair) -> None:
    old, new = pair
    d = diff_manifests(old, new)
    assert d["cases_added"] == ["cummin-f32-00"]
    assert d["cases_removed"] == ["exp-f32-00"]
    assert len(d["cases_changed"]) == 1
    ch = d["cases_changed"][0]
    assert ch["code"] == "add-f32-01"
    assert list(ch["fields"]) == ["dtype"]  # 字段级：只报真正变了的字段
    assert ch["fields"]["dtype"] == ("float32", "bfloat16")
    assert d["cases_kept_identical"] == 1  # add-f32-00
    assert d["cases_changed_field_freq"] == {"dtype": 1}


def test_diff_atoms_gained_lost(pair) -> None:
    old, new = pair
    d = diff_manifests(old, new)
    assert d["atoms_gained"] == ["FA-CTRL-SCAN"]
    assert d["atoms_lost"] == ["FA-MATH-EXP"]


def test_diff_passes_through_excluded_records(pair) -> None:
    old, new = pair
    d = diff_manifests(old, new)
    assert d["excluded_old"] == [{"code": "exp-i8-00", "reason": "参考实现自检：int8 溢出"}]
    assert d["excluded_new"] == [{"code": "add-i8-00", "reason": "参考实现自检：输出含 NaN"}]


def test_diff_dtype_and_size_buckets(pair) -> None:
    old, new = pair
    d = diff_manifests(old, new)
    dtype_keys = {r["key"] for r in d["dtype_delta"]}
    assert ("Add", "float32") in dtype_keys   # 2 → 1
    assert ("Add", "bfloat16") in dtype_keys  # 0 → 1
    size_keys = {r["key"] for r in d["size_delta"]}
    assert ("Exp", "SMALL") in size_keys  # 同规模下 Exp 算子消失


def test_render_markdown_contains_key_sections(pair) -> None:
    d = diff_manifests(*pair)
    md = render_markdown(d)
    assert "## v0-old → v0-new" in md
    assert "新增算子 1 个：Cummin" in md
    assert "移除算子 1 个：Exp" in md
    assert "同编号但内容变更** 1 个" in md
    assert "dtype：\"float32\" → \"bfloat16\"" in md
    assert "参考实现自检：int8 溢出" in md  # 旧版剔除原因可见
    assert "参考实现自检：输出含 NaN" in md
    assert "内容未变的用例：1 个" in md


# ---------- 真实语料：自动化 diff 必须复现手工审计结论 ----------


def test_v04_to_v05_diff_matches_manual_audit() -> None:
    """v0.4→v0.5 的手工审计结论（2026-10-13 逐算子清点）必须被自动 diff 复现。"""
    old = load_manifest(PROJECT_ROOT / "corpus" / "v0.4-cov20-frozen" / "manifest.json")
    new = load_manifest(PROJECT_ROOT / "corpus" / "v0.5-cov20-frozen" / "manifest.json")
    d = diff_manifests(old, new)
    assert len(new.cases) == 289 and len(old.cases) == 240
    assert len(d["operators_added"]) == 11  # 11 个融合算子
    rows = {r["operator"]: r for r in d["op_case_delta"]}
    # 手工清点过的按算子再分配（composition-and-pattern-dimensions.md §6 第 1 行）
    assert rows["SwiGlu"] == {"operator": "SwiGlu", "old": 20, "new": 6, "delta": -14}
    assert rows["Scatter"] == {"operator": "Scatter", "old": 20, "new": 10, "delta": -10}
    assert len(d["atoms_gained"]) == 16  # 71 - 55
    assert d["atoms_lost"] == []  # 无原子失去覆盖


def test_committed_evolution_doc_matches_regeneration() -> None:
    """docs/corpus-evolution.md 与脚本再生成结果逐字一致（防文档过期）。"""
    script = PROJECT_ROOT / "scripts" / "corpus_evolution.py"
    spec = importlib.util.spec_from_file_location("corpus_evolution", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    committed = (PROJECT_ROOT / "docs" / "corpus-evolution.md").read_text(
        encoding="utf-8"
    )
    assert mod.render_all() == committed, (
        "docs/corpus-evolution.md 与清单推导结果不一致；"
        "请重跑 python scripts/corpus_evolution.py --all --out docs/corpus-evolution.md"
    )


# ---------- 生成日志 ----------


def test_generation_log_appends_steps(tmp_path: Path) -> None:
    path = tmp_path / LOG_FILENAME
    log = GenerationLog(path)
    log.step("build_start", per_category=20)
    log.step("build_end", manifest_written=False)

    events = read_log(path)
    steps = [e["step"] for e in events]
    assert steps[0] == "log_header"  # 新文件自动带头部记录（格式版本）
    assert steps[1:] == ["build_start", "build_end"]
    assert all("ts" in e for e in events)
    assert events[1]["per_category"] == 20
    assert events[2]["manifest_written"] is False


def test_generation_log_is_append_only(tmp_path: Path) -> None:
    """重复构建不抹历史：第二次构建追加在既有事件之后。"""
    path = tmp_path / LOG_FILENAME
    GenerationLog(path).step("build_start")
    GenerationLog(path).step("build_start")  # 模拟重跑

    events = read_log(path)
    assert len(events) == 3  # 1 头部 + 2 次构建各 1 条
    assert [e["step"] for e in events[1:]] == ["build_start", "build_start"]
