# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""文档与规范一致性的守卫测试。

技术报告里有若干条文与实测环境或彼此冲突，这些偏差必须有据可查，
不能散落在代码注释里被人遗忘。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFLICTS = PROJECT_ROOT / "docs" / "spec-conflicts.md"


def test_conflict_report_exists() -> None:
    assert CONFLICTS.is_file(), "缺少技术报告条文冲突清单"


def test_conflict_report_names_the_cross_track_total_conflict() -> None:
    """§3.3 禁止换算，§6.4 却合成 Total —— 必须记录并指明采用哪一方。"""
    text = CONFLICTS.read_text(encoding="utf-8")
    assert "3.3" in text and "6.4" in text
    assert "不产出" in text and "Total" in text


def test_conflict_report_covers_all_known_conflicts() -> None:
    text = CONFLICTS.read_text(encoding="utf-8")
    for topic in ["容差", "TRITON_INTERPRET", "B0", "UNSUPPORTED", "等权",
                  "SQLite", "float64", "锚点", "num_warps", "manifest_hash"]:
        assert topic in text, f"冲突清单缺少主题：{topic}"


def test_conflict_report_records_two_engineering_constraints() -> None:
    text = CONFLICTS.read_text(encoding="utf-8")
    assert "inspect.getsourcelines" in text, "需记录 kernel 必须是真实文件"
    assert "同名" in text, "需记录 triton 与 triton-ascend 目录同名覆盖风险"


def test_spec_declares_no_total_composition() -> None:
    from triton_eval.specload import load_spec

    assert load_spec().compose_total is False


def test_capability_checklist_matches_data() -> None:
    """确认版能力点清单必须与机器清单 + 题库证据逐字一致（防文档过期）。

    docs/feature-atoms-checklist.md 是给合作方定稿用的交付物，过期了
    会误导定稿；改 spec/feature_atoms.yaml 或题库后必须重跑生成脚本。
    """
    import importlib.util

    script = PROJECT_ROOT / "scripts" / "build_capability_checklist.py"
    spec = importlib.util.spec_from_file_location("build_capability_checklist", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rendered = mod.render()
    committed = (PROJECT_ROOT / "docs" / "feature-atoms-checklist.md").read_text(
        encoding="utf-8"
    )
    assert rendered == committed, (
        "docs/feature-atoms-checklist.md 与数据不一致；"
        "请重跑 python scripts/build_capability_checklist.py"
    )


def test_capability_checklist_covers_all_active_atoms() -> None:
    """确认版文档必须列全全部生效与预留条目（随 yaml 版本动态）——漏项即失真。"""
    import yaml

    fa = yaml.safe_load(
        (PROJECT_ROOT / "spec" / "feature_atoms.yaml").read_text(encoding="utf-8")
    )
    text = (PROJECT_ROOT / "docs" / "feature-atoms-checklist.md").read_text(
        encoding="utf-8"
    )
    for atom in fa["atoms"]:
        assert f"`{atom['id']}`" in text, f"确认版清单漏了生效条目 {atom['id']}"
    for atom in fa["reserved"]:
        assert f"`{atom['id']}`" in text, f"确认版清单漏了预留条目 {atom['id']}"
