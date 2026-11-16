# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""分类分级体系测试（L0，无 NPU 依赖）。"""

from __future__ import annotations

import pytest

from triton_eval.taxonomy import (
    DEFAULT_CASES_PER_CATEGORY,
    Category,
    Feasibility,
    Level,
    all_categories,
    categories_missing_operators,
    feasibility_of,
    is_known_category,
)


def test_thirteen_categories() -> None:
    assert len(all_categories()) == 13, "必须是 13 个语义族"


def test_unknown_category_rejected() -> None:
    assert is_known_category("Elementwise") is True
    assert is_known_category("NotACategory") is False


def test_known_categories_cover_expected_set() -> None:
    expected = {
        "Elementwise", "Reduction", "Broadcast", "MaskPredicate", "Normalization",
        "IndexGather", "ScatterUpdate", "Contraction", "LayoutTransform",
        "SortSelect", "NumericalStable", "FusedComposite", "VVFusion",
    }
    assert set(all_categories()) == expected


def test_level_is_independent_of_category() -> None:
    """level 与 category 正交：互不决定。"""
    assert [lv.value for lv in Level] == ["L1", "L2", "L3", "L4"]


def test_infeasible_category_kept_as_placeholder() -> None:
    """无可行 triton 算子的类别必须保留占位，不得从体系中移除。"""
    assert feasibility_of("VVFusion") is Feasibility.INFEASIBLE
    assert "VVFusion" in all_categories(), "不得因无算子而移除该类别"


def test_native_categories() -> None:
    for cat in ("Elementwise", "Reduction", "Broadcast", "MaskPredicate",
                "Normalization", "IndexGather", "ScatterUpdate"):
        assert feasibility_of(cat) is Feasibility.NATIVE, f"{cat} 应为 triton 擅长"


def test_feasible_categories_need_operator_selection() -> None:
    """这些类别可做但要挑算子（如 Contraction 只选 matmul，不选卷积）。"""
    for cat in ("Contraction", "LayoutTransform", "SortSelect",
                "NumericalStable", "FusedComposite"):
        assert feasibility_of(cat) is Feasibility.FEASIBLE


def test_missing_categories_surfaced() -> None:
    present = {"Elementwise", "Reduction"}
    missing = categories_missing_operators(present)
    assert "VVFusion" in missing
    assert "Elementwise" not in missing
    assert len(missing) == 11


def test_default_cases_per_category() -> None:
    assert DEFAULT_CASES_PER_CATEGORY == 5


def test_category_enum_values_stable() -> None:
    """枚举值是与参考仓库对齐的契约，不得随意改名。"""
    assert Category.ELEMENTWISE.value == "Elementwise"
    assert Category.VV_FUSION.value == "VVFusion"
