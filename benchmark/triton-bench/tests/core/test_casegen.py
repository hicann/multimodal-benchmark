# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""用例派生测试（L0，无 NPU 依赖）。"""

from __future__ import annotations

import pytest

from triton_eval.casegen import (
    count_by_category,
    count_by_level,
    derive_cases,
    generate_all,
)
from triton_eval.operator_registry import ALL_OPERATORS, by_name
from triton_eval.taxonomy import DEFAULT_CASES_PER_CATEGORY, categories_missing_operators


# ---------- 任务 3.4 ----------


def test_derived_cases_within_spec() -> None:
    """所有派生用例必须落在算子规格声明范围内。"""
    for op in ALL_OPERATORS:
        for case in derive_cases(op, count=3):
            op.validate_case_params(case.dtype, case.shape, case.attrs)


def test_derivation_covers_multiple_dimensions() -> None:
    """派生必须跨 shape / dtype / attrs / value_range 多维度，不能全同。"""
    op = by_name("Exp")
    assert op is not None
    cases = derive_cases(op, count=5)
    assert len({tuple(c.shape) for c in cases}) >= 2, "shape 应有变化"
    assert len({c.dtype for c in cases}) >= 2, "dtype 应有变化"
    assert len({c.value_range.kind for c in cases}) >= 2, "value_range 应有变化"


def test_derived_case_has_no_implementation_fields() -> None:
    """用例只描述数据与判定参数，不得含任何实现路径字段。"""
    op = by_name("Exp")
    assert op is not None
    case = derive_cases(op, count=1)[0]
    fields = set(case.model_dump())
    assert "kernel_path" not in fields, "用例不得绑定 kernel 路径"
    assert "entry" not in fields, "用例不得绑定入口函数名"


def test_derived_case_records_category_and_level() -> None:
    op = by_name("RmsNorm")
    assert op is not None
    case = derive_cases(op, count=1)[0]
    assert case.category == "Normalization"
    assert case.level == "L2"


# ---------- 任务 3.5 ----------


def test_five_cases_per_category() -> None:
    """每 category 绑 5 个 case——额度按 category 分配，不按算子数。"""
    cases = generate_all(ALL_OPERATORS)
    counts = count_by_category(cases)
    for cat, n in counts.items():
        assert n == DEFAULT_CASES_PER_CATEGORY, f"{cat} 有 {n} 个，应为 5"
    assert len(counts) == 12, "12 个 category 有算子（VVFusion 空缺）"
    assert len(cases) == 12 * DEFAULT_CASES_PER_CATEGORY == 60


def test_category_with_two_operators_shares_quota() -> None:
    """同 category 下多个算子时，额度均分而非各自 5 个。

    曾经的 bug：按算子分配导致 Elementwise（2 个算子）产出 10 个 case。
    """
    counts = count_by_category(generate_all(ALL_OPERATORS))
    assert counts["Elementwise"] == 5
    assert counts["Normalization"] == 5


def test_missing_category_is_empty_not_absent() -> None:
    """VVFusion 无算子：用例数为 0，且能被列为空缺。"""
    counts = count_by_category(generate_all(ALL_OPERATORS))
    assert "VVFusion" not in counts
    assert categories_missing_operators(set(counts)) == ["VVFusion"]


def test_level_distribution_recorded() -> None:
    levels = count_by_level(generate_all(ALL_OPERATORS))
    assert set(levels) <= {"L1", "L2", "L3", "L4"}
    assert sum(levels.values()) == 60


def test_layout_transform_generates_non_contiguous() -> None:
    """LayoutTransform 的核心语义是布局变换，应生成非连续用例。"""
    op = by_name("Transpose")
    assert op is not None
    cases = derive_cases(op, count=5)
    assert any(not c.contiguous for c in cases), "应有非连续用例"


def test_other_categories_stay_contiguous() -> None:
    op = by_name("Exp")
    assert op is not None
    cases = derive_cases(op, count=5)
    assert all(c.contiguous for c in cases)


# ---------- 参数合法性（修 bug 后固化） ----------


def test_dim_attributes_respect_shape_rank() -> None:
    """dim 不得越界：曾经生成过 dim=3 而 shape 只有 2 维。"""
    for op in ALL_OPERATORS:
        for case in derive_cases(op, count=5):
            _assert_case_dimensions(case)


def _assert_case_dimensions(case):
    for name in ("dim", "axis"):
        if name in case.attrs and isinstance(case.attrs[name], int):
            assert 0 <= case.attrs[name] < len(case.shape), (
                f"{case.code}: {name}={case.attrs[name]} 越界，shape={case.shape}"
            )


def test_perm_is_a_valid_permutation() -> None:
    """perm 必须是 rank 的合法置换；曾经对 1 维 shape 生成 [1,1]。"""
    op = by_name("Transpose")
    assert op is not None
    for case in derive_cases(op, count=5):
        perm = case.attrs.get("perm")
        assert perm is not None
        assert sorted(perm) == list(range(len(case.shape))), (
            f"{case.code}: perm={perm} 不是 shape={case.shape} 的合法置换"
        )


def test_required_attrs_always_present() -> None:
    for op in ALL_OPERATORS:
        for name in op.required_attrs():
            for case in derive_cases(op, count=5):
                assert name in case.attrs, f"{case.code} 缺必填属性 {name}"


def test_seeds_are_unique_and_deterministic() -> None:
    op = by_name("Exp")
    assert op is not None
    a = derive_cases(op, count=5)
    b = derive_cases(op, count=5)
    assert [c.seed for c in a] == [c.seed for c in b], "派生必须确定性"
    assert len({c.seed for c in a}) == 5, "种子应互不相同"


def test_case_codes_unique() -> None:
    cases = generate_all(ALL_OPERATORS)
    codes = [c.code for c in cases]
    assert len(codes) == len(set(codes)), "用例业务键必须唯一"
