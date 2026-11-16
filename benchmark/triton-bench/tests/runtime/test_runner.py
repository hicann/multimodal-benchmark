# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""执行参数组装的测试（L0，无需 NPU）。

守护一条实测教训：
    CLI 与端到端测试曾经各自组装 execute_case 的参数。测试漏传了
    make_inputs，多输入算子只拿到**一个**张量，参考实现以
    `TypeError: reference() missing 1 required positional argument: 'mask'`
    失败并被判 ORACLE_FAIL——看起来像评测方资产坏了，
    实际是调用方没传全参数。

    这类分叉无法靠「记得同步」避免，只能让两边调用同一个函数。
    本测试锁住 execute_kwargs 对多输入算子的行为。
"""

from __future__ import annotations

from triton_eval.models import CaseSource, CaseSpec
from triton_eval.operator_registry import by_name
from triton_eval.runner import execute_kwargs


def _case(operator: str, attrs: dict | None = None) -> CaseSpec:
    return CaseSpec(
        code=f"{operator}-x", operator=operator,
        source=CaseSource.PARAM_VARIANT, dtype="float32", shape=[8, 16],
        seed=1, attrs=attrs or {},
    )


def test_multi_input_operators_carry_make_inputs() -> None:
    """多输入算子必须带上构造过程，否则参考实现会缺参数。"""
    for name in ("Maximum", "MaskedScale", "RmsNorm", "Gather",
                 "Scatter", "GroupedMatmul", "CrossEntropyLoss"):
        kw = execute_kwargs(_case(name))
        assert kw["make_inputs"] is not None, f"{name} 缺少 make_inputs，参考实现会缺参数"


def test_single_input_operators_fall_back_to_default() -> None:
    for name in ("Exp", "Softmax", "Gelu"):
        kw = execute_kwargs(_case(name))
        assert kw["make_inputs"] is None, f"{name} 是单输入，应回落到默认构造"


def test_attrs_defaulted_from_operator_spec() -> None:
    """用例未给的属性由规格补默认值——否则参考实现拿到 None。"""
    kw = execute_kwargs(_case("Softmax"))
    spec = by_name("Softmax")
    for a in spec.attrs:
        if not a.required:
            assert kw["attrs"][a.name] == a.default


def test_required_attr_taken_from_case() -> None:
    kw = execute_kwargs(_case("Transpose", {"perm": [1, 0]}))
    assert kw["attrs"]["perm"] == [1, 0]


def test_category_provided_for_runtime_guard() -> None:
    """运行期兜底需要 category 才能对布局类放行。"""
    assert execute_kwargs(_case("Transpose"))["category"] == "LayoutTransform"
    assert execute_kwargs(_case("Exp"))["category"] == "Elementwise"


def test_impl_defaults_to_none() -> None:
    """未给实现时 impl 必须是 None——judge 据此判 SUBMISSION_INVALID。"""
    assert execute_kwargs(_case("Exp"))["impl"] is None
