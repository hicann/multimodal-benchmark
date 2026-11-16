# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""组装一次用例执行的参数（L0，无 NPU 依赖）。

为什么要有这个模块：
    CLI 与端到端测试曾经各自组装 execute_case 的参数，两边分叉。
    测试漏传了 make_inputs，于是多输入算子只拿到**一个**张量，
    参考实现以 `TypeError: reference() missing 1 required positional
    argument: 'mask'` 失败，被判 ORACLE_FAIL——
    看起来像评测方资产坏了，实际是调用方没传全参数。

    这类分叉无法靠"记得同步"避免，只能让两边调用同一个函数。
"""

from __future__ import annotations

from typing import Any

from .models import CaseSpec
from .operator_registry import by_name
from .oracle import (
    cross_reference_for,
    make_inputs_for as operator_make_inputs,
    reference_for,
)


def execute_kwargs(case: CaseSpec, impl: Any = None) -> dict[str, Any]:
    """给出执行一个用例所需的全部参数。

    `impl` 由调用方（提交载入结果）传入；为 None 时判 SUBMISSION_INVALID。
    """
    spec_op = by_name(case.operator)

    # 属性以**用例**为准，规格只补默认值：
    # 「沿哪一维归约」是用例的一部分，写死默认值会让同一算子的
    # 多个变体退化成同一道题。
    attrs = spec_op.resolve_attrs(case.attrs) if spec_op else dict(case.attrs)

    return {
        "impl": impl,
        "reference": reference_for(case.operator),
        "cross_reference": cross_reference_for(case.operator),
        "attrs": attrs,
        # 多输入算子的输入构造来自 ref.py；缺失时执行器回落成单张量，
        # 而多输入算子的参考实现会因此缺参数并报错。
        "make_inputs": operator_make_inputs(case.operator),
        "category": spec_op.category.value if spec_op else None,
    }
