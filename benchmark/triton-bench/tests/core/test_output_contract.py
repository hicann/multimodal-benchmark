# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""执行期输出契约（多输出 / 元数 / dtype）。

守护一类「名义覆盖、实际零校验」的假象：
    Cummin 声明了 values 与 indices 两个输出，看起来覆盖了多输出，
    但比对只取第一个元素——第二个输出从未参与判定。
    实测（ADV-F1）：indices 全填垃圾，5 个用例全部 PASS。

    这类假象比「没覆盖」更危险：它会让人以为测过了。

本测试不需要 NPU：判定逻辑在 inputs.as_outputs，
比对聚合在执行器，二者都可在纯 torch 上验证。
"""

from __future__ import annotations

import pytest

from triton_eval.inputs import _as_tensor, as_outputs

pytestmark = pytest.mark.torch


def test_single_tensor_normalized() -> None:
    import torch

    t = torch.zeros(2, 2)
    assert as_outputs(t) == [t]


def test_multi_output_all_kept() -> None:
    """多输出必须全部保留——只留第一个会让第二个输出零校验。"""
    import torch

    v, i = torch.zeros(4), torch.zeros(4, dtype=torch.long)
    assert as_outputs((v, i)) == [v, i]
    assert as_outputs([v, i]) == [v, i]


def test_as_tensor_takes_first_for_diagnosis_only() -> None:
    """_as_tensor 取首个，仅供诊断与形状检查用，不得用于判定。"""
    import torch

    v, i = torch.zeros(4), torch.zeros(4, dtype=torch.long)
    assert _as_tensor((v, i)) is v


def test_non_tensor_rejected() -> None:
    with pytest.raises(TypeError):
        as_outputs(42)
    with pytest.raises(TypeError):
        as_outputs(("a", "b"))


def test_mixed_tuple_keeps_only_tensors() -> None:
    """返回 (tensor, 元数据) 是常见写法，非张量的元数据不应导致拒绝。"""
    import torch

    v = torch.zeros(4)
    assert as_outputs((v, {"meta": 1})) == [v]


def test_output_arity_mismatch_flagged() -> None:
    """少交一个输出等价于「没算」，不得因第一个输出对了就放过。"""
    import torch

    expected = as_outputs((torch.zeros(4), torch.zeros(4, dtype=torch.long)))
    actual = as_outputs(torch.zeros(4))
    assert len(actual) != len(expected)


def test_dtype_mismatch_is_detected() -> None:
    """float64 偷渡：用更高精度算完交上来，等于绕过低精度这条考题。"""
    import torch

    a = torch.zeros(4, dtype=torch.float64)
    e = torch.zeros(4, dtype=torch.float32)
    assert a.dtype != e.dtype
