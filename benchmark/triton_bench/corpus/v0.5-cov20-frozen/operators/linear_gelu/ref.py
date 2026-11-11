# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""LinearGelu 的参考实现（评测方资产）——组合参考。

参考 = 原语参考的顺序组合（单一代码事实来源，不新写 oracle）：
先分组矩阵乘（单组即普通矩阵乘），后 gelu。
原语 ref.py 按同语料目录的相对路径加载——参考随语料版本整体迁移，
组合参考永远引用同一版本的原语，不会串版本。
"""

from __future__ import annotations

from triton_eval.refcheck import load_reference_segment


def _seg(entry: str):
    return load_reference_segment(__file__, entry)


_GMM = _seg("grouped_matmul")
_GELU = _seg("gelu")


def make_inputs(*, seed, dtype, shape, device, attrs):
    """输入构造与 GroupedMatmul 完全一致：x / weight / bias 由段 1 决定。"""
    return _GMM.make_inputs(seed=seed, dtype=dtype, shape=shape, device=device,
                            attrs=attrs)


def reference(x, weight, bias, group_list):
    y = _GMM.reference(x, weight, bias, group_list)
    return _GELU.reference(y)
