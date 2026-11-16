# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GatherExp 的参考实现（评测方资产）——组合参考。

参考 = 原语参考的顺序组合：先 gather（纯索引搬运），后 exp。
输入构造复用 Gather 段——index 由构造约束在 [0, size(dim)) 内
（越界索引在不同后端行为不一致，无法判定，见 Gather 段的说明）。
"""

from __future__ import annotations

from triton_eval.refcheck import load_reference_segment


def _seg(entry: str):
    return load_reference_segment(__file__, entry)


_GATHER = _seg("gather")
_EXP = _seg("exp")


def make_inputs(*, seed, dtype, shape, device, attrs):
    return _GATHER.make_inputs(seed=seed, dtype=dtype, shape=shape,
                               device=device, attrs=attrs)


def reference(x, index, dim=0, base=-1.0, scale=1.0, shift=0.0):
    y = _GATHER.reference(x, index, dim)
    return _EXP.reference(y, base, scale, shift)
