# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MaskedGelu 的参考实现（评测方资产）——组合参考。

参考 = 原语参考的顺序组合：掩码保留（mask=0 处输出置零）后接 gelu
等价于 gelu(x) * mask——先 gelu 后掩码与先掩码后 gelu 数值一致
（掩码外输入为 0 时 gelu(0)=0，但本参考按「先算后抹」实现，
与提交侧常见写法均兼容）。掩码构造复用 MaskedScale 段（0/1 整数掩码）。
"""

from __future__ import annotations

from triton_eval.refcheck import load_reference_segment

import torch


def _seg(entry: str):
    return load_reference_segment(__file__, entry)


_MASKED = _seg("masked_scale")
_GELU = _seg("gelu")


def make_inputs(*, seed, dtype, shape, device, attrs):
    return _MASKED.make_inputs(seed=seed, dtype=dtype, shape=shape,
                               device=device, attrs=attrs)


def reference(x, mask, approximate="none"):
    g = _GELU.reference(x, approximate)
    return (g.to(torch.float32) * mask.to(torch.float32)).to(x.dtype)
