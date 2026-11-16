# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MaskedSoftmax 的参考实现（评测方资产）——组合参考。

参考 = 原语参考的顺序组合：掩码保留（mask=0 的位置置 -inf，等效于
MaskedScale 段「掩码外不参与」的归一化版语义），后 softmax 沿 dim。

确定性要点：make_inputs 保证**每条归约行至少一个未掩码元素**——
全 -inf 行的 softmax 是 NaN，会污染「输出有限」门禁；随机 0/1 掩码
在小行宽下有小概率整行全零，故显式修补（首元素置 1），确定性保持。
"""

from __future__ import annotations

from triton_eval.refcheck import load_reference_segment

import torch


def _seg(entry: str):
    return load_reference_segment(__file__, entry)


_MASKED = _seg("masked_scale")
_SOFTMAX = _seg("softmax")


def make_inputs(*, seed, dtype, shape, device, attrs):
    x, mask = _MASKED.make_inputs(seed=seed, dtype=dtype, shape=shape,
                                  device=device, attrs=attrs)
    dim = int(attrs.get("dim", len(shape) - 1))
    if dim < 0:
        dim += len(shape)
    # 每条归约行保底一个未掩码元素：全掩码行的 softmax 无定义（NaN）
    m = mask
    if m.dtype in (torch.int8, torch.uint8, torch.bool, torch.int32, torch.int64):
        row_any = m.to(torch.int64).sum(dim=dim, keepdim=True)
        patch = (row_any == 0).to(m.dtype)
        idx = [slice(None)] * m.dim()
        idx[dim] = 0
        m[tuple(idx)] = m[tuple(idx)] + patch.squeeze(dim).to(m.dtype)
    return [x, m]


def reference(x, mask, dim):
    x32 = x.to(torch.float32)
    keep = mask.to(torch.float32)
    masked = x32.masked_fill(keep == 0, float("-inf"))
    y = _SOFTMAX.reference(masked, dim)
    return y.to(x.dtype)
