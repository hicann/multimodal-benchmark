# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D10（正控制）：非布局类算子返回非连续输出（SUB-EXE-002 应拦截）。

数值完全正确（手写 exp），但输出是 (n,2) 缓冲的第 0 列视图：
形状对、值对、不连续。Elementwise 类不属于布局变换，
执行期兜底应判 SUBMISSION_INVALID。

取非连续视图用的是 buf[:, 0]（下标表达式，不是调用），
静态扫描同样看不见——只能靠执行期兜底。
"""

from __future__ import annotations

import math

import torch


def _walk(t):
    if t.dim() == 0:
        yield float(t)
        return
    for sub in t:
        yield from _walk(sub)


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    lb = 1.0 if base in (None, -1.0) else math.log(float(base))
    vals = [math.exp((v * float(scale) + float(shift)) * lb) for v in _walk(x)]

    shape = list(x.shape)
    if not shape:
        return torch.tensor(vals[0], dtype=x.dtype, device=x.device)

    n = 1
    for s in shape:
        n *= s
    buf = torch.empty((n, 2), dtype=x.dtype, device=x.device)
    out = buf[:, 0]  # 非连续视图：形状 (n,)、步长 2
    for i, v in enumerate(vals):
        out[i] = v
    return out
