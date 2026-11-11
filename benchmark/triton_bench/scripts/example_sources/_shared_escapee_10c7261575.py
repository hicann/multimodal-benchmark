# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""被 ADV-E1 的清单引用的「逃逸」实现文件。

它位于提交目录**之外**（examples/adversarial-submissions/_shared/）：
- 清单允许 path 含 ../ 与绝对路径，无根目录约束；
- 加载会真的执行它（等于从任意可读位置导入代码）；
- 提交快照/指纹只扫提交根内部，这个文件对「评测的就是当时那一份」不可见。
"""

from __future__ import annotations

__all__ = [
    'math',
    'torch',
]

import math

import torch






def _walk(t):
    if t.dim() == 0:
        yield float(t)
        return
    for sub in t:
        yield from _walk(sub)


def _nest(shape, it):
    if not shape:
        return next(it)
    return [_nest(shape[1:], it) for _ in range(shape[0])]


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    lb = 1.0 if base in (None, -1.0) else math.log(float(base))
    vals = [math.exp((v * float(scale) + float(shift)) * lb) for v in _walk(x)]
    if not list(x.shape):
        return torch.tensor(vals[0], dtype=x.dtype, device=x.device)
    return torch.tensor(
        _nest(list(x.shape), iter(vals)), dtype=x.dtype, device=x.device
    )
