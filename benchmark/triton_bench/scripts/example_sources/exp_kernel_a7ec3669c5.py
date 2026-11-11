# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-E4：positional-only 形参（签名校验只查名字，不查绑定语义）。

inspect.signature 列出的参数名与规格完全一致
（x, base, scale, shift），加载校验通过。
但 `/` 之后 base/scale/shift 是仅位置参数——
执行器按「张量位置传 + 属性关键字传」的约定调用：
    impl(x, base=2.0, scale=1.0, shift=0.0)   → TypeError
每个用例都 RUNTIME_FAIL，错误却算在被测提交头上。

同类问题的另一半是 Scatter（见 ADV-C1）：
规格签名 (data, dim, indices, updates, reduce) 与
位置输入 (data, index, updates) + 关键字 dim 的调用约定结构性冲突。
"""

from __future__ import annotations

__all__ = [
    '_nest',
    '_walk',
    'math',
    'torch',
]

import math

import torch
from scripts.example_sources._shared_escapee_10c7261575 import _nest, _walk










def exp(x, base=-1.0, scale=1.0, shift=0.0, /):
    lb = 1.0 if base in (None, -1.0) else math.log(float(base))
    vals = [math.exp((v * float(scale) + float(shift)) * lb) for v in _walk(x)]
    if not list(x.shape):
        return torch.tensor(vals[0], dtype=x.dtype, device=x.device)
    return torch.tensor(
        _nest(list(x.shape), iter(vals)), dtype=x.dtype, device=x.device
    )
