# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Maximum 的参考实现（评测方资产）。

多输入算子：主 shape 只描述 x1，x2 由 make_inputs 一并造出。
"""

from __future__ import annotations

import torch

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)


def _rand(seed: int, n: int, dtype):
    """按 seed 造 n 个数。整型用 randint 而非 randn→int（后者截断出大量 0）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dtype in _INT_DTYPES:
        return torch.randint(1, 100, (n,), generator=g, dtype=torch.int64).to(dtype)
    return torch.randn(n, generator=g).to(dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    """两个输入必须用**不同 seed**：
    同源输入会让 maximum 退化成恒等映射，任何实现都能「通过」。
    """
    n = 1
    for d in shape:
        n *= d
    x1 = _rand(seed, n, dtype).reshape(shape).to(device)
    x2 = _rand(seed + 1, n, dtype).reshape(shape).to(device)
    return [x1, x2]


def reference(x1, x2):
    """逐元素取大。纯选择操作，不涉及算术精度，故不走 fp32 中转。"""
    return torch.maximum(x1, x2)
