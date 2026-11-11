# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Scatter 的参考实现（评测方资产）。

两个关键约束：

1. index 在 dim 维上**不得重复**。同一位置被写两次时，最终值取决于写入
   顺序，而顺序由实现自行决定——那样用例就不可复现，确定性筛查会剔除它，
   或者更糟：把实现差异误判成错误。故用 argsort 生成沿 dim 的排列。

2. index / updates 在 dim 维上取主 shape 的一半：既保证有足够写入量，
   又留出「未被写入」的位置来验证原样保留。
"""

from __future__ import annotations

import torch

from triton_eval.inputs import reference_sample
from triton_eval.refcheck import scatter_reference


def _rand(seed: int, n: int, dtype):
    return reference_sample(seed, n, dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    dim = int(attrs.get("dim", 0))
    n = 1
    for d in shape:
        n *= d
    data = _rand(seed, n, dtype).reshape(shape).to(device)

    idx_shape = list(shape)
    idx_shape[dim] = max(1, idx_shape[dim] // 2)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    # argsort 给出 [0, k) 的一个排列 → 沿 dim 唯一；k <= size(dim) 保证不越界
    index = torch.rand(idx_shape, generator=g).argsort(dim=dim).to(torch.int64)

    m = 1
    for d in idx_shape:
        m *= d
    updates = _rand(seed + 2, m, dtype).reshape(idx_shape).to(device)
    return [data, index.to(device), updates]


def reference(data, index, updates, dim=0, reduce=None):
    """reduce=None 表示「后写覆盖」；add / mul 为累积写入。"""
    return scatter_reference(data, index, updates, dim, reduce)
