# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Scatter 的参考实现（评测方资产）。

关键约束：

1. reduce=None（覆盖写入）：index 在 dim 维上**不得重复**。同一位置被写
   两次时，最终值取决于写入顺序，而顺序由实现自行决定——那样用例就不
   可复现。故用 argsort 生成沿 dim 的排列。

2. reduce=add（并发累加，v0.5 新变体）：索引**允许重复**——这正是该变体
   的考点：「和」与写入顺序无关是语义要求（FA-CONCUR-ACCUM 载体）。
   仅整型（int32/int64，生成侧已限定）：
     - 整型加法精确且交换结合，参考在加速卡上走原子加法也两遍一致；
     - 浮点加法不满足结合律，顺序差异会变成 ULP 级噪声，语义不可判定；
     - 窄整型会溢出（updates 与 data 各取 [1,100)，累积和超出 int8/16）。
   构造改用均匀 randint——重复是自然发生的，不是人为排列。

3. index / updates 在 dim 维上取主 shape 的一半：既保证有足够写入量，
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
    reduce = attrs.get("reduce")
    n = 1
    for d in shape:
        n *= d
    data = _rand(seed, n, dtype).reshape(shape).to(device)

    idx_shape = list(shape)
    idx_shape[dim] = max(1, idx_shape[dim] // 2)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    if reduce == "add":
        # 并发累加变体：均匀随机索引，重复自然发生（考点即顺序无关累加）。
        # 防御性兜底：非宽整型直接退回唯一排列——浮点/窄整型的并发和
        # 不可判定/会溢出，生成侧 _combo_allowed 已限定，此处双保险。
        # 注意 dtype 是 torch dtype 对象（make_inputs 的调用约定），不是字符串
        if dtype in (torch.int32, torch.int64):
            size = shape[dim] if 0 <= dim < len(shape) else shape[-1]
            index = torch.randint(0, size, idx_shape, generator=g,
                                  dtype=torch.int64)
        else:
            index = torch.rand(idx_shape, generator=g).argsort(dim=dim) \
                .to(torch.int64)
    else:
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
