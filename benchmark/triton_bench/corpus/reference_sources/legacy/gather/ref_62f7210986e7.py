# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Gather 的参考实现（评测方资产）：y = x.gather(dim, index)。

index 的取值范围必须由 make_inputs 约束在 [0, size(dim))：
越界索引在不同后端上的行为不一致（报错 / 截断 / 取到脏数据），
那会让「谁对谁错」无法判定。
"""

from __future__ import annotations

import torch

from triton_eval.inputs import reference_sample


def _rand(seed: int, n: int, dtype):
    return reference_sample(seed, n, dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    dim = int(attrs.get("dim", 0))
    n = 1
    for d in shape:
        n *= d
    x = _rand(seed, n, dtype).reshape(shape).to(device)

    size = shape[dim] if 0 <= dim < len(shape) else shape[-1]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    index = torch.randint(0, size, (n,), generator=g, dtype=torch.int64)
    return [x, index.reshape(shape).to(device)]


def reference(x, index, dim=0):
    """纯索引搬运，不涉及算术精度，故不走 fp32 中转。"""
    return torch.gather(x, dim, index)
