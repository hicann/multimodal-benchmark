# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MaskedScale 的参考实现（评测方资产）：y = x * mask * scale。

dtype 归属规则：
    规格里 x 只接受浮点、mask 接受 int8 / uint8 / 浮点。
    而用例的 dtype 是从二者并集里轮换出来的，故按类型判断归属——
    整型 dtype 给 mask（x 取 fp32），浮点 dtype 给 x（mask 取 int8）。
"""

from __future__ import annotations

import torch

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)


def _rand(seed: int, n: int, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dtype in _INT_DTYPES:
        return torch.randint(1, 100, (n,), generator=g, dtype=torch.int64).to(dtype)
    return torch.randn(n, generator=g).to(dtype)


def _mask(seed: int, n: int, dtype):
    """掩码只取 0/1：随机浮点掩码会把「是否掩码」和「乘了个系数」混在一起。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=g, dtype=torch.int64).to(dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    n = 1
    for d in shape:
        n *= d
    if dtype in _INT_DTYPES:
        x = _rand(seed, n, torch.float32)
        mask = _mask(seed + 1, n, dtype)
    else:
        x = _rand(seed, n, dtype)
        mask = _mask(seed + 1, n, torch.int8)
    return [x.reshape(shape).to(device), mask.reshape(shape).to(device)]


def reference(x, mask, scale=1.0):
    x32 = x.to(torch.float32)
    return (x32 * mask.to(torch.float32) * float(scale)).to(x.dtype)
