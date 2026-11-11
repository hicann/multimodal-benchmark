# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""RmsNorm 的参考实现（评测方资产）：y = x / sqrt(mean(x^2) + eps) * gamma。

gamma 的形状与 x 最后一维一致，由 make_inputs 造出。
"""

from __future__ import annotations

import torch


def _rand(seed: int, n: int, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, generator=g).to(dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    n = 1
    for d in shape:
        n *= d
    x = _rand(seed, n, dtype).reshape(shape).to(device)
    gamma = _rand(seed + 1, shape[-1], dtype).to(device)
    return [x, gamma]


def reference(x, gamma, epsilon=1e-6):
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    y = x32 * torch.rsqrt(var + float(epsilon)) * gamma.to(torch.float32)
    return y.to(x.dtype)
