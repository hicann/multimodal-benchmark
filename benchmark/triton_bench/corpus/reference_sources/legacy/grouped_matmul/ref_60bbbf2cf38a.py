# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GroupedMatmul 的参考实现（评测方资产）。

分组语义：x 的行按 group_list 切成 E 段，第 g 段与 weight[g] 相乘并加 bias[g]。
因此 make_inputs 必须读 group_list 才能造出 weight / bias——
这也是「输入构造必须下沉到算子」最典型的例子：
单看主 shape [M, K] 根本推不出 E 和 N。
"""

from __future__ import annotations

import torch

N_OUT = 64  # 输出维度 N：与 K 解耦，避免 weight 随 shape 平方增长


def _rand(seed: int, n: int, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, generator=g).to(dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    group_list = [int(v) for v in (attrs.get("group_list") or [])]
    m = shape[0]
    k = shape[1] if len(shape) > 1 else 1
    if not group_list or sum(group_list) != m:
        # group_list 与 M 不符时以 M 兜底为一组：
        # 宁可退化成一个普通 matmul，也不要造出无法对齐的输入
        group_list = [m]
    e = len(group_list)

    x = _rand(seed, m * k, dtype).reshape(m, k).to(device)
    weight = _rand(seed + 1, e * k * N_OUT, dtype).reshape(e, k, N_OUT).to(device)
    bias = _rand(seed + 2, e * N_OUT, dtype).reshape(e, N_OUT).to(device)
    return [x, weight, bias]


def reference(x, weight, bias, group_list):
    x32 = x.to(torch.float32)
    w32 = weight.to(torch.float32)
    b32 = bias.to(torch.float32)
    outs = []
    start = 0
    for g, cnt in enumerate([int(v) for v in group_list]):
        xi = x32[start:start + cnt]
        outs.append(torch.matmul(xi, w32[g]) + b32[g])
        start += cnt
    return torch.cat(outs, dim=0).to(x.dtype)
