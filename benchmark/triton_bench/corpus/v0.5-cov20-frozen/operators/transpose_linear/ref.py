# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TransposeLinear 的参考实现（评测方资产）——组合参考。

参考 = 原语参考的顺序组合：先 transpose（rank-2 固定置换 [1,0]——
rank-2 下唯一有意义的非平凡置换，规格不设 perm 属性），
后分组矩阵乘。考察点：布局变换直接喂收缩，中间转置不落盘。

shape 耦合（与其它收缩融合不同，必须说清）：
    主输入 x:[M, K]，转置后 [K, M] 作为矩阵乘的行堆——矩阵乘的
    行数是 K、列数（收缩维）是 M。因此 weight 形如 [E, M, N]，
    group_list 之和必须等于 K（即 shape[1]）。
    make_inputs 借用 GroupedMatmul 段的构造：以 shape=[K, M] 调它，
    其 weight/bias 正是本算子所需，其 x 弃用（本算子的 x 自造）。
"""

from __future__ import annotations

from triton_eval.refcheck import load_reference_segment

import torch


def _seg(entry: str):
    return load_reference_segment(__file__, entry)


_GMM = _seg("grouped_matmul")


def _rand(seed: int, n: int, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, generator=g).to(dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    m, k = shape[0], shape[1]
    x = _rand(seed, m * k, dtype).reshape(m, k).to(device)
    # 以 [K, M] 借用段构造：weight=[E, M, N]、bias=[E, N]（K 是矩阵乘行数）
    _xt, weight, bias = _GMM.make_inputs(seed=seed + 13, dtype=dtype,
                                         shape=[k, m], device=device,
                                         attrs=attrs)
    return [x, weight, bias]


def reference(x, weight, bias, group_list):
    xt = x.to(torch.float32).transpose(0, 1)
    y = _GMM.reference(xt, weight, bias, group_list)
    return y.to(x.dtype)
