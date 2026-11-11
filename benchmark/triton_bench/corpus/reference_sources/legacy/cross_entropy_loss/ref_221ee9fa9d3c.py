# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CrossEntropyLoss 的参考实现（评测方资产）。

两处约定：

1. dtype 归属：input 只接受浮点、target 接受 int32 / int64 / 浮点，
   而用例 dtype 取自二者并集。整型 dtype 归 target（input 取 fp32），
   浮点 dtype 归 input（target 取 int64 硬标签）。

2. rank==1 的 shape 视为**单样本** [1, C]：
   规格允许 input 为 1 维（即 (C,)），但交叉熵需要「样本」这一维，
   直接把 [C] 当 N=C 会造出 C 个单类样本，语义失真。
"""

from __future__ import annotations

import torch

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)


def make_inputs(*, seed, dtype, shape, device, attrs):
    if dtype in _INT_DTYPES:
        in_dtype, tgt_dtype = torch.float32, dtype
    else:
        in_dtype, tgt_dtype = dtype, torch.int64

    in_shape = list(shape) if len(shape) >= 2 else [1, shape[0]]
    n, c = in_shape[0], in_shape[1]
    tgt_shape = [n] + in_shape[2:]

    n_in = 1
    for d in in_shape:
        n_in *= d
    n_t = 1
    for d in tgt_shape:
        n_t *= d

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_in, generator=g).to(in_dtype).reshape(in_shape).to(device)
    g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    target = torch.randint(0, c, (n_t,), generator=g2, dtype=torch.int64).to(tgt_dtype)
    return [x, target.reshape(tgt_shape).to(device)]


def reference(input_value, target, reduction="mean", ignore_index=-100):
    """loss 在 fp32 下计算（log_softmax 对低精度敏感），再转回 input 的 dtype。"""
    x32 = input_value.to(torch.float32)
    loss = torch.nn.functional.cross_entropy(
        x32, target.to(torch.long), reduction=reduction, ignore_index=int(ignore_index)
    )
    return loss.to(input_value.dtype)
