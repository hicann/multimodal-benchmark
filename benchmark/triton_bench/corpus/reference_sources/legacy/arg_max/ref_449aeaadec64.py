# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ArgMax 的参考实现（评测方资产）。

整型输入必须**互不相同**（用 randperm）：
存在并列最大值时，「返回哪个下标」是实现自由，torch 与 triton 可以合理地
返回不同下标，而两者都对。那种用例不具备可判定性，不应进入冻结集。
"""

from __future__ import annotations

import torch

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)


def make_inputs(*, seed, dtype, shape, device, attrs):
    n = 1
    for d in shape:
        n *= d
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dtype in _INT_DTYPES:
        t = torch.randperm(n, generator=g).to(torch.int64).to(dtype)
    else:
        t = torch.randn(n, generator=g).to(dtype)
    return [t.reshape(shape).to(device)]


def reference(input_value, dim, keepdim=False):
    """比较在 fp32 域进行（dtype 转换单调，比较顺序不变）。"""
    return torch.argmax(input_value.to(torch.float32), dim=dim, keepdim=bool(keepdim))
