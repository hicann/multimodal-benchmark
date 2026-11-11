# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-STR-007：用 triton.autotune 自行搜索启动配置。

启动配置（BLOCK / num_warps / num_stages）应当由评测方给定，
由被测方自选等于把考题改成开卷；且 autotune 的搜索开销
不计入被测实现的耗时，性能数据会失真。
"""

import torch
import triton
import triton.language as tl


def _cfgs():
    return [triton.Config({"BLOCK": b}, num_warps=w)
            for b in (256, 512, 1024) for w in (4, 8)]


@triton.autotune(configs=_cfgs(), key=["n"])
@triton.jit
def _exp_kernel(x_ptr, o_ptr, n, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    tl.store(o_ptr + offs, tl.exp(tl.load(x_ptr + offs, mask=mask)), mask=mask)


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    out = torch.empty_like(x)
    n = x.numel()
    _exp_kernel[(triton.cdiv(n, 1024),)](x, out, n)
    return out
