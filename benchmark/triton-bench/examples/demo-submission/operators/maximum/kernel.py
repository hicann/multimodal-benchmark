# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Maximum 的 triton 实现（合法提交样例）：y = maximum(x1, x2)。

纯选择操作，不涉及算术精度，故不做 fp32 中转——
整型用例要求精确相等，多一次转换反而引入风险。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum_kernel(a_ptr, b_ptr, o_ptr, n, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, tl.maximum(a, b), mask=mask)


def maximum(x1, x2):
    # 非连续输入先挡住：kernel 线性寻址，硬跑是静默错误（2026-10-12 真机
    # 实测：maximum-i8-S123 静默算错）。.contiguous() 属布局搬运，逐元素类不允许用
    # ——见 exp/kernel.py 的说明。
    if not x1.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    # 非连续输入先挡住：kernel 线性寻址，硬跑是静默错误（2026-10-12 真机
    # 实测：maximum-i8-S123 静默算错）。.contiguous() 属布局搬运，逐元素类不允许用
    # ——见 exp/kernel.py 的说明。
    if not x2.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    out = torch.empty_like(x1)
    n = x1.numel()
    block = 1024
    _maximum_kernel[(triton.cdiv(n, block),)](x1, x2, out, n, block=block)
    return out
