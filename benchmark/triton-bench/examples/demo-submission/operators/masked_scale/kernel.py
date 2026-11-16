# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MaskedScale 的 triton 实现（合法提交样例）：y = x * mask * scale。

mask 可能是整型（int8 / uint8）也可能是浮点，因此统一转到 fp32 再相乘——
与参考实现的「fp32 计算 → 转回目标 dtype」路径保持一致。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_scale_kernel(x_ptr, m_ptr, o_ptr, n, scale, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    tl.store(o_ptr + offs, (x * m * scale).to(o_ptr.dtype.element_ty), mask=mask)


def masked_scale(x, mask, scale=1.0):
    # 非连续输入先挡住：kernel 线性寻址，硬跑是静默错误（2026-10-12 真机
    # 实测：masked_scale-f32-S141 / i8-S207 静默算错）。.contiguous() 属布局搬运，逐元素类不允许用
    # ——见 exp/kernel.py 的说明。
    if not x.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    # 非连续输入先挡住：kernel 线性寻址，硬跑是静默错误（2026-10-12 真机
    # 实测：masked_scale-f32-S141 / i8-S207 静默算错）。.contiguous() 属布局搬运，逐元素类不允许用
    # ——见 exp/kernel.py 的说明。
    if not mask.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    out = torch.empty_like(x)
    n = x.numel()
    block = 1024
    _masked_scale_kernel[(triton.cdiv(n, block),)](
        x, mask, out, n, float(scale), block=block
    )
    return out
