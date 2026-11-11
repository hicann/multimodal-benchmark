# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""RmsNorm 的 triton 实现（合法提交样例）：x * rsqrt(mean(x^2) + eps) * gamma。

每个 program 处理一行（最后一维）。gamma 的长度与最后一维一致。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(x_ptr, g_ptr, o_ptr, cols, eps, block: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, block)
    mask = offs < cols
    # 越界位置填 0：求和时不能让它们贡献，也不能引入 NaN
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / cols
    r = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    tl.store(o_ptr + row * cols + offs, (x * r * g).to(o_ptr.dtype.element_ty),
             mask=mask)


def rms_norm(x, gamma, epsilon=1e-6):
    # 非连续输入先挡住：kernel 线性寻址，硬跑是静默错误（2026-10-12 真机
    # 实测：rms_norm-f32-S85 静默算错）。.contiguous() 属布局搬运，逐元素类不允许用
    # ——见 exp/kernel.py 的说明。
    if not x.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    # 非连续输入先挡住：kernel 线性寻址，硬跑是静默错误（2026-10-12 真机
    # 实测：rms_norm-f32-S85 静默算错）。.contiguous() 属布局搬运，逐元素类不允许用
    # ——见 exp/kernel.py 的说明。
    if not gamma.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    out = torch.empty_like(x)
    cols = x.shape[-1]
    rows = x.numel() // cols
    block = max(1, triton.next_power_of_2(cols))
    _rms_norm_kernel[(rows,)](x, gamma, out, cols, float(epsilon), block=block)
    return out
