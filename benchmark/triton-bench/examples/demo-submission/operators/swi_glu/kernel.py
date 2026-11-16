# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SwiGlu 的 triton 实现（合法提交样例）：output = silu(a) * b。

在最后一维上把输入拆成两等份，输出在该维长度减半。
silu 用 sigmoid 的定义写：a / (1 + exp(-a))。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swi_glu_kernel(x_ptr, o_ptr, cols, half, block: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, block)
    mask = offs < half
    a = tl.load(x_ptr + row * cols + offs, mask=mask).to(tl.float32)
    b = tl.load(x_ptr + row * cols + half + offs, mask=mask).to(tl.float32)
    y = a / (1.0 + tl.exp(-a)) * b
    tl.store(o_ptr + row * half + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


def _swi_glu_implementation(input_value, dim=-1):
    """只覆盖 dim=-1：非最后一维需要先做维度搬运，
    而 FusedComposite 类不允许用 permute/transpose 绕道。"""
    d = dim if dim >= 0 else dim + input_value.dim()
    if d != input_value.dim() - 1:
        raise NotImplementedError(f"样例只实现 dim=-1，实际 dim={dim}")
    # 非连续输入先挡住：kernel 线性寻址，硬跑是静默错误（2026-10-12 真机
    # 实测：swi_glu-f16-S123 静默算错）。.contiguous() 属布局搬运，逐元素类不允许用
    # ——见 exp/kernel.py 的说明。
    if not input_value.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    cols = input_value.shape[-1]
    half = cols // 2
    rows = input_value.numel() // cols
    out = torch.empty(list(input_value.shape[:-1]) + [half], dtype=input_value.dtype,
                      device=input_value.device)
    block = max(1, triton.next_power_of_2(half))
    _swi_glu_kernel[(rows,)](input_value, out, cols, half, block=block)
    return out


def swi_glu(input_value, dim=-1):
    """Use the registered input_value contract at the API boundary."""
    return _swi_glu_implementation(input_value, dim)
