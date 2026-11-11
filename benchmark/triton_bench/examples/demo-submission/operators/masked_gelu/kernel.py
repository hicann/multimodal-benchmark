# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MaskedGelu 的 triton 实现（合法提交样例）：mask=1 处 gelu(x)，mask=0 处置零。

要点：
  - 融合算子的样例内核是**锚点载体**（按算子锚点要求全部用例判对才能
    冻结基准），故按 stride 寻址实现，不以守卫挡非连续输入
  - tanh 用 exp 表达（本环境已验证 tl.exp 可用，tanh 内建未经验证）——
    口径同 gelu/kernel.py
  - approximate="none"（erf 精确路径）未实现：erf 需要 libdevice，
    未在本环境验证。宁可显性 NotImplementedError，也不用未经验证的
    近似凑一个"通过"（口径同 gelu/kernel.py 的既有决策）
  - mask 与 x 同形：mask 是评测方构造的连续张量，x 可能非连续——
    两者寻址分开（x 用 strides，mask 按连续 flat 下标）
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_gelu_kernel(x_ptr, m_ptr, o_ptr, geometry, configuration: tl.constexpr):
    n, xs0, xs1, xs2, d0, d1, d2 = geometry
    (block,) = configuration
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    i2 = offs % d2
    r = offs // d2
    i1 = r % d1
    r = r // d1
    i0 = r % d0
    x_off = i0 * xs0 + i1 * xs1 + i2 * xs2
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(m_ptr + offs, mask=mask, other=0).to(tl.float32) != 0
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner) * keep.to(tl.float32)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


def masked_gelu(x, mask, approximate="none"):
    if approximate != "tanh":
        raise NotImplementedError(
            f"样例只实现 approximate='tanh'（gelu 的 erf 精确路径需要 "
            f"libdevice，未在本环境验证），实际收到 {approximate!r}")
    if x.dim() > 3:
        raise NotImplementedError(f"样例只实现 rank≤3，实际 rank={x.dim()}")
    if not mask.is_contiguous():
        # mask 由评测方构造、恒连续；到这说明布局约定被破坏，显性暴露
        raise NotImplementedError("mask 张量预期为连续布局")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n = x.numel()
    d0, d1, d2 = (x.shape + (1,) * (3 - x.dim()))
    xs = (x.stride() + (0,) * (3 - x.dim()))
    block = 1024
    _masked_gelu_kernel[triton.cdiv(n, block),](x, mask, out, (n, xs[0], xs[1], xs[2], d0, d1, d2), (block,))
    return out
