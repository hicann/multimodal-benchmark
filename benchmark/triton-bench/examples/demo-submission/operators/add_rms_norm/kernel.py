# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AddRmsNorm 的 triton 实现（合法提交样例）。

y = rms_norm(x1 + x2) * gamma：残差加与归一化一次完成——
加完的中间结果不落显存，这正是融合的意义。

要点：
  - 融合算子的样例内核是**锚点载体**（要求全部用例判对），按 stride
    寻址实现（x1/x2 可非连续；gamma 由评测方构造为连续）
  - 每个 program 处理一行（最后一维归约）。行可以很长（rank-1 时整条
    4M 元素是一行）——两遍分块循环：Σ(x1+x2)² 累加，再归一化写回
  - fp32 域累加（口径同 rms_norm/kernel.py：方差在 fp32 域求，避免
    fp16 平方的精度塌陷）
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rms_norm_kernel(x1_ptr, x2_ptr, g_ptr, o_ptr, geometry, configuration: tl.constexpr):
    cols, eps, a1, a2, ac, b1, b2, bc, gs, r1, r2 = geometry
    (block,) = configuration
    p = tl.program_id(0)
    i2 = p % r2
    t = p // r2
    i1 = t % r1
    base_a = i1 * a1 + i2 * a2
    base_b = i1 * b1 + i2 * b2
    var_acc = 0.0
    for k0 in range(0, cols, block):
        offs = k0 + tl.arange(0, block)
        valid = offs < cols
        xa = tl.load(x1_ptr + base_a + offs * ac, mask=valid, other=0.0).to(tl.float32)
        xb = tl.load(x2_ptr + base_b + offs * bc, mask=valid, other=0.0).to(tl.float32)
        s = xa + xb
        var_acc += tl.sum(s * s, axis=0)
    rr = 1.0 / tl.sqrt(var_acc / cols + eps)
    for k0 in range(0, cols, block):
        offs = k0 + tl.arange(0, block)
        valid = offs < cols
        xa = tl.load(x1_ptr + base_a + offs * ac, mask=valid, other=0.0).to(tl.float32)
        xb = tl.load(x2_ptr + base_b + offs * bc, mask=valid, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=valid, other=0.0).to(tl.float32)
        y = (xa + xb) * rr * g
        tl.store(o_ptr + p * cols + offs, y.to(o_ptr.dtype.element_ty), mask=valid)


def _outer(shape, strides):
    """最内维以外的维度，前补齐到 2 个：返回 (r1, r2), (步长1, 步长2)。

    rank-1 → (1,1)/(0,0)（单行，基址 0）；rank-2 → (1,R)/(0,stride0)；
    rank-3 → (B,S)/(stride0,stride1)。r2 对齐**最后一个**外维。
    """
    os_, ob = tuple(shape[:-1]) or (1,), tuple(strides[:-1]) or (0,)
    r2 = os_[-1] if len(os_) >= 1 else 1
    r1 = os_[-2] if len(os_) >= 2 else 1
    b2 = ob[-1] if len(ob) >= 1 else 0
    b1 = ob[-2] if len(ob) >= 2 else 0
    return r1, r2, b1, b2


def add_rms_norm(x1, x2, gamma, epsilon=1e-6):
    if x1.dim() > 3:
        raise NotImplementedError(f"样例只实现 rank≤3，实际 rank={x1.dim()}")
    if not gamma.is_contiguous():
        raise NotImplementedError("gamma 张量预期为连续布局")
    cols = x1.shape[-1]
    rows = x1.numel() // cols
    out = torch.empty(x1.shape, dtype=x1.dtype, device=x1.device)
    r1a, r2a, a1, a2 = _outer(x1.shape, x1.stride())
    r1b, r2b, b1, b2 = _outer(x2.shape, x2.stride())
    if not ((r1a, r2a) == (r1b, r2b)):
        raise ValueError("x1/x2 外维必须一致")
    block = 1024
    _add_rms_norm_kernel[rows,](
        x1,
        x2,
        gamma,
        out,
        (cols, float(epsilon), a1, a2, x1.stride(-1), b1, b2, x2.stride(-1), gamma.stride(0), r1a, r2a),
        (block,),
    )
    return out
