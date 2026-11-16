# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GatherExp 的 triton 实现（合法提交样例）：y = exp(x.gather(dim, index))。

要点：
  - 融合算子的样例内核是**锚点载体**（要求全部用例判对），按 stride
    寻址实现，支持任意 dim 与非连续 x——gather 的语义就是寻址本身
  - base != -1 时用 exp(z·ln(base)) 算幂（口径同 exp/kernel.py），
    不调 torch 内置 gather/exp（SUB-STR-001）
  - index 与输出同形且由评测方构造为连续；x 用 strides 寻址
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_exp_kernel(x_ptr, i_ptr, o_ptr, geometry, configuration: tl.constexpr):
    n, xs0, xs1, xs2, d0, d1, d2, dim, scale, shift, log_base = geometry
    (block,) = configuration
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    i2 = offs % d2
    r = offs // d2
    i1 = r % d1
    r = r // d1
    i0 = r % d0
    idx = tl.load(i_ptr + offs, mask=mask, other=0).to(tl.int64)
    x_off = (
        tl.where(dim == 0, idx * xs0, i0.to(tl.int64) * xs0)
        + tl.where(dim == 1, idx * xs1, i1.to(tl.int64) * xs1)
        + tl.where(dim == 2, idx * xs2, i2.to(tl.int64) * xs2)
    )
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    z = (x * scale + shift) * log_base
    y = tl.exp(z)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


def gather_exp(x, index, dim=0, base=-1.0, scale=1.0, shift=0.0):
    if x.dim() > 3:
        raise NotImplementedError(f"样例只实现 rank≤3，实际 rank={x.dim()}")
    if index.dtype not in (torch.int32, torch.int64):
        raise NotImplementedError(
            f"index 预期为 int32/int64，实际 {index.dtype}")
    if not index.is_contiguous():
        raise NotImplementedError("index 张量预期为连续布局")
    d = dim if dim >= 0 else dim + x.dim()
    if not (0 <= d < x.dim()):
        raise NotImplementedError(f"dim={dim} 超出 rank={x.dim()}")

    out = torch.empty(index.shape, dtype=x.dtype, device=x.device)
    n = out.numel()
    d0, d1, d2 = (x.shape + (1,) * (3 - x.dim()))
    xs = (x.stride() + (0,) * (3 - x.dim()))
    log_base = 1.0 if (base is None or math.isclose(base, -1.0, rel_tol=0.0, abs_tol=0.0)) else math.log(base)
    block = 1024
    _gather_exp_kernel[triton.cdiv(n, block),](
        x, index, out, (n, xs[0], xs[1], xs[2], d0, d1, d2, d, float(scale), float(shift), float(log_base)), (block,)
    )
    return out
