# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Transpose 的 triton 实现（合法提交样例）。

做法：输出按 perm 后的形状连续排布，kernel 只负责「按重排后的步长取源、顺序写目标」。
布局搬运发生在 kernel 内，而不是调用 torch.permute 后拷贝——
后者虽被布局类白名单放行，但那就等于把核心语义整个交给框架。

维度统一补齐到 8 维（不足补 1、对应步长补 0）：
    这样 kernel 里的下标分解是固定展开的，不依赖动态 rank 的循环。
    v0.4 语料起出现 rank-5 用例（transpose-f32-S140），原 MAX_RANK=4
    的版本在 rank-5 上静默丢掉第 5 维——新语料第一次出手就抓到了它。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_RANK = 8


@triton.jit
def _permute_kernel(x_ptr, o_ptr, geometry, configuration: tl.constexpr):
    n, d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7 = geometry
    (block,) = configuration
    "把输出的 flat 下标分解成多维下标，再按重排后的源步长取数。\n\n    输出自身是连续的，故目标偏移就是 flat 下标本身。\n    从最后一维逐层剥出各维下标（每层一次除余）。\n    "
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    i7 = offs % d7
    r = offs // d7
    i6 = r % d6
    r = r // d6
    i5 = r % d5
    r = r // d5
    i4 = r % d4
    r = r // d4
    i3 = r % d3
    r = r // d3
    i2 = r % d2
    r = r // d2
    i1 = r % d1
    i0 = r // d1
    src = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5 * s5 + i6 * s6 + i7 * s7
    tl.store(o_ptr + offs, tl.load(x_ptr + src, mask=mask), mask=mask)


def transpose(x, perm):
    perm = [int(p) for p in perm]
    out_shape = [x.shape[p] for p in perm]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    n = out.numel()
    dims = list(out_shape) + [1] * (MAX_RANK - len(out_shape))
    # 第 k 维用 x 在 perm[k] 上的步长：输出下标 k 对应源下标 perm[k]
    strides = [x.stride(p) for p in perm] + [0] * (MAX_RANK - len(perm))

    block = 1024
    _permute_kernel[triton.cdiv(n, block),](
        x,
        out,
        (
            n,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            dims[4],
            dims[5],
            dims[6],
            dims[7],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
        ),
        (block,),
    )
    return out
