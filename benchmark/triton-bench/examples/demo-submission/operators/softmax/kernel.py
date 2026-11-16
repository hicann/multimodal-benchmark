# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Softmax 的 triton 实现（合法提交样例）。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, o_ptr, row_stride, cols, block: tl.constexpr):
    """每个 program 处理一行，在**最后一维**内做归一化。"""
    row = tl.program_id(0)
    offs = tl.arange(0, block)
    mask = offs < cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask,
                other=float("-inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    p = tl.exp(x - m)
    z = tl.sum(p, axis=0)
    tl.store(o_ptr + row * row_stride + offs,
             (p / z).to(o_ptr.dtype.element_ty), mask=mask)


def softmax(x, dim=-1):
    """沿 dim 做 softmax。

    样例只覆盖 dim=-1：冻结用例里 softmax 的 dim 取规格默认值 -1，
    且 Normalization 类算子不被允许用 permute/transpose 绕道
    （布局变换只对 LayoutTransform 类开放），故此处不做维度搬运。
    曾经的实现把整个张量 flatten 后在**全部元素**上做一次归一化，
    等价于 dim=None，与参考语义不符，2D/3D 用例会被正确判成 WRONG_RESULT。
    """
    d = dim if dim >= 0 else dim + x.dim()
    if d != x.dim() - 1:
        raise NotImplementedError(
            f"样例实现只覆盖 dim=-1，实际 dim={dim}（请自行实现维度搬运）"
        )
    out = torch.empty_like(x)
    cols = x.shape[-1]
    rows = x.numel() // cols
    block = max(1, triton.next_power_of_2(cols))
    _softmax_kernel[(rows,)](x, out, cols, cols, block=block)
    return out
