# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TransposeLinear 的 triton 实现（合法提交样例）。

y = grouped_matmul(x^T)：布局变换直接喂收缩，转置不落显存——
不是「先转置成新张量再乘」，而是把 x 的两个维度步长**互换**后
直接作为收缩的行/列步长：转置段在寻址层完成，这正是融合的意义。

要点：
  - 分组语义：group_list 切的是**收缩的行维**，即 x 的第 1 维（K）
  - 每个 program 一个 (行块, 列块)：K 分块累加 tl.dot，fp32 累加器
  - x 按步长寻址（互换步长即转置；切片 view 保留步长，不搬数据）
  - weight/bias 按组切（评测方构造为连续）；输出行块按组写入完整 y
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_linear_kernel(x_ptr, w_ptr, b_ptr, o_ptr, geometry, configuration: tl.constexpr):
    m, k, n, ar, ac = geometry
    bk, bm, bn = configuration
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * bm + tl.arange(0, bm)
    rn = pid_n * bn + tl.arange(0, bn)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for k0 in range(0, k, bk):
        rk = k0 + tl.arange(0, bk)
        a = tl.load(x_ptr + rm[:, None] * ar + rk[None, :] * ac, mask=(rm[:, None] < m) & (rk[None, :] < k), other=0.0)
        w = tl.load(w_ptr + rk[:, None] * n + rn[None, :], mask=(rk[:, None] < k) & (rn[None, :] < n), other=0.0)
        acc += tl.dot(a, w)
    bias = tl.load(b_ptr + rn, mask=rn < n, other=0.0).to(tl.float32)
    y = acc + bias[None, :]
    tl.store(
        o_ptr + rm[:, None] * n + rn[None, :], y.to(o_ptr.dtype.element_ty), mask=(rm[:, None] < m) & (rn[None, :] < n)
    )


def transpose_linear(x, weight, bias, group_list):
    if x.dim() != 2:
        raise NotImplementedError(f"样例只实现 rank-2，实际 rank={x.dim()}")
    m, k = x.shape
    n = weight.shape[-1]
    if sum(int(g) for g in group_list) != k:
        raise NotImplementedError("group_list 之和必须等于 K（shape[1]）")
    out = torch.empty((k, n), dtype=x.dtype, device=x.device)

    start = 0
    for g, cnt in enumerate(group_list):
        cnt = int(cnt)
        # 切片是视图（不搬数据）：x 的列切片作为收缩的行堆
        xg = x[:, start:start + cnt]
        wg = weight[g]
        bg = bias[g]
        og = out[start:start + cnt]
        # 转置段在寻址层完成：收缩的「行」沿 x 第 1 维（步长互换）
        grid = (triton.cdiv(cnt, 32), triton.cdiv(n, 32))
        _transpose_linear_kernel[grid](xg, wg, bg, og, (cnt, m, n, xg.stride(1), xg.stride(0)), (32, 32, 32))
        start += cnt
    return out
