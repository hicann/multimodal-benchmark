# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""RmsNormLinear 的 triton 实现（合法提交样例）。

y = grouped_matmul(rms_norm(x)·gamma)：归一化直接喂收缩，
归一化后的中间结果不落显存——这正是该融合算子的考点。

两个 kernel：
  1. 行归一化系数：每行 Σx² 分块累加 → scale[row] = rsqrt(Σ/K + eps)
  2. 融合收缩：A 块 load 后在 **fp32 域**乘 scale[row]·gamma[k]，**舍回
     输入精度**（段语义：rms_norm 原语参考的出口是 .to(x.dtype)，组合
     的中间张量本来就是输入精度的——2026-10-13 真机归因：不舍回时
     半精度用例全错，本地模拟与真机误差逐位吻合），再升 fp32 做 dot

分组语义同 GroupedMatmul：group_list 切 x 的行维。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_scale_kernel(x_ptr, s_ptr, rows, cols, eps,
                      xs0, xs1, block: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for k0 in range(0, cols, block):
        offs = k0 + tl.arange(0, block)
        valid = offs < cols
        x = tl.load(x_ptr + row * xs0 + offs * xs1, mask=valid,
                    other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(s_ptr + row, 1.0 / tl.sqrt(acc / cols + eps))


@triton.jit
def _rms_linear_kernel(x_ptr, g_ptr, w_ptr, b_ptr, s_ptr, o_ptr, geometry, configuration: tl.constexpr):
    m, k, n, xs0, xs1 = geometry
    scale_fused, bk, bm, bn = configuration
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * bm + tl.arange(0, bm)
    rn = pid_n * bn + tl.arange(0, bn)
    row_scale = tl.load(s_ptr + rm, mask=rm < m, other=0.0)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for k0 in range(0, k, bk):
        rk = k0 + tl.arange(0, bk)
        a = tl.load(
            x_ptr + rm[:, None] * xs0 + rk[None, :] * xs1, mask=(rm[:, None] < m) & (rk[None, :] < k), other=0.0
        )
        if scale_fused:
            gk = tl.load(g_ptr + rk, mask=rk < k, other=0.0).to(tl.float32)
            a32 = a.to(tl.float32) * row_scale[:, None] * gk[None, :]
            a_mid = a32.to(x_ptr.dtype.element_ty)
            w = tl.load(w_ptr + rk[:, None] * n + rn[None, :], mask=(rk[:, None] < k) & (rn[None, :] < n), other=0.0)
            acc += tl.dot(a_mid, w)
        else:
            w = tl.load(w_ptr + rk[:, None] * n + rn[None, :], mask=(rk[:, None] < k) & (rn[None, :] < n), other=0.0)
            acc += tl.dot(a, w)
    bias = tl.load(b_ptr + rn, mask=rn < n, other=0.0).to(tl.float32)
    y = acc + bias[None, :]
    tl.store(
        o_ptr + rm[:, None] * n + rn[None, :], y.to(o_ptr.dtype.element_ty), mask=(rm[:, None] < m) & (rn[None, :] < n)
    )


def rms_norm_linear(x, gamma, weight, bias, group_list, epsilon=1e-6):
    if x.dim() != 2:
        raise NotImplementedError(f"样例只实现 rank-2，实际 rank={x.dim()}")
    m, k = x.shape
    n = weight.shape[-1]
    if gamma.shape[0] != k:
        raise NotImplementedError("gamma 长度须等于 K")
    if sum(int(g) for g in group_list) != m:
        raise NotImplementedError("group_list 之和必须等于 M")

    # 段 1：逐行归一化系数（scale 逐行一个标量，不算「中间结果落盘」——
    # 落盘的是 K 维全量中间张量，融合省的是那一个）
    scale = torch.empty((m,), dtype=torch.float32, device=x.device)
    block = 1024
    _rms_scale_kernel[(m,)](x, scale, m, k, float(epsilon),
                            x.stride(0), x.stride(1), block=block)

    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    start = 0
    for g, cnt in enumerate(group_list):
        cnt = int(cnt)
        xg = x[start:start + cnt]
        wg = weight[g]
        bg = bias[g]
        og = out[start:start + cnt]
        sg = scale[start:start + cnt]
        grid = (triton.cdiv(cnt, 32), triton.cdiv(n, 32))
        _rms_linear_kernel[grid](xg, gamma, wg, bg, sg, og, (cnt, k, n, xg.stride(0), xg.stride(1)), (True, 32, 32, 32))
        start += cnt
    return out
