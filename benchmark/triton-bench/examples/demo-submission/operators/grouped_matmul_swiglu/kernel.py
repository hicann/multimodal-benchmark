# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GroupedMatmulSwiGlu 的 triton 实现（合法提交样例）。

y = swiglu(grouped_matmul(x))：矩阵乘直接喂门控，[M, N] 的中间
结果不落显存——每个 program 同时算出配对的两列块（j 与 j+N/2），
在寄存器里完成 silu(a)·b。

实现形态（2026-10-13 真机三轮定案，两个内核 + 中间结果过显存）：
  段语义要求矩阵乘段的出口**舍回输入精度**再喂 silu（组合参考 =
  原语参考顺序调用，grouped_matmul 原语的出口是 .to(x.dtype)）。
  该后端会**静默消除** f32→输入精度→f32 的舍回对——第二轮以指针
  元素类型为目标仍逐位复现未舍回的误差（f16 abs 0.25），第三轮
  rms_norm_linear 证明「以输入 dtype 参与 tl.dot」可强制舍入生效，
  但 silu 段是逐元素、没有 dot 可挂。显存往返（以输入精度存中间
  缓冲再读回）是唯一不被优化器消除的舍回形式。C13 已定案：融合
  算子不强制单内核——两内核串联的中间落显存代价由评分测量，
  这正是该维度要考察的性能差距。

要点：
  - silu 用 sigmoid 定义写：a / (1 + exp(-a))（口径同 swi_glu/kernel.py，
    tanh/sigmoid 内建未在本环境验证）
  - 输出维减半：段 1 的列块数按 N/2 计，每块算两个 dot
  - 分组语义同 GroupedMatmul；fp32 累加器
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gmm_stage_kernel(x_ptr, w_ptr, b_ptr, m_ptr, geometry, configuration: tl.constexpr):
    m, k, n, xs0, xs1 = geometry
    bk, bm, bn = configuration
    "段 1：分组矩阵乘 → 中间结果以输入精度落 scratch（物理舍回）。"
    half = n // 2
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * bm + tl.arange(0, bm)
    rn = pid_n * bn + tl.arange(0, bn)
    acc_a = tl.zeros((bm, bn), dtype=tl.float32)
    acc_b = tl.zeros((bm, bn), dtype=tl.float32)
    for k0 in range(0, k, bk):
        rk = k0 + tl.arange(0, bk)
        a = tl.load(
            x_ptr + rm[:, None] * xs0 + rk[None, :] * xs1, mask=(rm[:, None] < m) & (rk[None, :] < k), other=0.0
        )
        wa = tl.load(w_ptr + rk[:, None] * n + rn[None, :], mask=(rk[:, None] < k) & (rn[None, :] < n), other=0.0)
        wb = tl.load(
            w_ptr + rk[:, None] * n + (rn[None, :] + half), mask=(rk[:, None] < k) & (rn[None, :] < half), other=0.0
        )
        acc_a += tl.dot(a, wa)
        acc_b += tl.dot(a, wb)
    bias_a = tl.load(b_ptr + rn, mask=rn < half, other=0.0).to(tl.float32)
    bias_b = tl.load(b_ptr + rn + half, mask=rn < half, other=0.0).to(tl.float32)
    tl.store(
        m_ptr + rm[:, None] * n + rn[None, :],
        (acc_a + bias_a[None, :]).to(m_ptr.dtype.element_ty),
        mask=(rm[:, None] < m) & (rn[None, :] < half),
    )
    tl.store(
        m_ptr + rm[:, None] * n + (rn[None, :] + half),
        (acc_b + bias_b[None, :]).to(m_ptr.dtype.element_ty),
        mask=(rm[:, None] < m) & (rn[None, :] < half),
    )


@triton.jit
def _swiglu_stage_kernel(m_ptr, o_ptr, rows, n, half,
                         block: tl.constexpr):
    """段 2：silu(a)·b，读回以输入精度落盘的中间结果（已物理舍回）。"""
    p = tl.program_id(0)
    offs = tl.arange(0, block)
    valid = offs < half
    a = tl.load(m_ptr + p * n + offs, mask=valid,
                other=0.0).to(tl.float32)
    b = tl.load(m_ptr + p * n + offs + half, mask=valid,
                other=0.0).to(tl.float32)
    y = a / (1.0 + tl.exp(-a)) * b
    tl.store(o_ptr + p * half + offs,
             y.to(o_ptr.dtype.element_ty), mask=valid)


def grouped_matmul_swiglu(x, weight, bias, group_list):
    if x.dim() != 2:
        raise NotImplementedError(f"样例只实现 rank-2，实际 rank={x.dim()}")
    m, k = x.shape
    n = weight.shape[-1]
    if n % 2 != 0:
        raise NotImplementedError(f"N={n} 必须为偶数（swiglu 对半拆分）")
    if sum(int(g) for g in group_list) != m:
        raise NotImplementedError("group_list 之和必须等于 M")
    half = n // 2
    out = torch.empty((m, half), dtype=x.dtype, device=x.device)
    # 中间缓冲以输入精度分配：段 1 的 store 即完成段语义的舍回
    mid = torch.empty((m, n), dtype=x.dtype, device=x.device)

    start = 0
    for g, cnt in enumerate(group_list):
        cnt = int(cnt)
        xg = x[start:start + cnt]
        wg = weight[g]
        bg = bias[g]
        mg = mid[start:start + cnt]
        grid = (triton.cdiv(cnt, 32), triton.cdiv(half, 32))
        _gmm_stage_kernel[grid](xg, wg, bg, mg, (cnt, k, n, xg.stride(0), xg.stride(1)), (32, 32, 32))
        start += cnt
    # 段 2 逐行（half ≤ 32 单块）；整批一次启动，不分组（逐元素无组语义）
    _swiglu_stage_kernel[(m,)](mid, out, m, n, half,
                               block=max(16, triton.next_power_of_2(half)))
    return out
