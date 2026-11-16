# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""探针对：沿归约维切分并行 split-K（载体：高瘦矩阵乘）。

出具 FA-PAT-SPLIT-K 的实证结论：
  朴素版 —— M/N 二维网格单趟 K 循环（M=64、N=64 时并行度只有 4 块，
            K=65536 的长循环串行在每个 program 里）；
  手法版 —— K 拆成 8 段并行（grid 扩 8 倍），部分和落盘后按固定
            顺序合并（确定性两段式——生产内核求确定性的标准 split-K
            形态；2026-10-13 第三轮由原子加改为两段式：原子合并顺序
            由硬件调度，对拍噪声不可控）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_kernel(x_ptr, w_ptr, b_ptr, o_ptr, m, k, n,
                  bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * bm + tl.arange(0, bm)
    rn = pid_n * bn + tl.arange(0, bn)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for k0 in range(0, k, bk):
        rk = k0 + tl.arange(0, bk)
        a = tl.load(x_ptr + rm[:, None] * k + rk[None, :],
                    mask=(rk[None, :] < k), other=0.0)
        b = tl.load(w_ptr + rk[:, None] * n + rn[None, :],
                    mask=(rk[:, None] < k) & (rn[None, :] < n), other=0.0)
        acc += tl.dot(a, b)
    bias = tl.load(b_ptr + rn, mask=rn < n, other=0.0)
    tl.store(o_ptr + rm[:, None] * n + rn[None, :],
             (acc + bias[None, :]).to(o_ptr.dtype.element_ty))


@triton.jit
def _splitk_partial_kernel(x_ptr, w_ptr, p_ptr, m, k, n,
                           split: tl.constexpr,
                           bm: tl.constexpr, bn: tl.constexpr,
                           bk: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)          # K 段号：并行度扩 SPLIT 倍
    k_chunk = k // split
    k_base = pid_k * k_chunk
    rm = pid_m * bm + tl.arange(0, bm)
    rn = pid_n * bn + tl.arange(0, bn)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for k0 in range(0, k_chunk, bk):
        rk = k_base + k0 + tl.arange(0, bk)
        a = tl.load(x_ptr + rm[:, None] * k + rk[None, :],
                    mask=(rk[None, :] < k), other=0.0)
        b = tl.load(w_ptr + rk[:, None] * n + rn[None, :],
                    mask=(rk[:, None] < k) & (rn[None, :] < n),
                    other=0.0)
        acc += tl.dot(a, b)
    tl.store(p_ptr + pid_k * (m * n) + rm[:, None] * n + rn[None, :],
             acc, mask=(rm[:, None] < m) & (rn[None, :] < n))


@triton.jit
def _splitk_combine_kernel(p_ptr, b_ptr, o_ptr, m, n,
                           split: tl.constexpr,
                           bm: tl.constexpr, bn: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * bm + tl.arange(0, bm)
    rn = pid_n * bn + tl.arange(0, bn)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for s in range(0, split):         # 固定顺序合并：两遍执行逐位一致
        acc += tl.load(p_ptr + s * (m * n) + rm[:, None] * n + rn[None, :],
                       mask=(rm[:, None] < m) & (rn[None, :] < n),
                       other=0.0)
    bias = tl.load(b_ptr + rn, mask=rn < n, other=0.0)
    tl.store(o_ptr + rm[:, None] * n + rn[None, :],
             (acc + bias[None, :]).to(o_ptr.dtype.element_ty),
             mask=(rm[:, None] < m) & (rn[None, :] < n))


def make_inputs(shape, device="cpu", seed=20261013):
    m, k = shape
    n = 64
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m * k, generator=g).reshape(m, k).to(device)
    w = torch.randn(k * n, generator=g).reshape(k, n).to(device)
    bias = torch.randn(n, generator=g).to(device)
    return x, w, bias


def run_naive(x, w, bias):
    m, k = x.shape
    n = w.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    _naive_kernel[grid](x, w, bias, out, m, k, n, bm=32, bn=32, bk=32)
    return out


def run_pattern(x, w, bias):
    m, k = x.shape
    n = w.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    part = torch.empty((8, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    _splitk_partial_kernel[grid + (8,)](x, w, part, m, k, n,
                                        split=8, bm=32, bn=32, bk=32)
    _splitk_combine_kernel[grid](part, bias, out, m, n,
                                 split=8, bm=32, bn=32)
    return out


def torch_ref(x, w, bias):
    return x.to(torch.float32) @ w.to(torch.float32) \
        + bias.to(torch.float32).unsqueeze(0)


# GEMM 族对拍容差（文档化的平台口径）：长 K（本探针 65536）下，任何
# 跨实现的 f32 矩阵乘都带求和顺序噪声——dot 归因探针实测 K=65536 时
# 与 torch 分歧 abs 4.7e-3 / rel 2.1e-3（rms_norm_linear-f32-X56 同
# 形状同款内核印证）；K 段重排后再合并会再叠加同量级噪声。1e-2 覆盖
# 实测值并留 ~2× 余量。跑批脚本按 CROSS_TOL 查表
CROSS_TOL = (1e-2, 1e-2)
