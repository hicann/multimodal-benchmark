# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""探针对：收缩类分块 + 偏置向量广播（载体：分组矩阵乘）。

出具两条原子的实证结论：
  FA-PAT-TILE-GEMM   —— 朴素版每输出元素一个 program、K 标量循环、
                        bias 逐元素读（无分块、无访存合并）；
                        手法版 BM×BN×BK 分块 tl.dot + bias 向量读。
  FA-PAT-VEC-BROADCAST —— 同一对内核的 bias 路径：朴素版 bias 展开成
                        [M,N] 矩阵逐元素读；手法版 bias[N] 向量一次读、
                        块内广播驻留寄存器。

探针两版算同一道题；跑批脚本（scripts/probe_validation.py）负责
torch 对拍与计时，本文件只提供两版入口与输入构造。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_kernel(x_ptr, w_ptr, be_ptr, o_ptr, m, k, n,
                  block: tl.constexpr):
    """每输出元素一个 program：K 标量累加，bias 从展开矩阵逐元素读。"""
    pid = tl.program_id(0)
    n_value = pid % n
    m_value = pid // n
    acc = 0.0
    for k_value in range(0, k):
        a = tl.load(x_ptr + m_value * k + k_value)
        b = tl.load(w_ptr + k_value * n + n_value)
        acc += a * b
    bias = tl.load(be_ptr + m_value * n + n_value)   # bias 展开矩阵：逐元素读
    tl.store(o_ptr + m_value * n + n_value, acc + bias)


@triton.jit
def _tiled_kernel(x_ptr, w_ptr, b_ptr, o_ptr, m, k, n,
                  bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    """分块 tl.dot + bias 向量读（块内广播驻留）。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * bm + tl.arange(0, bm)
    rn = pid_n * bn + tl.arange(0, bn)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for k0 in range(0, k, bk):
        rk = k0 + tl.arange(0, bk)
        a = tl.load(x_ptr + rm[:, None] * k + rk[None, :],
                    mask=(rm[:, None] < m) & (rk[None, :] < k), other=0.0)
        b = tl.load(w_ptr + rk[:, None] * n + rn[None, :],
                    mask=(rk[:, None] < k) & (rn[None, :] < n), other=0.0)
        acc += tl.dot(a, b)
    bias = tl.load(b_ptr + rn, mask=rn < n, other=0.0)
    y = acc + bias[None, :]
    tl.store(o_ptr + rm[:, None] * n + rn[None, :],
             y.to(o_ptr.dtype.element_ty),
             mask=(rm[:, None] < m) & (rn[None, :] < n))


# GEMM 族对拍容差（文档化的平台口径）：f32 分块 dot 与 torch matmul
# 在 K≥2048 时有 1e-4~1e-3 量级的求和顺序噪声（2026-10-13 真机实测，
# 近零输出被抵消放大），1e-4 的对拍容差会把合法写法判死
CROSS_TOL = (5e-3, 5e-3)


def make_inputs(shape, device="cpu", seed=20261013):
    m, k = shape
    n = 64
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m * k, generator=g).reshape(m, k).to(device)
    w = torch.randn(k * n, generator=g).reshape(k, n).to(device)
    bias = torch.randn(n, generator=g).to(device)
    # 朴素版的 bias 展开矩阵（[M, N]，构造一次，计时中只读）
    bias_expanded = bias.unsqueeze(0).expand(m, n).contiguous().to(device)
    return x, w, bias, bias_expanded


def run_naive(x, w, bias, bias_expanded):
    m, k = x.shape
    n = w.shape[1]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _naive_kernel[(m * n,)](x, w, bias_expanded, out, m, k, n, block=1)
    return out


def run_pattern(x, w, bias, bias_expanded):
    m, k = x.shape
    n = w.shape[1]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    _tiled_kernel[grid](x, w, bias, out, m, k, n, bm=32, bn=32, bk=32)
    return out


def torch_ref(x, w, bias, *_):
    return x.to(torch.float32) @ w.to(torch.float32) \
        + bias.to(torch.float32).unsqueeze(0)
