# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""f32 tl.dot 与 torch matmul 的精度分歧归因探针（评测方诊断资产）。

背景（2026-10-13 两轮真机实测）：三个融合 GEMM 算子的大 K f32 用例
WRONG（grouped_matmul_swiglu / transpose_linear [2048,2048]、
rms_norm_linear [64,65536]），本地纯 f32 复刻内核数学与参考差异为 0
——分歧是 triton 分块 dot 与 torch matmul 各自的**求和顺序**噪声，
随 K 增长、近零输出被抵消放大。tile_gemm 探针 (2048,2048) 同款内核
在 1e-4 内与 torch 一致（顺序噪声在该量级内），排除了「dot 精度损坏」
的强假设。本探针量化分歧随 K 与分块 BK 的变化，为「容差覆盖登记
（附归因证据）或内核分块调整」提供数据。

模块化存放的原因：triton 内核必须能从模块级全局解析 tl——放在脚本
main() 里定义的内核拿不到 tl（2026-10-13 第二轮真机该探针即崩在此）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dot_kernel(a_ptr, w_ptr, o_ptr, m, k, n,
                bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    # 与 demo 融合 GEMM 内核同款惯用法：分块循环 tl.dot、fp32 累加器
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * bm + tl.arange(0, bm)
    rn = pid_n * bn + tl.arange(0, bn)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for k0 in range(0, k, bk):
        rk = k0 + tl.arange(0, bk)
        a = tl.load(a_ptr + rm[:, None] * k + rk[None, :],
                    mask=(rm[:, None] < m) & (rk[None, :] < k),
                    other=0.0)
        w = tl.load(w_ptr + rk[:, None] * n + rn[None, :],
                    mask=(rk[:, None] < k) & (rn[None, :] < n),
                    other=0.0)
        acc += tl.dot(a, w)
    tl.store(o_ptr + rm[:, None] * n + rn[None, :],
             acc, mask=(rm[:, None] < m) & (rn[None, :] < n))


def make_inputs(k, m=64, n=64, device="cpu", seed=20261013):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(m * k, generator=g).reshape(m, k).to(device)
    w = torch.randn(k * n, generator=g).reshape(k, n).to(device)
    return a, w


def run_bk(a, w, bk):
    m, k = a.shape
    n = w.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    _dot_kernel[grid](a, w, out, m, k, n, bm=32, bn=32, bk=bk)
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    return out


def torch_ref(a, w):
    return torch.matmul(a, w)
