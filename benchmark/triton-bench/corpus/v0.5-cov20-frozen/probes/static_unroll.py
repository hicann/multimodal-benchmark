# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""探针对：静态循环展开（载体：8 组分组矩阵乘）。

出具 FA-CTRL-STATIC-UNROLL 的实证结论：
  朴素版 —— 组循环用 range（动态循环，运行期迭代）；
  手法版 —— 组循环用 tl.static_range（编译期完全展开，组间指令调度合并）。

设计更正（2026-10-13 真机第二轮）：初版把 tl.static_range 用在 **K 循环**
上——K 是运行期标量，static_range 要求编译期边界，前端直接
CompilationError。本原子考察的本就是「group_list 长度 ≤ 8 → 小循环次数
可静态展开」：组数是编译期常量（8），展开的应该是**组循环**。

诚实性注记：本对探针的收益**可能很小**（展开优化是否生效取决于
编译器后端）。这正是实证机制存在的理由——实测说话，不达标就如实
降级为「声明性考察」，不装点覆盖率。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# triton-ascend 若无 static_range，回落 range（此时两版同构，实测无差，
# 结论如实记「声明性考察」；不假装测了展开）
_UNROLL = getattr(tl, "static_range", range)


@triton.jit
def _store_group(x_ptr, w_ptr, b_ptr, o_ptr, geometry, configuration: tl.constexpr):
    group, row_count, inner_count, column_count, row_program, column_program = geometry
    row_tile, column_tile, inner_tile = configuration
    rows = row_program * row_tile + tl.arange(0, row_tile)
    columns = column_program * column_tile + tl.arange(0, column_tile)
    accumulator = tl.zeros((row_tile, column_tile), dtype=tl.float32)
    for inner_start in range(0, inner_count, inner_tile):
        inner = inner_start + tl.arange(0, inner_tile)
        left = tl.load(
            x_ptr + group * (row_count * inner_count) + rows[:, None] * inner_count + inner[None, :],
            mask=inner[None, :] < inner_count, other=0.0,
        )
        right = tl.load(
            w_ptr + group * (inner_count * column_count) + inner[:, None] * column_count + columns[None, :],
            mask=(inner[:, None] < inner_count) & (columns[None, :] < column_count), other=0.0,
        )
        accumulator += tl.dot(left, right)
    bias = tl.load(b_ptr + group * column_count + columns, mask=columns < column_count, other=0.0)
    tl.store(
        o_ptr + group * (row_count * column_count) + rows[:, None] * column_count + columns[None, :],
        (accumulator + bias[None, :]).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _dynamic_kernel(x_ptr, w_ptr, b_ptr, o_ptr, geometry, configuration: tl.constexpr):
    m, k, n = geometry
    bm, bn, bk, groups = configuration
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    for g in range(groups):
        _store_group(x_ptr, w_ptr, b_ptr, o_ptr, (g, m, k, n, pid_m, pid_n), (bm, bn, bk))


@triton.jit
def _static_kernel(x_ptr, w_ptr, b_ptr, o_ptr, geometry, configuration: tl.constexpr):
    m, k, n = geometry
    bm, bn, bk, groups = configuration
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    for g in _UNROLL(groups):
        _store_group(x_ptr, w_ptr, b_ptr, o_ptr, (g, m, k, n, pid_m, pid_n), (bm, bn, bk))


# GEMM 族对拍容差：跨实现的 f32 矩阵乘在长 K 下有求和顺序噪声
# （2026-10-13 真机实测 K≥2048 时与 torch 分歧 1e-4~1e-3 量级、近零
# 输出被抵消放大），1e-4 的对拍容差会把合法写法判死；5e-3 为文档化
# 的平台口径，跑批脚本按此查表
CROSS_TOL = (5e-3, 5e-3)


def make_inputs(shape=(64, 512), device="cpu", seed=20261013):
    """8 组 × [M, K]：group 数固定 8（FA-CTRL-STATIC-UNROLL 的谓词上界）。"""
    m, k = shape
    n = 64
    groups = 8
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(groups * m * k, generator=g).reshape(groups, m, k).to(device)
    w = torch.randn(groups * k * n, generator=g).reshape(groups, k, n).to(device)
    b = torch.randn(groups * n, generator=g).reshape(groups, n).to(device)
    return x, w, b


def _run(kernel, x, w, b):
    groups, m, k = x.shape
    n = w.shape[-1]
    out = torch.empty((groups, m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    kernel[grid](x, w, b, out, (m, k, n), (32, 32, 32, groups))
    return out


def run_naive(x, w, b):
    return _run(_dynamic_kernel, x, w, b)


def run_pattern(x, w, b):
    return _run(_static_kernel, x, w, b)


def torch_ref(x, w, b):
    return torch.bmm(x.to(torch.float32), w.to(torch.float32)) \
        + b.to(torch.float32).unsqueeze(1)
