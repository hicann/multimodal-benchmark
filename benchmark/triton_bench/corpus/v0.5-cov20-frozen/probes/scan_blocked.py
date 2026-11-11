# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""探针对：分块两趟扫描（载体：大规模前缀最小值扫描）。

出具 FA-PAT-SCAN-BLOCKED 的实证结论：
  朴素版 —— 单 program 顺序标量扫描（前缀扫描的串行本质形态；
            滚动最小值逐元素推进）；
  手法版 —— Hillis-Steele 对数步并行扫描：步长每轮翻倍，全量元素
            并行做 min 合并，log2(N)≈20 **轮 kernel** 完成。
            每轮一个 launch（轮间全局同步是正确性要求——网格内
            无全局屏障，单 kernel 多轮是错的）；ping-pong 读写避免
            本轮读到本轮已写的值。

计时注记：朴素版是串行本质，1M 载体上单次可达秒级——本探针在
跑批脚本中按「rep=1」计时（其余探针 rep=3）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_kernel(x_ptr, o_ptr, n):
    acc = float("inf")
    for k in range(0, n):
        v = tl.load(x_ptr + k)
        acc = tl.minimum(acc, v)
        tl.store(o_ptr + k, acc)


@triton.jit
def _hs_step(src_ptr, dst_ptr, n, stride, block: tl.constexpr):
    offs = tl.program_id(0) * block + tl.arange(0, block)
    mask = offs < n
    v = tl.load(src_ptr + offs, mask=mask, other=float("inf"))
    other = tl.load(src_ptr + tl.maximum(offs - stride, 0),
                    mask=mask & (offs >= stride), other=float("inf"))
    tl.store(dst_ptr + offs, tl.minimum(v, other), mask=mask)


def make_inputs(shape, device="cpu", seed=20261013):
    n = shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, generator=g).to(device)
    return (x,)


def run_naive(x, *_):
    out = torch.empty_like(x)
    _naive_kernel[(1,)](x, out, x.numel())
    return out


def run_pattern(x, *_):
    n = x.numel()
    src = x.clone()
    dst = torch.empty_like(x)
    grid = (triton.cdiv(n, 1024),)
    for s in range(20):
        stride = 1 << s
        _hs_step[grid](src, dst, n, stride, block=1024)
        src, dst = dst, src
    return src


def torch_ref(x, *_):
    # dim 必须显式：torch 2.9 上 torch.cummin(x) 单参形式报
    # "invalid combination of arguments"（2026-10-13 真机第二轮实测）
    return torch.cummin(x, 0)[0]
