# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""探针对：多趟归约（载体：大规模 max 归约）。

出具 FA-PAT-MULTIPASS-REDUCE 的实证结论：
  朴素版 —— 单 program 一趟扫完全部 1M 元素（归约维超出单趟寄存器
            容量时只能串行循环）；
  手法版 —— 两级：第一级 256 个 program 各归约一个块写中间数组，
            第二级一个 program 归并 256 个部分最大值。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_PARTS = 256


@triton.jit
def _naive_kernel(x_ptr, o_ptr, n, block: tl.constexpr):
    acc = float("-inf")
    for k0 in range(0, n, block):
        offs = k0 + tl.arange(0, block)
        x = tl.load(x_ptr + offs, mask=offs < n, other=float("-inf"))
        acc = tl.maximum(acc, tl.max(x, axis=0))
    tl.store(o_ptr, acc)


@triton.jit
def _partial_kernel(x_ptr, p_ptr, n, part, block: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * part
    acc = float("-inf")
    for k0 in range(0, part, block):
        offs = base + k0 + tl.arange(0, block)
        x = tl.load(x_ptr + offs, mask=offs < n, other=float("-inf"))
        acc = tl.maximum(acc, tl.max(x, axis=0))
    tl.store(p_ptr + pid, acc)


@triton.jit
def _merge_kernel(p_ptr, o_ptr, parts, block: tl.constexpr):
    offs = tl.arange(0, block)
    x = tl.load(p_ptr + offs, mask=offs < parts, other=float("-inf"))
    tl.store(o_ptr, tl.max(x, axis=0))


def make_inputs(shape, device="cpu", seed=20261013):
    n = shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, generator=g).to(device)
    return (x,)


def run_naive(x, *_):
    out = torch.empty((), dtype=x.dtype, device=x.device)
    _naive_kernel[(1,)](x, out, x.numel(), block=1024)
    return out


def run_pattern(x, *_):
    n = x.numel()
    part = (n + _PARTS - 1) // _PARTS
    parts = torch.empty((_PARTS,), dtype=x.dtype, device=x.device)
    out = torch.empty((), dtype=x.dtype, device=x.device)
    _partial_kernel[(_PARTS,)](x, parts, n, part, block=1024)
    _merge_kernel[(1,)](parts, out, _PARTS, block=_PARTS)
    return out


def torch_ref(x, *_):
    return x.max()
