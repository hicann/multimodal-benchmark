# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""探针对：向量化访存与边界掩码（载体：逐元素 exp）。

出具两条原子的实证结论：
  FA-PAT-VEC-LOAD  —— 朴素版 BLOCK=1（每元素一个 program、标量访存）；
                       手法版 BLOCK=1024（块内向量访存）。对齐形状测。
  FA-PAT-MASK-TAIL —— 同一对内核在非对齐形状（99971，不整除块宽）上
                       测：手法版满块 + 边界掩码的代价远小于放弃向量化。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_kernel(x_ptr, o_ptr, n, log_base, block: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return
    x = tl.load(x_ptr + pid).to(tl.float32)
    tl.store(o_ptr + pid, tl.exp(x * log_base).to(o_ptr.dtype.element_ty))


@triton.jit
def _vector_kernel(x_ptr, o_ptr, n, log_base, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, tl.exp(x * log_base).to(o_ptr.dtype.element_ty),
             mask=mask)


def make_inputs(shape, device="cpu", seed=20261013):
    n = shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, generator=g).to(device)
    return (x,)


def run_naive(x, *_):
    out = torch.empty_like(x)
    n = x.numel()
    _naive_kernel[(n,)](x, out, n, 1.0, block=1)
    return out


def run_pattern(x, *_):
    out = torch.empty_like(x)
    n = x.numel()
    _vector_kernel[(triton.cdiv(n, 1024),)](x, out, n, 1.0, block=1024)
    return out


def torch_ref(x, *_):
    return torch.exp(x.to(torch.float32))
