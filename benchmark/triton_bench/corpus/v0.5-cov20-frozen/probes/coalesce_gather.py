# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""探针对：聚集访存（载体：大规模 gather）。

出具 FA-PAT-COALESCE-GATHER 的实证结论：
  朴素版 —— BLOCK=1，每元素一个 program、纯随机访存（每次访存
            独立发射，无合并）；
  手法版 —— BLOCK=1024，块内访存合并（相邻 program 的地址段在同一
            批访存请求里被硬件合并）。

注：冻结用例的索引由种子随机生成（全范围均匀），本探针同款分布——
「窄窗口索引」的构造需输入构造层扩展（已在方案文档记录为裁剪项），
实证的是块化 gather 的合并访存收益。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_kernel(x_ptr, i_ptr, o_ptr, block: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.load(i_ptr + pid).to(tl.int64)
    tl.store(o_ptr + pid, tl.load(x_ptr + idx))


@triton.jit
def _blocked_kernel(x_ptr, i_ptr, o_ptr, n, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    idx = tl.load(i_ptr + offs, mask=mask, other=0).to(tl.int64)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x, mask=mask)


def make_inputs(shape, device="cpu", seed=20261013):
    n = shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, generator=g).to(device)
    idx = torch.randint(0, n, (n,), generator=g, dtype=torch.int64).to(device)
    return (x, idx)


def run_naive(x, idx):
    out = torch.empty_like(x)
    _naive_kernel[(x.numel(),)](x, idx, out, block=1)
    return out


def run_pattern(x, idx):
    out = torch.empty_like(x)
    n = x.numel()
    _blocked_kernel[(triton.cdiv(n, 1024),)](x, idx, out, n, block=1024)
    return out


def torch_ref(x, idx):
    return x[idx]
