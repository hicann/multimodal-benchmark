# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Small audited kernels used to exercise implementation coverage obligations.

All arithmetic runs on device. Output allocations and launch dimensions depend
only on public metadata. No evidence declaration in this submission earns credit.
"""
__all__ = ['tl', 'torch', 'triton']

import torch
import triton
import triton.language as tl




@triton.jit
def _witness_neg(x, out, n: tl.constexpr, stride: tl.constexpr, b: tl.constexpr):
    i = tl.program_id(0) * b + tl.arange(0, b)
    x_value = tl.load(x + i * stride, i < n, other=0)
    tl.store(out + i, -x_value, i < n)


@triton.jit
def _witness_neg4(x, out, d0: tl.constexpr, d1: tl.constexpr,
                  d2: tl.constexpr, d3: tl.constexpr, b: tl.constexpr):
    # Physical z packs logical axes 0,1; y is logical axis 2; x tiles axis 3.
    packed = tl.program_id(2)
    i0 = packed // d1
    i1 = packed % d1
    i2 = tl.program_id(1)
    i3 = tl.program_id(0) * b + tl.arange(0, b)
    offset = ((i0 * d1 + i1) * d2 + i2) * d3 + i3
    x_value = tl.load(x + offset, (i0 < d0) & (i3 < d3), other=0)
    tl.store(out + offset, -x_value, (i0 < d0) & (i3 < d3))


@triton.jit
def _witness_neg3(input_ptr, out, d0: tl.constexpr, d1: tl.constexpr,
                  d2: tl.constexpr, stride: tl.constexpr, block: tl.constexpr):
    first = tl.program_id(2)
    second = tl.program_id(1)
    last = tl.program_id(0) * block + tl.arange(0, block)
    offset = (first * d1 + second) * d2 + last
    valid = (first < d0) & (last < d2)
    value = tl.load(input_ptr + offset * stride, valid, other=0)
    tl.store(out + offset, -value, valid)


def neg(x):
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.ndim == 3:
        first, second, last = x.shape
        stride = x.stride(-1)
        if x.stride() != (second * last * stride, last * stride, stride):
            raise ValueError('three-dimensional witness requires a uniform linear stride')
        _witness_neg3[(triton.cdiv(last, 256), second, first)](
            x, out, first, second, last, stride, 256)
    elif x.ndim == 4 and x.is_contiguous():
        d0, d1, d2, d3 = x.shape
        _witness_neg4[(triton.cdiv(d3, 256), d2, d0 * d1)](
            x, out, d0, d1, d2, d3, 256)
    else:
        _witness_neg[(triton.cdiv(x.numel(), 256),)](
            x, out, x.numel(), x.stride(-1), 256)
    return out


@triton.jit
def _witness_masked_softmax(x, mask, out, configuration: tl.constexpr):
    r: tl.constexpr = configuration.value[0]
    l2: tl.constexpr = configuration.value[1]
    xs1: tl.constexpr = configuration.value[2]
    xs2: tl.constexpr = configuration.value[3]
    xsr: tl.constexpr = configuration.value[4]
    ms1: tl.constexpr = configuration.value[5]
    ms2: tl.constexpr = configuration.value[6]
    msr: tl.constexpr = configuration.value[7]
    os1: tl.constexpr = configuration.value[8]
    os2: tl.constexpr = configuration.value[9]
    osr: tl.constexpr = configuration.value[10]
    b: tl.constexpr = configuration.value[11]
    line = tl.program_id(0)
    a = line // l2
    b_value = line % l2
    r_value = tl.arange(0, b)
    value = tl.load(x + a * xs1 + b_value * xs2 + r_value * xsr, r_value < r, other=0).to(tl.float32)
    keep = tl.load(mask + a * ms1 + b_value * ms2 + r_value * msr, r_value < r, other=0)
    masked = tl.where((r_value < r) & (keep != 0), value, -float("inf"))
    e = tl.exp(masked - tl.max(masked, 0))
    result = e / tl.sum(e, 0)
    tl.store(out + a * os1 + b_value * os2 + r_value * osr, result, r_value < r)


def masked_softmax(x, mask, dim):
    dim = dim % x.ndim
    if x.ndim > 3 or x.shape[dim] > 1024 or x.shape != mask.shape:
        raise ValueError('bounded single-pass witness requires rank<=3, axis<=1024')
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    axes = [i for i in range(x.ndim) if i != dim]
    sizes = [1] * (2 - len(axes)) + [x.shape[i] for i in axes]
    xs = [0] * (2 - len(axes)) + [x.stride(i) for i in axes]
    ms = [0] * (2 - len(axes)) + [mask.stride(i) for i in axes]
    os = [0] * (2 - len(axes)) + [out.stride(i) for i in axes]
    _witness_masked_softmax[sizes[0] * sizes[1],](
        x,
        mask,
        out,
        (
            x.shape[dim],
            sizes[1],
            *xs,
            x.stride(dim),
            *ms,
            mask.stride(dim),
            *os,
            out.stride(dim),
            max(16, triton.next_power_of_2(x.shape[dim])),
        ),
        enable_fp_fusion=False,
    )
    return out
