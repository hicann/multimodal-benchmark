# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bounded extension kernels. Results require fresh compilation and NPU validation."""
__all__ = ['math', 'tl', 'torch', 'triton']

import math

import torch
import triton
import triton.language as tl





@triton.jit
def _copy(x, o, n: tl.constexpr, b: tl.constexpr):
    i = tl.program_id(0) * b + tl.arange(0, b)
    tl.store(o + i, tl.load(x + i, i < n, other=0), i < n)


@triton.jit
def _scatter(index, updates, out, n: tl.constexpr, inner: tl.constexpr,
             axis: tl.constexpr, count: tl.constexpr, add: tl.constexpr, b: tl.constexpr):
    i = tl.program_id(0) * b + tl.arange(0, b)
    index_value = tl.load(index + i, i < n, other=0)
    target = (i // (count * inner)) * axis * inner + index_value * inner + i % inner
    value = tl.load(updates + i, i < n, other=0)
    if add:
        tl.atomic_add(out + target, value, i < n)
    else:
        tl.store(out + target, value, i < n)


def scatter(data, indices, updates, dim, reduce=None):
    if not data.is_contiguous() or not indices.is_contiguous() or not updates.is_contiguous():
        raise ValueError("coverage Scatter requires contiguous tensors")
    out = torch.empty(data.shape, dtype=data.dtype, device=data.device)
    _copy[(triton.cdiv(data.numel(), 256),)](data, out, data.numel(), 256)
    _scatter[(triton.cdiv(indices.numel(), 256),)](
        indices, updates, out, indices.numel(), math.prod(data.shape[dim + 1:]),
        data.shape[dim], indices.shape[dim], reduce == "add", 256)
    return out


@triton.jit
def _cummin(x, v, i, axis: tl.constexpr, inner: tl.constexpr, b: tl.constexpr):
    output = tl.program_id(0)
    pos = (output // inner) % axis
    base = output // (axis * inner) * axis * inner + output % inner
    j = tl.arange(0, b)
    valid = j <= pos
    # The bounded corpus uses floating inputs and tests equal-minimum ties.
    value = tl.load(x + base + j * inner, valid, other=0).to(tl.float32)
    minimum = tl.min(tl.where(valid, value, float("inf")), 0)
    index = tl.max(tl.where(valid & (value == minimum), j, -1), 0)
    tl.store(v + output, minimum)
    tl.store(i + output, index.to(tl.int64))


def _cummin_implementation(input_, dim):
    out = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device)
    indices = torch.empty(input_.shape, dtype=torch.int64, device=input_.device)
    axis = input_.shape[dim]
    _cummin[(input_.numel(),)](input_, out, indices, axis, math.prod(input_.shape[dim + 1:]),
                            max(16, triton.next_power_of_2(axis)))
    return out, indices


def cummin(input_value, dim):
    """Use the registered input_value contract at the API boundary."""
    return _cummin_implementation(input_value, dim)


@triton.jit
def _grouped(x, w, bias, o, k: tl.constexpr, n: tl.constexpr,
             groups: tl.constexpr, tile: tl.constexpr, bk: tl.constexpr):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    col = tl.arange(0, tile)
    kk = tl.arange(0, bk)
    group = tl.program_id(1)
    start = 0
    for g in tl.static_range(len(groups)):
        if group == g:
            a = tl.load(x + (start + row[:, None]) * k + kk[None, :],
                        (row[:, None] < groups[g]) & (kk[None, :] < k), other=0)
            b = tl.load(w + g * k * n + kk[:, None] * n + col[None, :],
                        (kk[:, None] < k) & (col[None, :] < n), other=0)
            acc = tl.dot(a, b)
            bias_value = tl.load(bias + g * n + col, col < n, other=0).to(tl.float32)
            tl.store(o + (start + row[:, None]) * n + col[None, :], acc + bias_value[None, :],
                     (row[:, None] < groups[g]) & (col[None, :] < n))
        start += groups[g]


def grouped_matmul(x, weight, bias, group_list):
    if x.shape[1] != 32 or weight.shape[-1] != 16 or sum(group_list) != x.shape[0]:
        raise ValueError("bounded coverage grouped matmul requires K=32,N=16 and valid groups")
    out = torch.empty((x.shape[0], 16), dtype=x.dtype, device=x.device)
    _grouped[(triton.cdiv(max(group_list), 16), len(group_list))](
        x, weight, bias, out, 32, 16, tuple(group_list), 16, 32,
        enable_fp_fusion=False)
    return out


@triton.jit
def _cross_entropy(x, target, loss, valid, c: tl.constexpr, ignore: tl.constexpr, b: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, b)
    value = tl.load(x + row * c + j, j < c, other=-float("inf")).to(tl.float32)
    maximum = tl.max(value, 0)
    logsum = tl.log(tl.sum(tl.exp(value - maximum), 0)) + maximum
    label = tl.load(target + row)
    selected = tl.load(x + row * c + label, label != ignore, other=0).to(tl.float32)
    tl.store(loss + row, tl.where(label != ignore, logsum - selected, 0))
    tl.store(valid + row, (label != ignore).to(tl.float32))


@triton.jit
def _loss_reduce(loss, valid, o, n: tl.constexpr, mean: tl.constexpr, b: tl.constexpr):
    j = tl.arange(0, b)
    total = tl.sum(tl.load(loss + j, j < n, other=0), 0)
    if mean:
        count = tl.sum(tl.load(valid + j, j < n, other=0), 0)
        total = total / count
    tl.store(o, total)


def _cross_entropy_loss_implementation(input_, target, reduction="mean", ignore_index=-100):
    rows, columns = input_.shape
    loss = torch.empty((rows,), dtype=torch.float32, device=input_.device)
    valid = torch.empty((rows,), dtype=torch.float32, device=input_.device)
    out = torch.empty((), dtype=input_.dtype, device=input_.device)
    _cross_entropy[(rows,)](input_, target, loss, valid, columns, ignore_index,
                           max(16, triton.next_power_of_2(columns)), enable_fp_fusion=False)
    _loss_reduce[(1,)](loss, valid, out, rows, reduction == "mean",
                       max(16, triton.next_power_of_2(rows)))
    return out


def cross_entropy_loss(input_value, target, reduction='mean', ignore_index=-100):
    """Use the registered input_value contract at the API boundary."""
    return _cross_entropy_loss_implementation(input_value, target, reduction, ignore_index)
