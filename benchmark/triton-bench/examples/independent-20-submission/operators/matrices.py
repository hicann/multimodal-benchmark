# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Independent matrix copy, direct indexed loads, and masked tiled multiplication."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy(
    x_ptr,
    index_ptr,
    out_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    index_width: tl.constexpr,
    transpose: tl.constexpr,
    block: tl.constexpr,
):
    linear = tl.program_id(0) * block + tl.arange(0, block)
    if transpose:
        valid = linear < rows * cols
        offset = linear % rows * cols + linear // rows
    else:
        valid = linear < rows * index_width
        index = tl.load(index_ptr + linear, valid, other=0).to(tl.int32)
        offset = linear // index_width * cols + index
    value = tl.load(x_ptr + offset, valid, other=0)
    tl.store(out_ptr + linear, value, valid)


def evaluate_transpose(x, perm=None):
    rows, cols = x.shape
    output = torch.empty((cols, rows), dtype=x.dtype, device=x.device)
    _copy[(triton.cdiv(x.numel(), 256),)](x, x, output, rows, cols, rows, True, 256)
    return output


def evaluate_gather(x, index, dim=0):
    output = torch.empty(tuple(index.shape), dtype=x.dtype, device=x.device)
    _copy[(triton.cdiv(index.numel(), 256),)](x, index, output, x.shape[0], x.shape[1], index.shape[1], False, 256)
    return output


@triton.jit
def _multiply(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows: tl.constexpr,
    inner: tl.constexpr,
    cols: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
    block_inner: tl.constexpr,
):
    m = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    n = tl.program_id(1) * block_cols + tl.arange(0, block_cols)
    k = tl.arange(0, block_inner)
    accumulator = tl.zeros((block_rows, block_cols), tl.float32)
    for start in range(tl.cdiv(inner, block_inner)):
        current = start * block_inner + k
        left = tl.load(
            x_ptr + m[:, None] * inner + current[None, :], (m[:, None] < rows) & (current[None, :] < inner), other=0
        )
        right = tl.load(
            weight_ptr + current[:, None] * cols + n[None, :], (current[:, None] < inner) & (n[None, :] < cols), other=0
        )
        accumulator = tl.dot(left, right, accumulator)
    tl.store(out_ptr + m[:, None] * cols + n[None, :], accumulator, (m[:, None] < rows) & (n[None, :] < cols))


def evaluate_matmul(x, weight):
    output = torch.empty((x.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device)
    _multiply[(triton.cdiv(x.shape[0], 32), triton.cdiv(weight.shape[1], 32))](
        x, weight, output, x.shape[0], x.shape[1], weight.shape[1], 32, 32, 32
    )
    return output
