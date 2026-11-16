# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Separate row reductions and affine normalization; all data arithmetic is on device."""

import torch
import triton
import triton.language as tl


@triton.jit
def _max_chunks(x_ptr, partial_ptr, cols: tl.constexpr, chunks: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    col = chunk * block + tl.arange(0, block)
    values = tl.load(x_ptr + row * cols + col, col < cols, other=-float("inf")).to(tl.float32)
    tl.store(partial_ptr + row * chunks + chunk, tl.max(values, 0))


@triton.jit
def _row(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    kind: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    valid = col < cols
    x = tl.load(x_ptr + row * cols + col, valid, other=0).to(tl.float32)
    if kind == "sum":
        tl.store(out_ptr + row, tl.sum(x, 0))
    elif kind == "max" or kind == "argmax":
        maximum = tl.max(tl.where(valid, x, -float("inf")), 0)
        if kind == "argmax":
            result = tl.min(tl.where(valid & (x == maximum), col, 2147483647), 0)
        else:
            result = maximum
        tl.store(out_ptr + row, result)
    elif kind == "softmax":
        peak = tl.max(tl.where(valid, x, -float("inf")), 0)
        numerator = tl.where(valid, tl.exp(x - peak), 0.0)
        tl.store(out_ptr + row * cols + col, numerator / tl.sum(numerator, 0), valid)
    else:
        if kind == "layer":
            average = tl.sum(x, 0) / cols
            centered = tl.where(valid, x - average, 0.0)
        else:
            centered = x
        square_mean = tl.sum(centered * centered, 0) / cols
        gain = tl.load(weight_ptr + col, valid, other=0).to(tl.float32)
        normalized = centered * tl.rsqrt(square_mean + epsilon) * gain
        if kind == "layer":
            normalized += tl.load(bias_ptr + col, valid, other=0).to(tl.float32)
        tl.store(out_ptr + row * cols + col, normalized, valid)


def _run(x, kind, keepdim=False, weight=None, bias=None, epsilon=1e-6):
    rows, cols = x.shape
    reduction = kind in ("sum", "max", "argmax")
    shape = (rows, 1) if reduction and keepdim else (rows,) if reduction else (rows, cols)
    output = torch.empty(shape, dtype=torch.int64 if kind == "argmax" else x.dtype, device=x.device)
    if kind == "max" and cols > 8192:
        chunks = triton.cdiv(cols, 4096)
        partial = torch.empty((rows, chunks), dtype=torch.float32, device=x.device)
        _max_chunks[(rows, chunks)](x, partial, cols, chunks, 4096)
        _row[(rows,)](partial, partial, partial, output, rows, chunks, "max", epsilon, triton.next_power_of_2(chunks))
        return output
    _row[(rows,)](
        x,
        x if weight is None else weight,
        x if bias is None else bias,
        output,
        rows,
        cols,
        kind,
        epsilon,
        triton.next_power_of_2(cols),
    )
    return output


def evaluate_sum(x, dim=-1, keepdim=False):
    return _run(x, "sum", keepdim)


def evaluate_max(x, dim=-1, keepdim=False):
    return _run(x, "max", keepdim)


def evaluate_argmax(input_value, dim=None, keepdim=False):
    return _run(input_value, "argmax", keepdim)


def evaluate_softmax(x, dim=-1):
    return _run(x, "softmax")


def evaluate_rms(x, gamma, epsilon=1e-6):
    return _run(x, "rms", weight=gamma, epsilon=epsilon)


def evaluate_layer(x, weight, bias, epsilon=1e-6):
    return _run(x, "layer", weight=weight, bias=bias, epsilon=epsilon)
