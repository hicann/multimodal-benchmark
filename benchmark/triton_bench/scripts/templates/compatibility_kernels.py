# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Signed-int32 division and scalar-metadata grouped matmul candidates."""
__all__ = ['tl', 'torch', 'triton']

import torch
import triton
import triton.language as tl





@triton.jit
def _floor_divide_int32(first, second):
    negative_first = first < 0
    numerator = tl.where(negative_first, -(tl.minimum(first, 0) + 1), first)
    denominator = tl.where(second > 0, -tl.maximum(second, 0), second)
    quotient = tl.full(first.shape, 0, tl.int32)
    remainder = tl.full(first.shape, 0, tl.int32)
    for bit in tl.static_range(31, -1, -1):
        remainder = (remainder << 1) - ((numerator >> bit) & 1)
        subtract = remainder <= denominator
        remainder = remainder - tl.where(subtract, denominator, 0)
        quotient = (quotient << 1) - subtract.to(tl.int32)
    remainder = remainder - negative_first.to(tl.int32)
    subtract = remainder <= denominator
    remainder = remainder - tl.where(subtract, denominator, 0)
    quotient = quotient - subtract.to(tl.int32)
    negative = negative_first != (second < 0)
    signed = tl.where(negative, quotient, -quotient)
    return signed - (negative & (remainder != 0)).to(tl.int32)


@triton.jit
def _floor_div(input_ptr, divisor, output, count: tl.constexpr, input_stride: tl.constexpr,
               divisor_stride: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < count
    first = tl.load(input_ptr + offsets * input_stride, valid, other=0)
    second = tl.load(divisor + offsets * divisor_stride, valid, other=1)
    tl.store(output + offsets, _floor_divide_int32(first, second), valid)


def _linear_stride(tensor):
    stride = tensor.stride(-1)
    expected = stride
    for size, actual in zip(reversed(tensor.shape), reversed(tensor.stride())):
        if size > 1 and actual != expected:
            raise ValueError("FloorDiv requires a uniform flattened input stride")
        expected *= size
    return stride


def floor_div(x, y):
    if x.dtype != torch.int32 or y.dtype != torch.int32 or x.shape != y.shape:
        raise ValueError("FloorDiv requires equal-shaped int32 tensors")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _floor_div[(triton.cdiv(x.numel(), 256),)](
        x, y, out, x.numel(), _linear_stride(x), _linear_stride(y), 256)
    return out


@triton.jit
def _grouped(input_ptr, weight, bias, output, configuration: tl.constexpr):
    rows: tl.constexpr = configuration.value[0]
    start: tl.constexpr = configuration.value[1]
    group: tl.constexpr = configuration.value[2]
    reduction: tl.constexpr = configuration.value[3]
    columns: tl.constexpr = configuration.value[4]
    tile: tl.constexpr = configuration.value[5]
    bk: tl.constexpr = configuration.value[6]
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    col = tl.program_id(1) * tile + tl.arange(0, tile)
    reduction_value = tl.arange(0, bk)
    acc = tl.full((tile, tile), 0, tl.float32)
    for block in tl.static_range(0, reduction, bk):
        inner = block + reduction_value
        first = tl.load(
            input_ptr + (start + row[:, None]) * reduction + inner[None, :],
            (row[:, None] < rows) & (inner[None, :] < reduction),
            other=0,
        )
        second = tl.load(
            weight + group * reduction * columns + inner[:, None] * columns + col[None, :],
            (inner[:, None] < reduction) & (col[None, :] < columns),
            other=0,
        )
        acc = acc + tl.dot(first, second)
    bias_value = tl.load(bias + group * columns + col, col < columns, other=0).to(tl.float32)
    tl.store(
        output + (start + row[:, None]) * columns + col[None, :],
        acc + bias_value[None, :],
        (row[:, None] < rows) & (col[None, :] < columns),
    )


def grouped_matmul(x, weight, bias, group_list):
    if not group_list or any(rows <= 0 for rows in group_list) or sum(group_list) != x.shape[0]:
        raise ValueError("GroupedMatmul requires positive groups partitioning x")
    if (weight.shape[0] != len(group_list) or weight.shape[1] != x.shape[1]
            or bias.shape != (len(group_list), weight.shape[2])):
        raise ValueError("GroupedMatmul weight/bias metadata does not match groups")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("GroupedMatmul requires contiguous inputs")
    out = torch.empty((x.shape[0], weight.shape[2]), dtype=x.dtype, device=x.device)
    start = 0
    for group, rows in enumerate(group_list):
        _grouped[triton.cdiv(rows, 16), triton.cdiv(weight.shape[2], 16)](
            x, weight, bias, out, (rows, start, group, x.shape[1], weight.shape[2], 16, 16), enable_fp_fusion=False
        )
        start += rows
    return out
