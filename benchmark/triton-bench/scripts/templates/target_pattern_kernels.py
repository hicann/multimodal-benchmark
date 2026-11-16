# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Explicit bounded pattern implementations; host code only allocates and launches."""
__all__ = ['cummin', 'math', 'tl', 'torch', 'triton']

import math

import torch
import triton
import triton.language as tl

from scripts.templates.coverage_kernels import cummin





def _invalid_grouped_parameters(group_list, tensor, weight, bias):
    if (not group_list):
        return True
    if (any(rows <= 0 for rows in group_list)):
        return True
    if (sum(group_list) != tensor.shape[0]):
        return True
    if (weight.shape[:2] != (len(group_list), tensor.shape[1])):
        return True
    return bool(bias.shape != (len(group_list), weight.shape[2]))


@triton.jit
def _pattern_max_partial(x, partial, columns: tl.constexpr, chunks: tl.constexpr, block: tl.constexpr):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    offsets = chunk * block + tl.arange(0, block)
    values = tl.load(x + row * columns + offsets, offsets < columns, other=-float('inf')).to(tl.float32)
    maximum = tl.max(values, 0)
    tl.store(partial + row * chunks + chunk, maximum)


@triton.jit
def _pattern_max_finish(partial, out, chunks: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    values = tl.load(partial + row * chunks + offsets, offsets < chunks, other=-float('inf'))
    tl.store(out + row, tl.max(values, 0))


def reduce_max(x, dim=-1, keepdim=False):
    if x.ndim != 2 or dim % x.ndim != 1 or not x.is_contiguous():
        raise ValueError('ReduceMax candidate requires a contiguous last-axis input')
    rows, columns = x.shape
    chunks = triton.cdiv(columns, 1024)
    partial = torch.empty((rows, chunks), dtype=torch.float32, device=x.device)
    out = torch.empty((rows, 1) if keepdim else (rows,), dtype=x.dtype, device=x.device)
    _pattern_max_partial[(chunks, rows)](x, partial, columns, chunks, 1024)
    _pattern_max_finish[(rows,)](partial, out, chunks, triton.next_power_of_2(chunks))
    return out


@triton.jit
def _pattern_split_partial(x, weight, partial, rows: tl.constexpr, columns: tl.constexpr,
                           inner: tl.constexpr, segment: tl.constexpr, tile: tl.constexpr,
                           inner_tile: tl.constexpr):
    rows_value = tl.program_id(0) * tile + tl.arange(0, tile)
    columns_value = tl.program_id(1) * tile + tl.arange(0, tile)
    part = tl.program_id(2)
    offsets = tl.arange(0, inner_tile)
    accumulator = tl.zeros((tile, tile), tl.float32)
    for step in range(tl.cdiv(segment, inner_tile)):
        inner_value = part * segment + step * inner_tile + offsets
        first = tl.load(x + rows_value[:, None] * inner + inner_value[None, :],
                        (rows_value[:, None] < rows) & (inner_value[None, :] < inner), other=0)
        second = tl.load(weight + inner_value[:, None] * columns + columns_value[None, :],
                         (inner_value[:, None] < inner) & (columns_value[None, :] < columns), other=0)
        accumulator = tl.dot(first, second, accumulator)
    tl.store(partial + (part * rows + rows_value[:, None]) * columns + columns_value[None, :], accumulator,
             (rows_value[:, None] < rows) & (columns_value[None, :] < columns))


@triton.jit
def _pattern_split_finish(partial, out, elements: tl.constexpr, parts: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    parts_value = tl.arange(0, parts)
    values = tl.load(partial + parts_value[:, None] * elements + offsets[None, :],
                     offsets[None, :] < elements, other=0)
    tl.store(out + offsets, tl.sum(values, 0), offsets < elements)


def matmul(x, weight):
    if not x.is_contiguous() or not weight.is_contiguous() or x.shape[1] != weight.shape[0]:
        raise ValueError('Matmul candidate requires contiguous compatible inputs')
    rows, inner = x.shape
    columns = weight.shape[1]
    out = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    if rows <= 64 and inner >= 2**16:
        parts = 8
        segment = triton.cdiv(triton.cdiv(inner, parts), 32) * 32
        partial = torch.empty((parts, rows, columns), dtype=torch.float32, device=x.device)
        _pattern_split_partial[(triton.cdiv(rows, 16), triton.cdiv(columns, 16), parts)](
            x, weight, partial, rows, columns, inner, segment, 16, 32)
        _pattern_split_finish[(triton.cdiv(rows * columns, 256),)](partial, out, rows * columns, parts, 256)
    else:
        _matmul[(triton.cdiv(rows, 16), triton.cdiv(columns, 16))](
            x, weight, out, rows, columns, inner, 16, 32)
    return out


@triton.jit
def _pattern_min_pair(left, left_index, right, right_index):
    take_right = (right < left) | ((right == left) & (right_index > left_index))
    return tl.where(take_right, right, left), tl.where(take_right, right_index, left_index)


@triton.jit
def _pattern_scan_local(x, local_values, local_indices, totals, total_indices,
                        axis: tl.constexpr, inner: tl.constexpr, chunks: tl.constexpr, block: tl.constexpr):
    chunk = tl.program_id(0)
    line = tl.program_id(1)
    positions = chunk * block + tl.arange(0, block)
    base = line // inner * axis * inner + line % inner
    values = tl.load(x + base + positions * inner, positions < axis, other=float('inf')).to(tl.float32)
    indices = tl.where(positions < axis, positions, -1)
    scanned, scanned_indices = tl.associative_scan((values, indices), 0, _pattern_min_pair)
    tl.store(local_values + base + positions * inner, scanned, positions < axis)
    tl.store(local_indices + base + positions * inner, scanned_indices, positions < axis)
    last_lane = tl.minimum(block - 1, axis - chunk * block - 1)
    is_last = tl.arange(0, block) == last_lane
    minimum = tl.sum(tl.where(is_last, scanned, 0), 0)
    last_index = tl.max(tl.where(is_last, scanned_indices, -1), 0)
    tl.store(totals + line * chunks + chunk, minimum)
    tl.store(total_indices + line * chunks + chunk, last_index)


@triton.jit
def _pattern_scan_carry(totals, total_indices, carry, carry_indices,
                        chunks: tl.constexpr, block: tl.constexpr):
    line = tl.program_id(0)
    offsets = tl.arange(0, block)
    values = tl.load(totals + line * chunks + offsets, offsets < chunks, other=float('inf'))
    indices = tl.load(total_indices + line * chunks + offsets, offsets < chunks, other=-1).to(tl.int32)
    scanned, scanned_indices = tl.associative_scan((values, indices), 0, _pattern_min_pair)
    tl.store(carry + line * chunks + offsets + 1, scanned, offsets + 1 < chunks)
    tl.store(carry_indices + line * chunks + offsets + 1, scanned_indices, offsets + 1 < chunks)
    tl.store(carry + line * chunks, float('inf'))
    tl.store(carry_indices + line * chunks, -1)


@triton.jit
def _pattern_scan_finish(local_values, local_indices, carry, carry_indices, values, indices,
                         axis: tl.constexpr, inner: tl.constexpr, chunks: tl.constexpr, block: tl.constexpr):
    chunk = tl.program_id(0)
    line = tl.program_id(1)
    positions = chunk * block + tl.arange(0, block)
    base = line // inner * axis * inner + line % inner
    local = tl.load(local_values + base + positions * inner, positions < axis, other=float('inf'))
    local_index = tl.load(local_indices + base + positions * inner, positions < axis, other=-1)
    carry_value = tl.load(carry + line * chunks + chunk)
    carry_index = tl.load(carry_indices + line * chunks + chunk)
    take_carry = (carry_value < local) | ((carry_value == local) & (carry_index > local_index))
    tl.store(values + base + positions * inner, tl.where(take_carry, carry_value, local), positions < axis)
    tl.store(indices + base + positions * inner, tl.where(take_carry, carry_index, local_index), positions < axis)


def _cummin_implementation(input_, dim):
    if not input_.is_contiguous():
        raise ValueError('Cummin candidate requires contiguous input_')
    axis = input_.shape[dim]
    inner = math.prod(input_.shape[dim + 1:]) if dim >= 0 else math.prod(input_.shape[len(input_.shape) + dim + 1:])
    values = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device)
    indices = torch.empty(input_.shape, dtype=torch.int64, device=input_.device)
    if axis < 2**20:
        _cummin[(input_.numel(),)](input_, values, indices, axis, inner,
                                 max(16, triton.next_power_of_2(axis)))
        return values, indices
    chunks = triton.cdiv(axis, 512)
    lines = input_.numel() // axis
    local = torch.empty(input_.shape, dtype=torch.float32, device=input_.device)
    local_indices = torch.empty(input_.shape, dtype=torch.int64, device=input_.device)
    totals = torch.empty((lines, chunks), dtype=torch.float32, device=input_.device)
    total_indices = torch.empty((lines, chunks), dtype=torch.int64, device=input_.device)
    carry = torch.empty((lines, chunks), dtype=torch.float32, device=input_.device)
    carry_indices = torch.empty((lines, chunks), dtype=torch.int64, device=input_.device)
    _pattern_scan_local[(chunks, lines)](input_, local, local_indices, totals, total_indices, axis, inner, chunks, 512)
    _pattern_scan_carry[(lines,)](totals, total_indices, carry, carry_indices, chunks,
                                 triton.next_power_of_2(chunks))
    _pattern_scan_finish[(chunks, lines)](local, local_indices, carry, carry_indices, values, indices,
                                         axis, inner, chunks, 512)
    return values, indices




@triton.jit
def _pattern_bias_tile(x, weight, bias, out, configuration: tl.constexpr):
    rows: tl.constexpr = configuration.value[0]
    start: tl.constexpr = configuration.value[1]
    group: tl.constexpr = configuration.value[2]
    inner: tl.constexpr = configuration.value[3]
    columns: tl.constexpr = configuration.value[4]
    tile: tl.constexpr = configuration.value[5]
    inner_tile: tl.constexpr = configuration.value[6]
    rows_value = tl.program_id(0) * tile + tl.arange(0, tile)
    columns_value = tl.program_id(1) * tile + tl.arange(0, tile)
    offsets = tl.arange(0, inner_tile)
    accumulator = tl.zeros((tile, tile), tl.float32)
    for step in range(tl.cdiv(inner, inner_tile)):
        inner_value = step * inner_tile + offsets
        first = tl.load(
            x + (start + rows_value[:, None]) * inner + inner_value[None, :],
            (rows_value[:, None] < rows) & (inner_value[None, :] < inner),
            other=0,
        )
        second = tl.load(
            weight + group * inner * columns + inner_value[:, None] * columns + columns_value[None, :],
            (inner_value[:, None] < inner) & (columns_value[None, :] < columns),
            other=0,
        )
        accumulator = tl.dot(first, second, accumulator)
    bias_value = tl.load(bias + group * columns + columns_value, columns_value < columns, other=0).to(tl.float32)
    tl.store(
        out + (start + rows_value[:, None]) * columns + columns_value[None, :],
        accumulator + bias_value[None, :],
        (rows_value[:, None] < rows) & (columns_value[None, :] < columns),
    )


def grouped_matmul(x, weight, bias, group_list):
    if (_invalid_grouped_parameters(group_list, x, weight, bias)):
        raise ValueError('GroupedMatmul metadata does not partition inputs')
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError('GroupedMatmul candidate requires contiguous inputs')
    out = torch.empty((x.shape[0], weight.shape[2]), dtype=x.dtype, device=x.device)
    start = 0
    for group, rows in enumerate(group_list):
        kernel = _pattern_bias_tile if x.shape[0] >= 2048 else _grouped
        kernel[triton.cdiv(rows, 16), triton.cdiv(weight.shape[2], 16)](
            x, weight, bias, out, (rows, start, group, x.shape[1], weight.shape[2], 16, 16), enable_fp_fusion=False
        )
        start += rows
    return out
