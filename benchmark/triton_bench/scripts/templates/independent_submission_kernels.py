# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _pointwise_kernel(first, second, output, count, first_step: tl.constexpr,
                      second_step: tl.constexpr, operation: tl.constexpr,
                      scale: tl.constexpr, shift: tl.constexpr, logarithm: tl.constexpr):
    tile: tl.constexpr = 256
    positions = tl.program_id(0) * tile + tl.arange(0, tile)
    values = tl.load(first + positions * first_step, positions < count, other=0).to(tl.float32)
    if operation == 0:
        answer = tl.abs(values)
    elif operation == 1:
        answer = -values
    elif operation == 2:
        other = tl.load(second + positions * second_step, positions < count, other=0).to(tl.float32)
        answer = values + other
    else:
        exponent = values * scale + shift
        answer = tl.exp(exponent * logarithm)
    tl.store(output + positions, answer, positions < count)


def _pointwise(first, second, operation, scale=1.0, shift=0.0, logarithm=1.0):
    output = torch.empty(first.shape, device=first.device, dtype=first.dtype)
    first_step = first.stride(-1)
    second_step = second.stride(-1) if second is not None else first_step
    _pointwise_kernel[triton.cdiv(first.numel(), 256),](
        first,
        second if second is not None else first,
        output,
        first.numel(),
        first_step,
        second_step,
        operation,
        scale,
        shift,
        logarithm,
    )
    return output


def absolute(x):
    return _pointwise(x, None, 0)


def neg(x):
    return _pointwise(x, None, 1)


def add(x, y):
    return _pointwise(x, y, 2)


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    logarithm = 1.0 if math.isclose(base, -1.0, rel_tol=0.0, abs_tol=0.0) else math.log(base)
    return _pointwise(x, None, 3, scale, shift, logarithm)


@triton.jit
def _row_kernel(source, output, columns: tl.constexpr, stride_row: tl.constexpr,
                stride_column: tl.constexpr, normalize: tl.constexpr, tile: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, tile)
    values = tl.load(source + row * stride_row + column * stride_column,
                     column < columns, other=0).to(tl.float32)
    if normalize:
        masked = tl.where(column < columns, values, -float('inf'))
        weights = tl.exp(masked - tl.max(masked, 0))
        answer = weights / tl.sum(weights, 0)
        tl.store(output + row * columns + column, answer, column < columns)
    else:
        tl.store(output + row, tl.sum(values, 0))


def softmax(x, dim=-1):
    if x.ndim != 2 or dim not in (-1, 1):
        raise ValueError('This submission supports the registered 2D last-axis domain')
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _row_kernel[(x.shape[0],)](x, output, x.shape[1], x.stride(0), x.stride(1),
                             True, triton.next_power_of_2(x.shape[1]))
    return output


def reduce_sum(x, dim=-1, keepdim=False):
    if x.ndim != 2 or dim not in (-1, 1):
        raise ValueError('This submission supports the registered 2D last-axis domain')
    shape = (x.shape[0], 1) if keepdim else (x.shape[0],)
    output = torch.empty(shape, device=x.device, dtype=x.dtype)
    _row_kernel[(x.shape[0],)](x, output, x.shape[1], x.stride(0), x.stride(1),
                             False, triton.next_power_of_2(x.shape[1]))
    return output


@triton.jit
def _transpose_kernel(source, output, rows: tl.constexpr, columns: tl.constexpr,
                      stride_row: tl.constexpr, stride_column: tl.constexpr,
                      tile: tl.constexpr):
    positions = tl.program_id(0) * tile + tl.arange(0, tile)
    source_rows = positions % rows
    source_columns = positions // rows
    values = tl.load(source + source_rows * stride_row + source_columns * stride_column,
                     positions < rows * columns, other=0)
    tl.store(output + positions, values, positions < rows * columns)


def transpose(x, perm):
    if x.ndim != 2 or tuple(perm) != (1, 0):
        raise ValueError('This submission supports the registered 2D transpose domain')
    output = torch.empty((x.shape[1], x.shape[0]), device=x.device, dtype=x.dtype)
    _transpose_kernel[(triton.cdiv(x.numel(), 256),)](
        x, output, x.shape[0], x.shape[1], x.stride(0), x.stride(1), 256)
    return output


def transpose_view(x, perm):
    return x.permute(tuple(perm))


@triton.jit
def _matrix_kernel(first, second, output, rows: tl.constexpr, columns: tl.constexpr,
                   depth: tl.constexpr, first_row: tl.constexpr, first_column: tl.constexpr,
                   second_row: tl.constexpr, second_column: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    column = tl.program_id(1) * 32 + tl.arange(0, 32)
    reduction = tl.arange(0, 64)
    accumulator = tl.full((16, 32), 0, tl.float32)
    for section in range(0, depth, 64):
        inner = section + reduction
        left = tl.load(first + row[:, None] * first_row + inner[None, :] * first_column,
                       (row[:, None] < rows) & (inner[None, :] < depth), other=0)
        right = tl.load(second + inner[:, None] * second_row + column[None, :] * second_column,
                        (inner[:, None] < depth) & (column[None, :] < columns), other=0)
        accumulator += tl.dot(left, right)
    tl.store(output + row[:, None] * columns + column[None, :], accumulator,
             (row[:, None] < rows) & (column[None, :] < columns))


def matmul(x, weight):
    output = torch.empty((x.shape[0], weight.shape[1]), device=x.device, dtype=x.dtype)
    _matrix_kernel[(triton.cdiv(x.shape[0], 16), triton.cdiv(weight.shape[1], 32))](
        x, weight, output, x.shape[0], weight.shape[1], x.shape[1],
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1))
    return output


@triton.jit
def _minimum_pair(left_value, left_index, right_value, right_index):
    choose_right = (right_value < left_value) | ((right_value == left_value) & (right_index > left_index))
    return tl.where(choose_right, right_value, left_value), tl.where(choose_right, right_index, left_index)


@triton.jit
def _scan_local(source, values, indices, summary_values, summary_indices,
                length: tl.constexpr, inner: tl.constexpr, chunks: tl.constexpr,
                tile: tl.constexpr):
    line = tl.program_id(0)
    chunk = tl.program_id(1)
    offset = chunk * tile + tl.arange(0, tile)
    address = (line // inner) * length * inner + line % inner + offset * inner
    data = tl.load(source + address, offset < length, other=float('inf'))
    position = tl.where(offset < length, offset, -1).to(tl.int64)
    prefix_value, prefix_index = tl.associative_scan((data, position), 0, _minimum_pair)
    tl.store(values + address, prefix_value, offset < length)
    tl.store(indices + address, prefix_index, offset < length)
    last = tl.minimum(tile - 1, length - chunk * tile - 1)
    final_value = tl.min(tl.where(tl.arange(0, tile) == last, prefix_value, float('inf')), 0)
    final_index = tl.max(tl.where(tl.arange(0, tile) == last, prefix_index, -1), 0)
    tl.store(summary_values + line * chunks + chunk, final_value)
    tl.store(summary_indices + line * chunks + chunk, final_index)


@triton.jit
def _scan_finish(values, indices, summary_values, summary_indices,
                 length: tl.constexpr, inner: tl.constexpr, chunks: tl.constexpr,
                 summary_tile: tl.constexpr, tile: tl.constexpr):
    line = tl.program_id(0)
    chunk = tl.program_id(1)
    preceding = tl.arange(0, summary_tile)
    block_values = tl.load(summary_values + line * chunks + preceding, preceding < chunk, other=float('inf'))
    block_indices = tl.load(summary_indices + line * chunks + preceding, preceding < chunk, other=-1)
    carry_value = tl.min(block_values, 0)
    carry_index = tl.max(tl.where(block_values == carry_value, block_indices, -1), 0)
    offset = chunk * tile + tl.arange(0, tile)
    address = (line // inner) * length * inner + line % inner + offset * inner
    local_value = tl.load(values + address, offset < length, other=float('inf'))
    local_index = tl.load(indices + address, offset < length, other=-1)
    answer, answer_index = _minimum_pair(carry_value, carry_index, local_value, local_index)
    tl.store(values + address, answer, offset < length)
    tl.store(indices + address, answer_index, offset < length)


def _cummin_implementation(input_, dim):
    if not input_.is_contiguous():
        raise ValueError('This submission supports contiguous scan inputs')
    axis = dim % input_.ndim
    length = input_.shape[axis]
    inner = math.prod(input_.shape[axis + 1:])
    lines = input_.numel() // length
    chunks = triton.cdiv(length, 1024)
    values = torch.empty(input_.shape, device=input_.device, dtype=input_.dtype)
    indices = torch.empty(input_.shape, device=input_.device, dtype=torch.int64)
    summaries = torch.empty((lines, chunks), device=input_.device, dtype=input_.dtype)
    positions = torch.empty((lines, chunks), device=input_.device, dtype=torch.int64)
    _scan_local[(lines, chunks)](input_, values, indices, summaries, positions, length, inner, chunks, 1024)
    _scan_finish[(lines, chunks)](values, indices, summaries, positions, length, inner, chunks,
                                 triton.next_power_of_2(chunks), 1024)
    return values, indices


def cummin(input_value, dim):
    """Use the registered input_value contract at the API boundary."""
    return _cummin_implementation(input_value, dim)
