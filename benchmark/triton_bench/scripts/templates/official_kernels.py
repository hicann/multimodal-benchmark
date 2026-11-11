# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Real Triton kernels. Host code only allocates outputs and selects launch metadata."""
__all__ = ['extension', 'libdevice', 'math', 'tl', 'torch', 'triton', 'validate_elementwise_layout']

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
import triton.language.extra.cann.libdevice as libdevice

from src.triton_eval.inputs import validate_elementwise_layout






@triton.jit
def _floor_divide_int32(first, second):
    # Unsigned magnitudes retain abs(INT_MIN) == 2**31 exactly. Long
    # division avoids the backend integer-division path at int32 limits.
    numerator = first.to(tl.uint32)
    denominator = second.to(tl.uint32)
    numerator = tl.where(first < 0, 0 - numerator, numerator)
    denominator = tl.where(second < 0, 0 - denominator, denominator)
    quotient = tl.full(first.shape, 0, tl.uint32)
    remainder = tl.full(first.shape, 0, tl.uint32)
    for bit in tl.static_range(31, -1, -1):
        remainder = (remainder << 1) | ((numerator >> bit) & 1)
        subtract = remainder >= denominator
        remainder = tl.where(subtract, remainder - denominator, remainder)
        quotient = (quotient << 1) | subtract.to(tl.uint32)
    negative = (first < 0) != (second < 0)
    signed = tl.where(negative, 0 - quotient, quotient).to(tl.int32)
    return signed - (negative & (remainder != 0)).to(tl.int32)


@triton.jit
def _elementwise_binary_4(original, value, other_original, other, condition, offsets, valid, op: tl.constexpr):
    if op == "BitwiseOr":
        result = original | other_original
    elif op == "BitwiseXor":
        result = original ^ other_original
    elif op == "LogicalAnd":
        result = (value != 0) & (other != 0)
    elif op == "LogicalOr":
        result = (value != 0) | (other != 0)
    else:
        condition_value = tl.load(condition + offsets, valid, other=0)
        result = tl.where(condition_value, original, other_original)
    return result


@triton.jit
def _elementwise_binary_3(original, value, other_original, other, condition, offsets, valid, op: tl.constexpr):
    if op == "GreaterEqual":
        result = value >= other
    elif op == "Less":
        result = value < other
    elif op == "LessEqual":
        result = value <= other
    elif op == "BitwiseAnd":
        result = original & other_original
    else:
        result = _elementwise_binary_4(original, value, other_original, other, condition, offsets, valid, op)
    return result


@triton.jit
def _elementwise_binary_2(original, value, other_original, other, condition, offsets, valid, op: tl.constexpr):
    if op == "Minimum":
        result = tl.minimum(value, other)
    elif op == "Equal":
        result = value == other
    elif op == "NotEqual":
        result = value != other
    elif op == "Greater":
        result = value > other
    else:
        result = _elementwise_binary_3(original, value, other_original, other, condition, offsets, valid, op)
    return result


@triton.jit
def _elementwise_binary_1(original, value, other_original, other, condition, offsets, valid, op: tl.constexpr):
    if op == "FloorDiv":
        result = _floor_divide_int32(original, other_original)
    elif op == "Mod":
        result = original % other_original
    elif op == "Pow":
        result = tl.exp(other * tl.log(value))
    elif op == "Maximum":
        result = tl.maximum(value, other)
    else:
        result = _elementwise_binary_2(original, value, other_original, other, condition, offsets, valid, op)
    return result


@triton.jit
def _elementwise_binary_0(original, value, other_original, other, condition, offsets, valid, op: tl.constexpr):
    if op == "Add":
        result = value + other
    elif op == "Sub":
        result = value - other
    elif op == "Mul":
        result = value * other
    elif op == "Div":
        result = value / other
    else:
        result = _elementwise_binary_1(original, value, other_original, other, condition, offsets, valid, op)
    return result


@triton.jit
def _elementwise_unary_6(original, value, second, condition, offsets, valid, attributes: tl.constexpr):
    op: tl.constexpr = attributes.value[0]
    lower: tl.constexpr = attributes.value[4]
    upper: tl.constexpr = attributes.value[5]
    approximate: tl.constexpr = attributes.value[6]
    if op == "Clamp":
        result = tl.minimum(tl.maximum(value, lower), upper)
    elif op == "Mish":
        result = value * libdevice.tanh(libdevice.log1p(tl.exp(value)))
    elif op == "Gelu":
        if approximate == "none":
            result = value * 0.5 * (1 + tl.erf(value * 0.7071067811865476))
        else:
            result = value * 0.5 * (1 + libdevice.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)))
    else:
        other_original = tl.load(second + offsets, valid, other=1)
        other = other_original.to(tl.float32)
        result = _elementwise_binary_0(original, value, other_original, other, condition, offsets, valid, op)
    return result


@triton.jit
def _elementwise_unary_5(original, value, second, condition, offsets, valid, attributes: tl.constexpr):
    op: tl.constexpr = attributes.value[0]
    if op == "Relu":
        result = tl.maximum(value, 0)
    elif op == "Sigmoid":
        result = tl.sigmoid(value)
    elif op == "Signbit":
        result = value.to(tl.int32, bitcast=True) < 0
    elif op == "BitwiseNot":
        result = ~original
    else:
        result = _elementwise_unary_6(original, value, second, condition, offsets, valid, attributes)
    return result


@triton.jit
def _elementwise_unary_4(original, value, second, condition, offsets, valid, attributes: tl.constexpr):
    op: tl.constexpr = attributes.value[0]
    if op == "Cosh":
        result = libdevice.cosh(value)
    elif op == "Floor":
        result = tl.floor(value)
    elif op == "Ceil":
        result = tl.ceil(value)
    elif op == "Trunc":
        result = libdevice.trunc(value)
    else:
        result = _elementwise_unary_5(original, value, second, condition, offsets, valid, attributes)
    return result


@triton.jit
def _elementwise_unary_3(original, value, second, condition, offsets, valid, attributes: tl.constexpr):
    op: tl.constexpr = attributes.value[0]
    if op == "Cos":
        result = tl.cos(value)
    elif op == "Tan":
        result = libdevice.tan(value)
    elif op == "Tanh":
        result = libdevice.tanh(value)
    elif op == "Sinh":
        result = libdevice.sinh(value)
    else:
        result = _elementwise_unary_4(original, value, second, condition, offsets, valid, attributes)
    return result


@triton.jit
def _elementwise_unary_2(original, value, second, condition, offsets, valid, attributes: tl.constexpr):
    op: tl.constexpr = attributes.value[0]
    if op == "Log1p":
        result = libdevice.log1p(value)
    elif op == "Sqrt":
        result = tl.sqrt(value)
    elif op == "Rsqrt":
        result = tl.rsqrt(value)
    elif op == "Sin":
        result = tl.sin(value)
    else:
        result = _elementwise_unary_3(original, value, second, condition, offsets, valid, attributes)
    return result


@triton.jit
def _elementwise_unary_1(original, value, second, condition, offsets, valid, attributes: tl.constexpr):
    op: tl.constexpr = attributes.value[0]
    if op == "Expm1":
        result = libdevice.expm1(value)
    elif op == "Log":
        result = tl.log(value)
    elif op == "Log2":
        result = tl.log2(value)
    elif op == "Log10":
        result = libdevice.log10(value)
    else:
        result = _elementwise_unary_2(original, value, second, condition, offsets, valid, attributes)
    return result


@triton.jit
def _elementwise_unary_0(original, value, second, condition, offsets, valid, attributes: tl.constexpr):
    op: tl.constexpr = attributes.value[0]
    base: tl.constexpr = attributes.value[1]
    scale: tl.constexpr = attributes.value[2]
    shift: tl.constexpr = attributes.value[3]
    if op == "Abs":
        result = tl.abs(value)
    elif op == "Neg":
        result = -value
    elif op == "Exp":
        exponent = value * scale + shift
        if base == -1.0:
            result = tl.exp(exponent)
        else:
            result = tl.exp(exponent * tl.log(base))
    elif op == "Exp2":
        result = tl.exp2(value)
    else:
        result = _elementwise_unary_1(original, value, second, condition, offsets, valid, attributes)
    return result


@triton.jit
def _elementwise(x, y, condition, out, configuration: tl.constexpr):
    count: tl.constexpr = configuration.value[0]
    op: tl.constexpr = configuration.value[1]
    base: tl.constexpr = configuration.value[2]
    scale: tl.constexpr = configuration.value[3]
    shift: tl.constexpr = configuration.value[4]
    lower: tl.constexpr = configuration.value[5]
    upper: tl.constexpr = configuration.value[6]
    approximate: tl.constexpr = configuration.value[7]
    block: tl.constexpr = configuration.value[8]
    x_stride: tl.constexpr = configuration.value[9]
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < count
    original = tl.load(x + offsets * x_stride, valid, other=0)
    value = original.to(tl.float32)
    result = _elementwise_unary_0(
        original, value, y, condition, offsets, valid, (op, base, scale, shift, lower, upper, approximate)
    )
    tl.store(out + offsets, result, valid)


@triton.jit
def _rows(x, weight, bias, out, cols: tl.constexpr, op: tl.constexpr,
          epsilon: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    valid = columns < cols
    original = tl.load(x + row * cols + columns, valid, other=0)
    value = original.to(tl.float32)
    if op == "ReduceSum" or op == "ReduceMean":
        result = tl.sum(value, 0)
        if op == "ReduceMean":
            result = result / cols
        tl.store(out + row, result)
    elif op == "ReduceMax":
        result = tl.max(tl.where(valid, value, -float("inf")), 0)
        tl.store(out + row, result)
    elif op == "ReduceMin":
        result = tl.min(tl.where(valid, value, float("inf")), 0)
        tl.store(out + row, result)
    elif op == "ArgMax":
        result = tl.argmax(tl.where(valid, value, -float("inf")), 0, tie_break_left=True)
        tl.store(out + row, result.to(tl.int64))
    elif op == "ArgMin":
        result = tl.argmin(tl.where(valid, value, float("inf")), 0, tie_break_left=True)
        tl.store(out + row, result.to(tl.int64))
    elif op == "XorSum":
        result = tl.xor_sum(original, 0)
        tl.store(out + row, result)
    elif op == "CumSum":
        result = tl.cumsum(value, 0)
        tl.store(out + row * cols + columns, result, valid)
    elif op == "CumProd":
        result = tl.cumprod(tl.where(valid, value, 1.0), 0)
        tl.store(out + row * cols + columns, result, valid)
    elif op == "Softmax":
        row_maximum = tl.max(tl.where(valid, value, -float("inf")), 0)
        exponent = tl.where(valid, tl.exp(value - row_maximum), 0)
        result = exponent / tl.sum(exponent, 0)
        tl.store(out + row * cols + columns, result, valid)
    else:
        centered = value
        if op == "LayerNorm":
            mean = tl.sum(value, 0) / cols
            centered = tl.where(valid, value - mean, 0)
        variance = tl.sum(centered * centered, 0) / cols
        weight_value = tl.load(weight + columns, valid, other=0).to(tl.float32)
        result = centered * tl.rsqrt(variance + epsilon) * weight_value
        if op == "LayerNorm":
            bias_value = tl.load(bias + columns, valid, other=0).to(tl.float32)
            result = result + bias_value
        tl.store(out + row * cols + columns, result, valid)


@triton.jit
def _layout(x, y, out, rows: tl.constexpr, cols: tl.constexpr, other_cols: tl.constexpr,
            out_cols: tl.constexpr, op: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < rows * out_cols
    row = offsets // out_cols
    column = offsets % out_cols
    if op == "Transpose":
        result = tl.load(x + (offsets % rows) * cols + offsets // rows, valid, other=0)
    elif op == "Flip":
        result = tl.load(x + row * cols + cols - 1 - column, valid, other=0)
    elif op == "Concat":
        first = tl.load(x + row * cols + column, valid & (column < cols), other=0)
        second = tl.load(y + row * other_cols + column - cols, valid & (column >= cols), other=0)
        result = tl.where(column < cols, first, second)
    else:
        gate = tl.load(x + row * cols + column, valid, other=0).to(tl.float32)
        multiplier = tl.load(x + row * cols + out_cols + column, valid, other=0).to(tl.float32)
        result = gate * tl.sigmoid(gate) * multiplier
    tl.store(out + offsets, result, valid)


@triton.jit
def _gather(x, index, out, cols: tl.constexpr, out_cols: tl.constexpr,
            source_block: tl.constexpr, index_block: tl.constexpr):
    row = tl.program_id(0)
    source_columns = tl.arange(0, source_block)
    output_columns = tl.arange(0, index_block)
    value = tl.load(x + row * cols + source_columns, source_columns < cols, other=0)
    index_value = tl.load(index + row * out_cols + output_columns,
                          output_columns < out_cols, other=0).to(tl.int32)
    result = tl.gather(value, index_value, 0)
    tl.store(out + row * out_cols + output_columns, result, output_columns < out_cols)


@triton.jit
def _sort(x, out, cols: tl.constexpr, out_cols: tl.constexpr, descending: tl.constexpr,
          block: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    if descending:
        padding = -float("inf")
    else:
        padding = float("inf")
    value = tl.load(x + row * cols + columns, columns < cols, other=padding).to(tl.float32)
    # The CANN sort path also handles a one-element row. Store only the
    # requested prefix for TopKValues, including non-power-of-two k.
    result = extension.sort(value, descending=descending, dim=0)
    tl.store(out + row * out_cols + columns, result, columns < out_cols)


@triton.jit
def _matmul(x, weight, out, rows: tl.constexpr, cols: tl.constexpr, inner: tl.constexpr,
            tile: tl.constexpr, inner_tile: tl.constexpr):
    rows_value = tl.program_id(0) * tile + tl.arange(0, tile)
    columns = tl.program_id(1) * tile + tl.arange(0, tile)
    inner_value = tl.arange(0, inner_tile)
    accumulator = tl.zeros((tile, tile), tl.float32)
    for step in range(tl.cdiv(inner, inner_tile)):
        positions = step * inner_tile + inner_value
        first = tl.load(x + rows_value[:, None] * inner + positions[None, :],
                        (rows_value[:, None] < rows) & (positions[None, :] < inner), other=0)
        second = tl.load(weight + positions[:, None] * cols + columns[None, :],
                         (positions[:, None] < inner) & (columns[None, :] < cols), other=0)
        accumulator = tl.dot(first, second, accumulator)
    tl.store(out + rows_value[:, None] * cols + columns[None, :], accumulator,
             (rows_value[:, None] < rows) & (columns[None, :] < cols))




def _validate_dispatch_inputs(inputs, first, elementwise):
    return validate_elementwise_layout(inputs, first, elementwise)


def _dispatch_output_metadata(name, family, first, inputs, attrs, rows, columns):
    dtype = torch.bool if family in ("comparison", "logical") or name == "Signbit" else first.dtype
    if family in ("reduce", "arg"):
        output_shape = (rows, 1) if attrs.get("keepdim", False) else (rows,)
        if family == "arg":
            dtype = torch.int64
        return dtype, output_shape
    if family == "transpose":
        output_shape = (columns, rows)
        return dtype, output_shape
    if family == "concat":
        output_shape = (rows, columns + inputs[1].shape[1])
        return dtype, output_shape
    if family == "swiglu":
        output_shape = (rows, columns // 2)
        return dtype, output_shape
    if family == "gather":
        output_shape = tuple(inputs[1].shape)
        return dtype, output_shape
    if family == "topk":
        output_shape = (rows, attrs.get("k", 1))
        return dtype, output_shape
    if family == "matmul":
        output_shape = (rows, inputs[1].shape[1])
    else:
        output_shape = tuple(first.shape)
    return dtype, output_shape


def dispatch(name, inputs, attrs):
    family = FAMILIES[name]
    first = inputs[1] if family == "where" else inputs[0]
    elementwise = family in ("unary", "binary", "comparison", "logical",
                              "bitwise", "bitwise_unary", "where")
    rows, columns = first.shape[0], first.shape[-1]
    _validate_dispatch_inputs(inputs, first, elementwise)
    dtype, output_shape = _dispatch_output_metadata(name, family, first, inputs, attrs, rows, columns)
    output = torch.empty(output_shape, dtype=dtype, device=first.device)
    _launch_dispatch(name, inputs, attrs, output, family)
    return output


def _launch_dispatch(name, inputs, attrs, output, family):
    first = inputs[1] if family == "where" else inputs[0]
    rows, columns = first.shape[0], first.shape[-1]
    output_shape = output.shape
    second = inputs[1] if len(inputs) > 1 else first
    third = inputs[2] if len(inputs) > 2 else first
    if family in ("reduce", "arg", "scan", "softmax", "layer", "rms"):
        _rows[(rows,)](first, second, third, output, columns, name,
                       attrs.get("epsilon", 1e-6), triton.next_power_of_2(columns), enable_fp_fusion=False)
        return
    if family in ("transpose", "concat", "flip", "swiglu"):
        output_columns = columns if family == "transpose" else output_shape[1]
        _layout[(triton.cdiv(output.numel(), 256),)](
            first, second, output, rows, columns, second.shape[-1], output_columns, name, 256,
            enable_fp_fusion=False)
        return
    if family == "gather":
        _gather[(rows,)](first, second, output, columns, output_shape[1],
                         triton.next_power_of_2(columns), triton.next_power_of_2(output_shape[1]))
        return
    if family in ("sort", "topk"):
        _sort[(rows,)](first, output, columns, output_shape[1], attrs.get("descending", family == "topk"),
                       triton.next_power_of_2(columns))
        return
    if family == "matmul":
        _matmul[(triton.cdiv(rows, 16), triton.cdiv(output_shape[1], 16))](
            first, second, output, rows, output_shape[1], columns, 16, 32)
        return
    _launch_elementwise(name, inputs, attrs, output, family)


def _launch_elementwise(name, inputs, attrs, output, family):
    first = inputs[1] if family == "where" else inputs[0]
    second = inputs[1] if len(inputs) > 1 else first
    second_data = inputs[2] if family == "where" else second
    condition = inputs[0]
    _elementwise[triton.cdiv(first.numel(), 256),](
        first, second_data, condition, output,
        (
            first.numel(), name, attrs.get('base', -1.0), attrs.get('scale', 1.0),
            attrs.get('shift', 0.0), attrs.get('lower', -0.5), attrs.get('upper', 0.5),
            attrs.get('approximate', 'none'), 256, 1 if first.is_contiguous() else 2,
        ),
        enable_fp_fusion=False,
    )
