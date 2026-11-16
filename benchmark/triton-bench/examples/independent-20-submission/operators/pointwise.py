# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Independent regression submission: explicit Triton arithmetic, no evaluator imports."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _map(first_ptr, second_ptr, condition_ptr, output_ptr, configuration: tl.constexpr):
    count: tl.constexpr = configuration.value[0]
    first_stride: tl.constexpr = configuration.value[1]
    second_stride: tl.constexpr = configuration.value[2]
    condition_stride: tl.constexpr = configuration.value[3]
    kind: tl.constexpr = configuration.value[4]
    scale: tl.constexpr = configuration.value[5]
    shift: tl.constexpr = configuration.value[6]
    log_base: tl.constexpr = configuration.value[7]
    block: tl.constexpr = configuration.value[8]
    i = tl.program_id(0) * block + tl.arange(0, block)
    a = tl.load(first_ptr + i * first_stride, i < count, other=0).to(tl.float32)
    if kind == "abs":
        answer = tl.abs(a)
    elif kind == "neg":
        answer = -a
    elif kind == "exp":
        answer = tl.exp((a * scale + shift) * log_base)
    elif kind == "relu":
        answer = tl.maximum(a, 0.0)
    elif kind == "sigmoid":
        answer = 1.0 / (1.0 + tl.exp(-a))
    else:
        b = tl.load(second_ptr + i * second_stride, i < count, other=0).to(tl.float32)
        if kind == "add":
            answer = a + b
        elif kind == "sub":
            answer = a - b
        elif kind == "mul":
            answer = a * b
        elif kind == "maximum":
            answer = tl.maximum(a, b)
        elif kind == "equal":
            answer = a == b
        else:
            condition = tl.load(condition_ptr + i * condition_stride, i < count, other=0)
            answer = tl.where(condition, a, b)
    tl.store(output_ptr + i, answer, i < count)


def _launch(x, kind, y=None, condition=None, scale=1.0, shift=0.0, logarithm=1.0):
    output = torch.empty(tuple(x.shape), dtype=torch.bool if kind == "equal" else x.dtype, device=x.device)
    y = x if y is None else y
    condition = x if condition is None else condition
    _map[triton.cdiv(x.numel(), 256),](
        x,
        y,
        condition,
        output,
        (x.numel(), x.stride(-1), y.stride(-1), condition.stride(-1), kind, scale, shift, logarithm, 256),
    )
    return output


def evaluate_abs(x):
    return _launch(x, "abs")


def evaluate_neg(x):
    return _launch(x, "neg")


def evaluate_exp(x, base=-1.0, scale=1.0, shift=0.0):
    natural_base = math.isclose(base, -1.0, rel_tol=0.0, abs_tol=0.0)
    return _launch(x, "exp", scale=scale, shift=shift, logarithm=1.0 if natural_base else math.log(base))


def evaluate_relu(x):
    return _launch(x, "relu")


def evaluate_sigmoid(x):
    return _launch(x, "sigmoid")


def evaluate_add(x, y):
    return _launch(x, "add", y)


def evaluate_sub(x, y):
    return _launch(x, "sub", y)


def evaluate_mul(x, y):
    return _launch(x, "mul", y)


def evaluate_maximum(x1, x2):
    return _launch(x1, "maximum", x2)


def evaluate_equal(x, y):
    return _launch(x, "equal", y)


def evaluate_where(condition, x, y):
    return _launch(x, "where", y, condition)
