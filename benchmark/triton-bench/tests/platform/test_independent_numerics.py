# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""CPU interpretation of submission source; no compiler or hardware claims."""

import math
from pathlib import Path

import pytest
import torch
from test_implementation_kernel_semantics import cpu_kernels as _cpu_kernels_fixture

SOURCE = Path(__file__).resolve().parents[2] / "examples/independent-20-submission/operators"
POINTWISE = SOURCE / "pointwise.py"
ROWS = SOURCE / "rows.py"
MATRICES = SOURCE / "matrices.py"


def language(namespace):
    tl = namespace["tl"]
    tl.abs = torch.abs
    tl.rsqrt = torch.rsqrt

    def tl_min(value, axis):
        return torch.min(value, dim=axis).values

    tl.min = tl_min

    def tl_zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    tl.zeros = tl_zeros
    tl.cdiv = namespace["triton"].cdiv

    def tl_dot(left, right, accumulator):
        return accumulator + left.float() @ right.float()

    tl.dot = tl_dot
    namespace["math"] = math
    return tl


def pointwise_case(namespace, kind, x, y):
    unary = dict(abs=torch.abs, neg=torch.neg, relu=torch.relu, sigmoid=torch.sigmoid)
    binary = dict(add=torch.add, sub=torch.sub, mul=torch.mul, maximum=torch.maximum, equal=torch.eq)
    function = namespace["evaluate_" + kind]
    if kind in unary:
        return function(x), unary[kind](x)
    if kind in binary:
        return function(x, y), binary[kind](x, y)
    if kind == "where":
        condition = x > 0
        return function(condition, x, y), torch.where(condition, x, y)
    return function(x, base=2.0, scale=0.5, shift=0.2), torch.exp((x * 0.5 + 0.2) * math.log(2.0))


@pytest.mark.parametrize("cpu_kernels", [POINTWISE], indirect=True)
@pytest.mark.parametrize(
    "kind", ["abs", "neg", "exp", "relu", "sigmoid", "add", "sub", "mul", "maximum", "equal", "where"]
)
@pytest.mark.parametrize("stride", [1, 2])
def test_grouped_constants_preserve_pointwise_arithmetic(cpu_kernels, kind, stride):
    language(cpu_kernels)
    generator = torch.Generator().manual_seed(20261116)
    x = torch.randn(517 * stride, generator=generator)[::stride]
    y = torch.randn(517 * stride, generator=generator)[::stride]
    before = x.clone(), y.clone()
    actual, expected = pointwise_case(cpu_kernels, kind, x, y)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(x, before[0]) and torch.equal(y, before[1])


def row_case(namespace, kind, data):
    if kind == "sum":
        return namespace["evaluate_sum"](data, keepdim=True), data.sum(-1, keepdim=True)
    if kind == "max":
        return namespace["evaluate_max"](data), data.amax(-1)
    if kind == "argmax":
        return namespace["evaluate_argmax"](input_value=data, dim=-1), data.argmax(-1)
    if kind == "softmax":
        return namespace["evaluate_softmax"](data), data.softmax(-1)
    weight = torch.linspace(0.5, 1.5, data.shape[-1])
    if kind == "rms":
        return (
            namespace["evaluate_rms"](data, weight),
            data * torch.rsqrt((data * data).mean(-1, keepdim=True) + 1e-6) * weight,
        )
    bias = torch.linspace(-0.2, 0.2, data.shape[-1])
    return namespace["evaluate_layer"](data, weight, bias), torch.nn.functional.layer_norm(
        data, (data.shape[-1],), weight, bias, 1e-6
    )


@pytest.mark.parametrize("cpu_kernels", [ROWS], indirect=True)
@pytest.mark.parametrize("kind", ["sum", "max", "argmax", "softmax", "rms", "layer"])
def test_row_operator_bindings_and_masked_tail(cpu_kernels, kind):
    language(cpu_kernels)
    data = torch.randn((3, 37), generator=torch.Generator().manual_seed(20261116))
    before = data.clone()
    actual, expected = row_case(cpu_kernels, kind, data)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(data, before)


@pytest.mark.parametrize("cpu_kernels", [MATRICES], indirect=True)
@pytest.mark.parametrize("kind", ["transpose", "gather", "matmul"])
def test_matrix_operator_bindings_and_partial_tiles(cpu_kernels, kind):
    language(cpu_kernels)
    data = torch.randn((5, 37), generator=torch.Generator().manual_seed(20261116))
    before = data.clone()
    if kind == "transpose":
        actual, expected = cpu_kernels["evaluate_transpose"](data), data.T
    elif kind == "gather":
        index = torch.tensor([[0, 36, 11]] * 5)
        actual, expected = cpu_kernels["evaluate_gather"](data, index, dim=1), torch.gather(data, 1, index)
    else:
        weight = torch.randn((37, 9), generator=torch.Generator().manual_seed(20261117))
        actual, expected = cpu_kernels["evaluate_matmul"](data, weight), data @ weight
    torch.testing.assert_close(actual, expected)
    assert torch.equal(data, before)


# Give the imported fixture an explicit public name without shadowing test arguments.
interpreted_kernels = pytest.fixture(name="cpu_kernels")(_cpu_kernels_fixture.__wrapped__)
