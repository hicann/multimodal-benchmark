# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch references and deterministic, domain-aware inputs for the official catalogue."""

from __future__ import annotations

import math

from .inputs import validate_elementwise_layout
from .official_catalog import BY_NAME, ELEMENTWISE_FAMILIES


def validate_inputs(name, inputs, attrs):
    import torch

    candidate = BY_NAME[name]
    family = candidate.family
    expected = 3 if family in ("where", "layer") else 2 if family in (
        "binary", "comparison", "bitwise", "logical", "concat", "gather", "rms", "matmul") else 1
    if len(inputs) != expected:
        raise ValueError(f"{name}: expected {expected} input tensors")
    if any(not isinstance(tensor, torch.Tensor) for tensor in inputs):
        raise ValueError("all inputs must be tensors")
    first = inputs[1] if family == "where" else inputs[0]
    if any(tensor.device != first.device for tensor in inputs):
        raise ValueError("all inputs must be on one device")
    _validate_tensor_layout(candidate, family, first, inputs)
    _validate_family_shapes(name, family, first, inputs, attrs)
    _validate_index_inputs(name, family, first, inputs, attrs)
    _validate_numeric_domain(name, family, first, inputs, attrs)


def _validate_tensor_layout(candidate, family, first, inputs):
    elementwise = family in ELEMENTWISE_FAMILIES
    if not (1 <= first.ndim <= 4 if elementwise else first.ndim == 2) or min(first.shape) < 1:
        raise ValueError("candidate domain requires nonempty rank 1–4 elementwise or 2D row inputs")
    validate_elementwise_layout(inputs, first, elementwise)
    if str(first.dtype).removeprefix("torch.") not in candidate.dtypes:
        raise ValueError("main input dtype is outside the qualified domain")
    for tensor in inputs:
        if family == "gather" and tensor is inputs[1]:
            continue
        if family == "where" and tensor is inputs[0]:
            continue
        if tensor.dtype != first.dtype:
            raise ValueError("data input dtypes must agree")


def _validate_family_shapes(name, family, first, inputs, attrs):
    import torch

    if family in ("binary", "comparison", "bitwise", "logical"):
        if inputs[1].shape != first.shape:
            raise ValueError("this qualification domain requires equal binary operand shapes")
    if family == "where" and (inputs[0].dtype != torch.bool or any(
            tensor.shape != first.shape for tensor in inputs)):
        raise ValueError("where requires a bool condition with matching operand shapes")
    if family in ("reduce", "arg", "scan", "softmax", "gather", "swiglu", "concat", "flip", "sort", "topk"):
        if attrs.get("dim", -1) not in (-1, 1):
            raise ValueError("qualification domain supports only the last axis")
    if family == "transpose" and list(attrs.get("perm", [1, 0])) != [1, 0]:
        raise ValueError("transpose qualification requires perm=[1,0]")
    _validate_affine_shapes(family, first, inputs, attrs)


def _validate_affine_shapes(family, first, inputs, attrs):
    if family in ("layer", "rms"):
        if not math.isfinite(float(attrs.get("epsilon", 1e-6))) or attrs.get("epsilon", 1e-6) <= 0:
            raise ValueError("epsilon must be finite and positive")
        if any(tensor.shape != (first.shape[1],) for tensor in inputs[1:]):
            raise ValueError("normalization affine tensors must match the last dimension")
    if family == "concat" and (inputs[1].ndim != 2 or inputs[1].shape[0] != first.shape[0]):
        raise ValueError("concat row counts must agree")
    if family == "matmul" and (inputs[1].ndim != 2 or inputs[1].shape[0] != first.shape[1]):
        raise ValueError("matmul contracting dimensions must agree")
    if family == "swiglu" and first.shape[1] % 2:
        raise ValueError("SwiGlu last dimension must be even")


def _validate_index_inputs(name, family, first, inputs, attrs):
    import torch

    if family == "gather":
        index = inputs[1]
        if index.dtype != torch.int64 or index.ndim != 2 or index.shape[0] != first.shape[0]:
            raise ValueError("gather requires 2D int64 indices with matching rows")
        if bool(((index < 0) | (index >= first.shape[1])).any()):
            raise ValueError("gather index is out of bounds")
    if family == "topk":
        count = attrs.get("k", 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= first.shape[1]:
            raise ValueError("top-k requires 1 <= k <= last dimension")


def _validate_numeric_domain(name, family, first, inputs, attrs):
    if name in ("Log", "Log2", "Log10", "Rsqrt", "Pow") and bool((first <= 0).any()):
        raise ValueError("input must be positive")
    if name == "Log1p" and bool((first <= -1).any()):
        raise ValueError("log1p input must exceed -1")
    if name == "Sqrt" and bool((first < 0).any()):
        raise ValueError("sqrt input must be nonnegative")
    if name in ("Div", "FloorDiv", "Mod") and bool((inputs[1] == 0).any()):
        raise ValueError("division requires nonzero denominators")
    if name == "FloorDiv" and bool(((first == -2147483648) & (inputs[1] == -1)).any()):
        raise ValueError("int32 division overflow is outside the contract")
    if name == "Exp":
        base = float(attrs.get("base", -1.0))
        if base != -1.0 and (not math.isfinite(base) or base <= 0):
            raise ValueError("exp base must be -1 or finite and positive")
    if name == "Gelu" and attrs.get("approximate", "none") not in ("none", "tanh"):
        raise ValueError("unknown GELU approximation")
    if name == "Clamp" and not attrs.get("lower", -0.5) <= attrs.get("upper", 0.5):
        raise ValueError("clamp lower bound exceeds upper bound")


def make_inputs(name, *, seed, dtype, shape, device, attrs, value_range=None):
    import torch

    candidate = BY_NAME[name]
    family = candidate.family
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    rows, columns = shape[0], shape[-1]
    first = torch.randn(shape, generator=generator).clamp(-2.5, 2.5)
    second = torch.randn(shape, generator=generator).clamp(-2.5, 2.5)
    first.reshape(-1)[2::7] = 0
    second.reshape(-1)[1::5] = 0
    # Missing profile preserves historical v0.8 replay. New cases declare their
    # profile so distribution is independent of seed arithmetic and auditable.
    explicit = value_range is not None and (value_range.get("generator") == "official-domain-v2"
                                            or value_range.get("kind", "normal") != "normal")
    kind = (value_range or {}).get("kind", "normal")
    if explicit:
        from .official_generation import numeric_profiles

        if kind not in numeric_profiles(candidate, str(dtype).removeprefix("torch.")):
            raise ValueError(f"{name}: unsupported numeric profile {kind!r}")
        mode = 2 if kind == "boundary" else 0
    else:
        mode = seed % 4
    if mode == 1:
        first.zero_()
    elif mode == 2:
        first[:] = torch.tensor([-0.0, 0.0, -1.0, 1.0, -0.5, 0.5, -2.0, 2.0]).repeat(
            (first.numel() + 7) // 8)[:first.numel()].reshape(shape)
        second.copy_(first)
    first, second = _sample_numeric_domain(name, candidate, shape, generator, mode, first, second)
    if family == "concat":
        second = torch.randn((rows, columns // 2 + 1), generator=generator)
    if family == "matmul":
        first = first / 4
        second = torch.randint(-4, 5, (columns, rows + 3), generator=generator).float() / 8
    if explicit:
        if kind == "zeros":
            first.zero_()
        elif kind == "mixed":
            if first.numel() < 2:
                raise ValueError("mixed profile requires at least two elements")
            # Guarantee both a nonzero and a zero, independent of RNG output.
            first.reshape(-1)[0] = 1
            first.reshape(-1)[max(1, first.numel() // 2):] = 0
        elif kind == "extreme":
            first.mul_(20.0)
    return _pack_input_tensors(family, first, second, rows, columns, shape, generator, dtype, device)


def _pack_input_tensors(family, first, second, rows, columns, shape, generator, dtype, device):
    import torch

    data = [first]
    if family in ("binary", "comparison", "bitwise", "logical", "concat", "matmul"):
        data.append(second)
    if family == "where":
        condition = torch.randint(0, 2, shape, generator=generator, dtype=torch.bool)
        return [condition.to(device), first.to(device=device, dtype=dtype),
                second.to(device=device, dtype=dtype)]
    if family == "gather":
        index = torch.randint(0, columns, (rows, columns // 2 + 1), generator=generator)
        return [first.to(device=device, dtype=dtype), index.to(device)]
    if family in ("rms", "layer"):
        data.append(torch.rand(columns, generator=generator) + 0.5)
    if family == "layer":
        data.append(torch.rand(columns, generator=generator) - 0.5)
    return [tensor.to(device=device, dtype=dtype) for tensor in data]


def _sample_numeric_domain(name, candidate, shape, generator, mode, first, second):
    import torch

    if name in ("Log", "Log2", "Log10", "Rsqrt", "Pow"):
        first = first.abs() + 0.125
    elif name == "Sqrt":
        first = first.abs()
    elif name == "Log1p":
        first = first.abs() / 2 - 0.75
    elif name == "Tan":
        first = first / 2
    if name in ("Div", "FloorDiv", "Mod"):
        second = torch.where(second >= 0, second.abs() + 1, -second.abs() - 1)
    if name == "Pow":
        second = torch.randint(-3, 4, shape, generator=generator).float()
    if candidate.dtypes == ("int32",):
        first = torch.randint(-127, 128, shape, generator=generator)
        second = torch.randint(-127, 128, shape, generator=generator)
        if mode == 1:
            first.zero_()
        elif mode == 2:
            boundary = torch.tensor([-2147483648, 2147483647, -1, 0, 1, -128, 127, 1024])
            first = boundary.repeat((first.numel() + 7) // 8)[:first.numel()].reshape(shape)
        if name == "FloorDiv":
            second = torch.randint(1, 17, shape, generator=generator)
            if mode % 2:
                second = -second
    if name == "CumProd":
        first = torch.randint(-1, 2, shape, generator=generator).float()
        if mode == 0:
            first.fill_(1)
        elif mode == 2:
            first = torch.where(first == 0, 0.5, first)
    return first, second


def _floor_divide_int32(first, second):
    """Exact signed division with an FP32 estimate and integer residual correction.

    Keep the estimate in int64: INT_MAX / 1 rounds to 2**31 in FP32.
    Refining its residual bounds the remaining quotient error by one; the final
    comparisons enforce floor division for either sign of the denominator.
    """
    import torch

    numerator, denominator = first.to(torch.int64), second.to(torch.int64)
    quotient = torch.trunc(first.float() / second.float()).to(torch.int64)
    remainder = numerator - quotient * denominator
    quotient = quotient + torch.trunc(remainder.float() / second.float()).to(torch.int64)
    remainder = numerator - quotient * denominator
    below = ((denominator > 0) & (remainder < 0)) | ((denominator < 0) & (remainder > 0))
    above = ((denominator > 0) & (remainder >= denominator)) | ((denominator < 0) & (remainder <= denominator))
    return (quotient - below.to(torch.int64) + above.to(torch.int64)).to(first.dtype)


def reference(name, *inputs, **attrs):
    import torch

    validate_inputs(name, inputs, attrs)
    family = BY_NAME[name].family
    first = inputs[0]
    result_dtype = inputs[1].dtype if family == "where" else first.dtype
    value = first.float() if first.is_floating_point() else first
    other = inputs[1].float() if len(inputs) > 1 and inputs[1].is_floating_point() else (
        inputs[1] if len(inputs) > 1 else None)
    result = _elementwise_reference(name, inputs, attrs, first, value, other)
    if result.dtype == torch.bool or family == "arg":
        return result
    return result.to(result_dtype)


def _reference_functions():
    import torch

    unary = {"Abs": torch.abs, "Neg": torch.neg, "Exp2": torch.exp2,
             "Expm1": torch.expm1, "Log": torch.log, "Log2": torch.log2,
             "Log10": torch.log10, "Log1p": torch.log1p, "Sqrt": torch.sqrt,
             "Rsqrt": torch.rsqrt, "Sin": torch.sin, "Cos": torch.cos, "Tan": torch.tan,
             "Tanh": torch.tanh, "Sinh": torch.sinh, "Cosh": torch.cosh,
             "Floor": torch.floor, "Ceil": torch.ceil, "Trunc": torch.trunc,
             "Relu": torch.relu, "Sigmoid": torch.sigmoid, "Signbit": torch.signbit,
             "BitwiseNot": torch.bitwise_not}
    binary = {"Add": torch.add, "Sub": torch.sub, "Mul": torch.mul, "Div": torch.div,
              "FloorDiv": torch.floor_divide, "Mod": torch.fmod, "Pow": torch.pow,
              "Maximum": torch.maximum, "Minimum": torch.minimum,
              "Equal": torch.eq, "NotEqual": torch.ne, "Greater": torch.gt,
              "GreaterEqual": torch.ge, "Less": torch.lt, "LessEqual": torch.le,
              "BitwiseAnd": torch.bitwise_and, "BitwiseOr": torch.bitwise_or,
              "BitwiseXor": torch.bitwise_xor, "LogicalAnd": torch.logical_and,
              "LogicalOr": torch.logical_or}
    return unary, binary


def _elementwise_reference(name, inputs, attrs, first, value, other):
    import torch

    unary, binary = _reference_functions()
    if name == "Signbit":
        # Inspect IEEE bits so -0.0 is distinguished from +0.0 on every backend.
        bits = first.view(torch.int16 if first.dtype == torch.float16 else torch.int32)
        result = bits < 0
        return result
    if name == "FloorDiv":
        result = _floor_divide_int32(first, inputs[1])
        return result
    if name == "ArgMax":
        # Resolve ties explicitly instead of relying on the backend argmax rule.
        columns = torch.arange(first.shape[1], dtype=torch.int64, device=first.device)
        maxima = torch.amax(value, dim=-1, keepdim=True)
        indices = torch.where(value == maxima, columns, first.shape[1])
        result = torch.amin(indices, dim=-1, keepdim=attrs.get("keepdim", False))
        return result
    if name in unary:
        result = unary[name](value)
        return result
    if name in binary:
        result = binary[name](value, other)
        return result
    if name == "Exp":
        exponent = value * attrs.get("scale", 1.0) + attrs.get("shift", 0.0)
        base = attrs.get("base", -1.0)
        result = exponent.exp() if base == -1.0 else torch.pow(base, exponent)
        return result
    if name == "Clamp":
        result = torch.clamp(value, attrs.get("lower", -0.5), attrs.get("upper", 0.5))
        return result
    if name == "Where":
        result = torch.where(first, inputs[1], inputs[2])
    else:
        return _row_reference(name, inputs, attrs, first, value, other)
    return result


def _row_reference(name, inputs, attrs, first, value, other):
    import torch

    if name in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ArgMax", "ArgMin"):
        function = {"ReduceSum": torch.sum, "ReduceMean": torch.mean, "ReduceMax": torch.amax,
                    "ReduceMin": torch.amin, "ArgMax": torch.argmax, "ArgMin": torch.argmin}[name]
        result = function(value, dim=-1, keepdim=attrs.get("keepdim", False))
        return result
    if name == "XorSum":
        result = value[:, 0].clone()
        for column in range(1, value.shape[1]):
            result = torch.bitwise_xor(result, value[:, column])
        if attrs.get("keepdim", False):
            result = result[:, None]
        return result
    if name in ("CumSum", "CumProd"):
        result = (torch.cumsum if name == "CumSum" else torch.cumprod)(value, dim=-1)
        return result
    if name == "Transpose":
        result = first.permute(1, 0).contiguous()
        return result
    if name == "Concat":
        result = torch.cat(inputs, dim=-1)
        return result
    if name == "Flip":
        result = torch.flip(first, [-1])
        return result
    if name == "Gather":
        result = torch.gather(first, -1, inputs[1])
        return result
    if name in ("SortValues", "TopKValues"):
        result = torch.sort(first, dim=-1, descending=attrs.get("descending", name == "TopKValues")).values
        if name == "TopKValues":
            result = result[:, :attrs.get("k", 1)].contiguous()
        return result
    if name == "Softmax":
        result = torch.softmax(value, dim=-1)
    else:
        return _nonlinear_reference(name, inputs, attrs, first, value, other)
    return result


def _nonlinear_reference(name, inputs, attrs, first, value, other):
    import torch

    if name in ("RmsNorm", "LayerNorm"):
        centered = value - value.mean(-1, keepdim=True) if name == "LayerNorm" else value
        result = centered * torch.rsqrt((centered * centered).mean(-1, keepdim=True)
                                        + attrs.get("epsilon", 1e-6)) * other
        if name == "LayerNorm":
            result = result + inputs[2].float()
        return result
    if name == "Matmul":
        result = value @ other
        return result
    if name == "Mish":
        result = value * torch.tanh(torch.log1p(torch.exp(value)))
        return result
    if name == "Gelu":
        if attrs.get("approximate", "none") == "none":
            result = value * 0.5 * (1 + torch.erf(value / math.sqrt(2)))
        else:
            result = value * 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi)
                                                   * (value + 0.044715 * value ** 3)))
        return result
    if name == "SwiGlu":
        gate, multiplier = value.chunk(2, dim=-1)
        result = gate * torch.sigmoid(gate) * multiplier
    else:
        raise ValueError(f"unimplemented official reference: {name}")
    return result


def bind(name):
    def bound_reference(*inputs, **attrs):
        return reference(name, *inputs, **attrs)

    def bound_inputs(*, seed, dtype, shape, device, attrs, value_range=None):
        return make_inputs(name, seed=seed, dtype=dtype, shape=shape, device=device,
                           attrs=attrs, value_range=value_range)

    bound_inputs.main_input_index = 1 if BY_NAME[name].family == "where" else 0
    return bound_reference, bound_inputs
