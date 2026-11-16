# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent NumPy/scalar-math oracle, with no Torch reference delegation."""

from __future__ import annotations

import math


def reference(name, *inputs, **attrs):
    import numpy as np
    import torch

    arrays = [tensor.detach().float().numpy().astype(np.float64) if tensor.is_floating_point()
              else tensor.detach().numpy() for tensor in inputs]
    value = arrays[0]
    other = arrays[1] if len(arrays) > 1 else None
    result = _elementwise_reference(name, arrays, attrs, value, other)
    if name in ("Equal", "NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual",
                "LogicalAnd", "LogicalOr", "Signbit"):
        dtype = torch.bool
    elif name in ("ArgMax", "ArgMin"):
        dtype = torch.int64
    else:
        dtype = inputs[1].dtype if name == "Where" else inputs[0].dtype
    return torch.from_numpy(np.array(result, copy=True)).to(dtype)


def _reference_functions():
    import numpy as np

    unary = {"Abs": np.abs, "Neg": np.negative, "Exp2": np.exp2, "Expm1": np.expm1,
             "Log": np.log, "Log2": np.log2, "Log10": np.log10, "Log1p": np.log1p,
             "Sqrt": np.sqrt, "Rsqrt": lambda array: 1 / np.sqrt(array),
             "Sin": np.sin, "Cos": np.cos, "Tan": np.tan, "Tanh": np.tanh,
             "Sinh": np.sinh, "Cosh": np.cosh, "Floor": np.floor, "Ceil": np.ceil,
             "Trunc": np.trunc, "Relu": lambda array: np.maximum(array, 0),
             "Sigmoid": lambda array: 1 / (1 + np.exp(-array)), "Signbit": np.signbit,
             "BitwiseNot": np.bitwise_not}
    binary = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide,
              "FloorDiv": np.floor_divide, "Mod": np.fmod, "Pow": np.power,
              "Maximum": np.maximum, "Minimum": np.minimum, "Equal": np.equal,
              "NotEqual": np.not_equal, "Greater": np.greater, "GreaterEqual": np.greater_equal,
              "Less": np.less, "LessEqual": np.less_equal, "BitwiseAnd": np.bitwise_and,
              "BitwiseOr": np.bitwise_or, "BitwiseXor": np.bitwise_xor,
              "LogicalAnd": np.logical_and, "LogicalOr": np.logical_or}
    return unary, binary


def _elementwise_reference(name, arrays, attrs, value, other):
    import numpy as np

    unary, binary = _reference_functions()
    if name in unary:
        result = unary[name](value)
        return result
    if name in binary:
        result = binary[name](value, other)
        return result
    if name == "Exp":
        exponent = value * attrs.get("scale", 1.0) + attrs.get("shift", 0.0)
        result = np.exp(exponent) if attrs.get("base", -1.0) == -1 else np.power(attrs["base"], exponent)
        return result
    if name == "Clamp":
        result = np.clip(value, attrs.get("lower", -0.5), attrs.get("upper", 0.5))
        return result
    if name == "Where":
        result = np.where(value, arrays[1], arrays[2])
    else:
        return _row_reference(name, arrays, attrs, value, other)
    return result


def _row_reference(name, arrays, attrs, value, other):
    import numpy as np

    if name in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ArgMax", "ArgMin", "XorSum"):
        function = {"ReduceSum": np.sum, "ReduceMean": np.mean, "ReduceMax": np.max,
                    "ReduceMin": np.min, "ArgMax": np.argmax, "ArgMin": np.argmin,
                    "XorSum": np.bitwise_xor.reduce}[name]
        result = function(value, axis=-1)
        if attrs.get("keepdim", False):
            result = result[:, None]
        return result
    if name in ("CumSum", "CumProd"):
        result = (np.cumsum if name == "CumSum" else np.cumprod)(value, axis=-1)
        return result
    if name == "Transpose":
        result = value.T
        return result
    if name == "Concat":
        result = np.concatenate(arrays, axis=-1)
        return result
    if name == "Flip":
        result = value[:, ::-1]
        return result
    if name == "Gather":
        result = np.take_along_axis(value, arrays[1], axis=-1)
        return result
    if name in ("SortValues", "TopKValues"):
        result = np.sort(value, axis=-1)
        if attrs.get("descending", name == "TopKValues"):
            result = result[:, ::-1]
        if name == "TopKValues":
            result = result[:, :attrs.get("k", 1)]
        return result
    if name == "Softmax":
        exponent = np.exp(value - np.max(value, axis=-1, keepdims=True))
        result = exponent / np.sum(exponent, axis=-1, keepdims=True)
    else:
        return _nonlinear_reference(name, arrays, attrs, value, other)
    return result


def _nonlinear_reference(name, arrays, attrs, value, other):
    import numpy as np

    if name in ("RmsNorm", "LayerNorm"):
        centered = value - np.mean(value, axis=-1, keepdims=True) if name == "LayerNorm" else value
        result = centered / np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True)
                                    + attrs.get("epsilon", 1e-6)) * other
        if name == "LayerNorm":
            result = result + arrays[2]
        return result
    if name == "Matmul":
        result = np.matmul(value, other)
        return result
    if name == "Mish":
        result = value * np.tanh(np.logaddexp(0, value))
        return result
    if name == "Gelu":
        if attrs.get("approximate", "none") == "none":
            result = value * 0.5 * (1 + np.vectorize(math.erf)(value / math.sqrt(2)))
        else:
            result = value * 0.5 * (1 + np.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value ** 3)))
        return result
    if name == "SwiGlu":
        gate, multiplier = np.split(value, 2, axis=-1)
        result = gate / (1 + np.exp(-gate)) * multiplier
    else:
        raise ValueError(f"missing independent oracle: {name}")
    return result


def bind(name):
    def bound_reference(*inputs, **attrs):
        return reference(name, *inputs, **attrs)

    return bound_reference
