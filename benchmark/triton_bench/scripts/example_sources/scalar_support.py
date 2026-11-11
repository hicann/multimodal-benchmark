# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Pure scalar fixture scaffolding; exported attacks retain their own bad behavior."""
__all__ = [
    '_walk',
    'math',
    'product',
]

import math
from itertools import product

from scripts.example_sources._shared_escapee_10c7261575 import _walk






def softmax_values(tensor, dim):
    shape = list(tensor.shape)
    axis = dim if dim >= 0 else dim + len(shape)
    width = shape[axis]
    flat = list(_walk(tensor))
    outer = math.prod(shape[:axis])
    inner = math.prod(shape[axis + 1:])
    output = [0.0] * len(flat)
    for combined_index in range(outer * inner):
        outer_index, inner_index = divmod(combined_index, inner)
        base = outer_index * width * inner + inner_index
        segment = [flat[base + index * inner] for index in range(width)]
        maximum = max(segment)
        exponentials = [math.exp(value - maximum) for value in segment]
        total = sum(exponentials)
        for index, value in enumerate(exponentials):
            output[base + index * inner] = value / total
    return shape, output


def cummin_values(tensor, dim):
    shape = list(tensor.shape)
    axis = dim if dim >= 0 else dim + len(shape)
    width = shape[axis]
    flat = list(_walk(tensor))
    inner = 1
    for size in shape[axis + 1:]:
        inner *= size
    outer = len(flat) // (width * inner) if width * inner else 1
    values, indices = [0.0] * len(flat), [0] * len(flat)
    for outer_index, inner_index in product(range(outer), range(inner)):
        minimum, minimum_index = None, 0
        for index in range(width):
            position = (outer_index * width + index) * inner + inner_index
            value = flat[position]
            if minimum is None or value <= minimum:
                minimum, minimum_index = value, index
            values[position], indices[position] = minimum, minimum_index
    return shape, values, indices
