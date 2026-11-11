# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.jit
def _reverse(x, indices, out, s: tl.constexpr, d: tl.constexpr,
             rows: tl.constexpr, cols: tl.constexpr):
    rows_value = tl.program_id(0) * rows + tl.arange(0, rows)
    columns = tl.arange(0, cols)
    values = tl.load(x + rows_value[:, None] * d + columns[None, :],
                     (rows_value[:, None] < s) & (columns[None, :] < d), other=0)
    indices_value = tl.load(indices + rows_value, rows_value < s, other=0)
    for lane in tl.static_range(rows):
        row_values = extension.extract_slice(values, (lane, 0),
                                             (1, cols), (1, 1))
        index = extension.get_element(indices_value, (lane,))
        tl.store(out + index * d + columns[None, :], row_values,
                 (tl.program_id(0) * rows + lane < s)
                 & (columns[None, :] < d))


def token_reverse(x, indices, **attrs):
    rows, columns = x.shape
    output = torch.empty_like(x)
    _reverse[(triton.cdiv(rows, 8),)](
        x, indices, output, rows, columns, 8, triton.next_power_of_2(columns))
    return output
