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
def _rearrange(x, indices, out, s: tl.constexpr, d: tl.constexpr,
               rows: tl.constexpr, cols: tl.constexpr):
    rows_value = tl.program_id(0) * rows + tl.arange(0, rows)
    columns = tl.arange(0, cols)
    indices_value = tl.load(indices + rows_value, rows_value < s, other=0)
    output = tl.full((rows, cols), 0, tl.float32)
    for lane in tl.static_range(rows):
        index = extension.get_element(indices_value, (lane,))
        values = tl.load(x + index * d + columns[None, :],
                         (tl.program_id(0) * rows + lane < s)
                         & (columns[None, :] < d), other=0)
        output = extension.insert_slice(output, values, (lane, 0),
                                        (1, cols), (1, 1))
    tl.store(out + rows_value[:, None] * d + columns[None, :], output,
             (rows_value[:, None] < s) & (columns[None, :] < d))


def token_rearrange(x, indices, **attrs):
    rows, columns = x.shape
    output = torch.empty_like(x)
    _rearrange[(triton.cdiv(rows, 8),)](
        x, indices, output, rows, columns, 8, triton.next_power_of_2(columns))
    return output
