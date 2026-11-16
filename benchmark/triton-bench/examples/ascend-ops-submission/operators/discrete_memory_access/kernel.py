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


@triton.jit
def _pick(x, indices, out, count: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    indices_value = tl.load(indices + offsets, offsets < count, other=0)
    values = tl.load(x + indices_value, offsets < count, other=0)
    tl.store(out + offsets, values, offsets < count)


def discrete_pick(x, indices, **attrs):
    output = torch.empty(indices.shape, dtype=x.dtype, device=x.device)
    _pick[(triton.cdiv(indices.numel(), 256),)](x, indices, output, indices.numel(), 256)
    return output
