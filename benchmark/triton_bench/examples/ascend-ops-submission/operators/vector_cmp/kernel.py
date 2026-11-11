# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

import triton
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tut import load_tutorial

_t = load_tutorial("basic/002-vector_cmp.py")
_kernel = _t.npu_vector_cmp_kernel


def vector_cmp(x, **attrs):
    eps = float(attrs.get("eps", 1e-6))
    rows, columns = x.shape
    output = torch.empty_like(x)
    mean = torch.empty((rows,), dtype=x.dtype, device=x.device)
    reciprocal_std = torch.empty((rows,), dtype=x.dtype, device=x.device)
    block_rows = 1
    block_columns = triton.next_power_of_2(columns)
    num_warps = min(max(block_columns // 256, 1), 8)
    _kernel[(rows // block_rows, 1)](
        x, output, mean, reciprocal_std, x.stride(0), output.stride(0),
        rows, columns, eps, block_rows, block_columns, num_warps=num_warps)
    return output, mean, reciprocal_std
