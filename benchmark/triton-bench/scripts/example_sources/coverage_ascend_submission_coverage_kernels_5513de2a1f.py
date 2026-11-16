# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded extension kernels. Results require fresh compilation and NPU validation."""
__all__ = [
    '_copy',
    '_cross_entropy',
    '_cummin',
    '_grouped',
    '_loss_reduce',
    '_scatter',
    'cross_entropy_loss',
    'cummin',
    'grouped_matmul',
    'math',
    'scatter',
    'tl',
    'torch',
    'triton',
]

import math
import torch
import triton
import triton.language as tl
from scripts.templates.coverage_kernels import _copy
from scripts.templates.coverage_kernels import _scatter
from scripts.templates.coverage_kernels import scatter
from scripts.templates.coverage_kernels import _cummin
from scripts.templates.coverage_kernels import cummin
from scripts.templates.coverage_kernels import _grouped
from scripts.templates.coverage_kernels import grouped_matmul
from scripts.templates.coverage_kernels import _cross_entropy
from scripts.templates.coverage_kernels import _loss_reduce
from scripts.templates.coverage_kernels import cross_entropy_loss












def _cummin_implementation(input_value, dim):
    out = torch.empty(input_value.shape, dtype=input_value.dtype, device=input_value.device)
    indices = torch.empty(input_value.shape, dtype=torch.int64, device=input_value.device)
    axis = input_value.shape[dim]
    _cummin[(input_value.numel(),)](input_value, out, indices, axis, math.prod(input_value.shape[dim + 1:]),
                                    max(16, triton.next_power_of_2(axis)))
    return out, indices












def _cross_entropy_loss_implementation(input_value, target, reduction="mean", ignore_index=-100):
    rows, columns = input_value.shape
    loss = torch.empty((rows,), dtype=torch.float32, device=input_value.device)
    valid = torch.empty((rows,), dtype=torch.float32, device=input_value.device)
    out = torch.empty((), dtype=input_value.dtype, device=input_value.device)
    _cross_entropy[(rows,)](input_value, target, loss, valid, columns, ignore_index,
                            max(16, triton.next_power_of_2(columns)), enable_fp_fusion=False)
    _loss_reduce[(1,)](loss, valid, out, rows, reduction == "mean",
                       max(16, triton.next_power_of_2(rows)))
    return out


