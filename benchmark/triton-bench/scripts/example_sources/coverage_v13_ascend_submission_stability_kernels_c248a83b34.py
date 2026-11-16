# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Cross entropy with device-side max subtraction before loss accumulation."""
__all__ = [
    '_stable_cross_entropy',
    '_stable_loss_reduce',
    'cross_entropy_loss',
    'tl',
    'torch',
    'triton',
]

import torch
import triton
import triton.language as tl

from scripts.templates.coverage_kernels import cross_entropy_loss
from scripts.templates.stability_kernels import (
    _stable_cross_entropy,
    _stable_loss_reduce,
)








def _cross_entropy_loss_implementation(input_value, target, reduction='mean', ignore_index=-100):
    if reduction not in ('sum', 'mean'):
        raise ValueError('coverage sample supports sum or mean reduction')
    rows, columns = input_value.shape
    losses = torch.empty((rows,), dtype=torch.float32, device=input_value.device)
    valid = torch.empty((rows,), dtype=torch.float32, device=input_value.device)
    output = torch.empty((), dtype=input_value.dtype, device=input_value.device)
    _stable_cross_entropy[(rows,)](input_value, target, losses, valid, columns, ignore_index,
                                   max(16, triton.next_power_of_2(columns)), enable_fp_fusion=False)
    _stable_loss_reduce[(1,)](losses, valid, output, rows, reduction == 'mean',
                              max(16, triton.next_power_of_2(rows)))
    return output


