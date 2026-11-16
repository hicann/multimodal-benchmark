# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Cross entropy with device-side max subtraction before loss accumulation."""
__all__ = ['cross_entropy_loss', 'tl', 'torch', 'triton']

import torch
import triton
import triton.language as tl

from scripts.templates.coverage_kernels import cross_entropy_loss




@triton.jit
def _stable_cross_entropy(input_value, target, loss, valid, columns: tl.constexpr,
                          ignore: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    columns_value = tl.arange(0, block)
    value = tl.load(input_value + row * columns + columns_value, columns_value < columns,
                    other=-float('inf')).to(tl.float32)
    maximum = tl.max(value, 0)
    label = tl.load(target + row)
    selected = tl.load(input_value + row * columns + label, label != ignore,
                       other=0).to(tl.float32)
    logarithm = tl.log(tl.sum(tl.exp(value - maximum), 0))
    loss_value = logarithm - (selected - maximum)
    tl.store(loss + row, tl.where(label != ignore, loss_value, 0))
    tl.store(valid + row, (label != ignore).to(tl.float32))


@triton.jit
def _stable_loss_reduce(loss, valid, output, rows: tl.constexpr,
                        mean: tl.constexpr, block: tl.constexpr):
    rows_value = tl.arange(0, block)
    total = tl.sum(tl.load(loss + rows_value, rows_value < rows, other=0), 0)
    if mean:
        count = tl.sum(tl.load(valid + rows_value, rows_value < rows, other=0), 0)
        total = total / count
    tl.store(output, total)


def _cross_entropy_loss_implementation(input_, target, reduction='mean', ignore_index=-100):
    if reduction not in ('sum', 'mean'):
        raise ValueError('coverage sample supports sum or mean reduction')
    rows, columns = input_.shape
    losses = torch.empty((rows,), dtype=torch.float32, device=input_.device)
    valid = torch.empty((rows,), dtype=torch.float32, device=input_.device)
    output = torch.empty((), dtype=input_.dtype, device=input_.device)
    _stable_cross_entropy[(rows,)](input_, target, losses, valid, columns, ignore_index,
                                  max(16, triton.next_power_of_2(columns)), enable_fp_fusion=False)
    _stable_loss_reduce[(1,)](losses, valid, output, rows, reduction == 'mean',
                             max(16, triton.next_power_of_2(rows)))
    return output


