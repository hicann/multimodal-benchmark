# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Explicit bounded pattern implementations; host code only allocates and launches."""
__all__ = [
    '_grouped',
    '_cummin',
    '_invalid_grouped_parameters',
    '_matmul',
    '_pattern_bias_tile',
    '_pattern_max_finish',
    '_pattern_max_partial',
    '_pattern_min_pair',
    '_pattern_scan_carry',
    '_pattern_scan_finish',
    '_pattern_scan_local',
    '_pattern_split_finish',
    '_pattern_split_partial',
    'cummin',
    'grouped_matmul',
    'math',
    'matmul',
    'reduce_max',
    'tl',
    'torch',
    'triton',
]

import math

import torch
import triton
import triton.language as tl

from scripts.templates.compatibility_kernels import _grouped
from scripts.templates.coverage_kernels import _cummin, cummin
from scripts.templates.official_kernels import _matmul
from scripts.templates.target_pattern_kernels import (
    _invalid_grouped_parameters,
    _pattern_bias_tile,
    _pattern_max_finish,
    _pattern_max_partial,
    _pattern_min_pair,
    _pattern_scan_carry,
    _pattern_scan_finish,
    _pattern_scan_local,
    _pattern_split_finish,
    _pattern_split_partial,
    grouped_matmul,
    matmul,
    reduce_max,
)



























def _cummin_implementation(input_value, dim):
    if not input_value.is_contiguous():
        raise ValueError('Cummin candidate requires contiguous input')
    axis = input_value.shape[dim]
    inner = math.prod(input_value.shape[dim + 1:]
                      ) if dim >= 0 else math.prod(input_value.shape[len(input_value.shape) + dim + 1:])
    values = torch.empty(input_value.shape, dtype=input_value.dtype, device=input_value.device)
    indices = torch.empty(input_value.shape, dtype=torch.int64, device=input_value.device)
    if axis < 2**20:
        _cummin[(input_value.numel(),)](input_value, values, indices, axis, inner,
                                        max(16, triton.next_power_of_2(axis)))
        return values, indices
    chunks = triton.cdiv(axis, 512)
    lines = input_value.numel() // axis
    local = torch.empty(input_value.shape, dtype=torch.float32, device=input_value.device)
    local_indices = torch.empty(input_value.shape, dtype=torch.int64, device=input_value.device)
    totals = torch.empty((lines, chunks), dtype=torch.float32, device=input_value.device)
    total_indices = torch.empty((lines, chunks), dtype=torch.int64, device=input_value.device)
    carry = torch.empty((lines, chunks), dtype=torch.float32, device=input_value.device)
    carry_indices = torch.empty((lines, chunks), dtype=torch.int64, device=input_value.device)
    _pattern_scan_local[(chunks, lines)](input_value, local, local_indices,
                                         totals, total_indices, axis, inner, chunks, 512)
    _pattern_scan_carry[(lines,)](totals, total_indices, carry, carry_indices, chunks,
                                  triton.next_power_of_2(chunks))
    _pattern_scan_finish[(chunks, lines)](local, local_indices, carry, carry_indices, values, indices,
                                          axis, inner, chunks, 512)
    return values, indices












