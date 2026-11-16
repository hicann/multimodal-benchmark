# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Narrow input-domain additions for the opt-in pattern release."""
from __future__ import annotations


def _invalid_group_shapes(groups, first, inputs):
    if (min(groups) < 1):
        return True
    if (sum(groups) != first.shape[0]):
        return True
    if (inputs[1].shape != (len(groups), first.shape[1], 16)):
        return True
    return bool(inputs[2].shape != (len(groups), 16))


def validate_inputs(operator, inputs, attrs):
    import torch

    from .coverage_reference import validate_inputs as validate_previous
    from .operator_registry import by_name
    from .pattern_generation import candidate_plans

    if operator not in ('Cummin', 'GroupedMatmul'):
        validate_previous(operator, inputs, attrs)
        return
    spec = by_name(operator)
    if len(inputs) != len(spec.inputs) or any(not isinstance(tensor, torch.Tensor) for tensor in inputs):
        raise ValueError('pattern input arity/type mismatch')
    first = inputs[0]
    dtype = str(first.dtype).removeprefix('torch.')
    qualified = any(name == operator and declared_dtype == dtype and shape == list(first.shape)
                    and parameters == attrs for name, declared_dtype, shape, parameters in candidate_plans())
    if not qualified:
        validate_previous(operator, inputs, attrs)
        return
    spec.validate_case_params(dtype, list(first.shape), attrs)
    if any(tensor.dtype != first.dtype or tensor.device != first.device or not tensor.is_contiguous()
           for tensor in inputs):
        raise ValueError('pattern tensors require matching dtype/device and contiguous layout')
    if operator == 'Cummin':
        if first.ndim != 2 or attrs['dim'] % first.ndim != 1:
            raise ValueError('pattern scan requires a two-dimensional last-axis input')
    else:
        groups = attrs['group_list']
        if (_invalid_group_shapes(groups, first, inputs)):
            raise ValueError('pattern grouped weight/bias/partition mismatch')
