# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""New workload domains; inherited cases retain their previous input guards."""

from __future__ import annotations


def _qualified(operator, first, attrs):
    from triton_eval.workload_generation import candidate_pool

    dtype = str(first.dtype).removeprefix("torch.")
    for case in candidate_pool():
        if case.operator != operator or case.dtype != dtype:
            continue
        if case.shape == list(first.shape) and case.attrs == attrs:
            return True
    return False


def _consistent_tensor(tensor, first):
    import torch

    if not isinstance(tensor, torch.Tensor):
        return False
    return tensor.dtype == first.dtype and tensor.device == first.device and tensor.is_contiguous()


def _scan_inputs(inputs, attrs):
    first = inputs[0]
    if first.ndim != 3 or not 0 <= attrs["dim"] < 3:
        raise ValueError("workload scan axis out of domain")
    if not 2 <= first.shape[attrs["dim"]] <= 32:
        raise ValueError("workload scan axis out of domain")


def _grouped_inputs(inputs, attrs):
    first = inputs[0]
    groups = attrs["group_list"]
    if min(groups) <= 0 or sum(groups) != first.shape[0]:
        raise ValueError("workload group partition/weight/bias mismatch")
    if inputs[1].shape != (len(groups), 32, 16) or inputs[2].shape != (len(groups), 16):
        raise ValueError("workload group partition/weight/bias mismatch")


def validate_inputs(operator, inputs, attrs):
    from triton_eval.pattern_contracts import validate_inputs as previous
    from triton_eval.operator_registry import by_name

    first = inputs[0]
    if not _qualified(operator, first, attrs) or operator not in ("Cummin", "GroupedMatmul"):
        previous(operator, inputs, attrs)
        return
    spec = by_name(operator)
    spec.validate_case_params(str(first.dtype).removeprefix("torch."), list(first.shape), attrs)
    if len(inputs) != len(spec.inputs) or not all(
        _consistent_tensor(tensor, first) for tensor in inputs
    ):
        raise ValueError("workload input dtype/device/layout mismatch")
    validators = {"Cummin": _scan_inputs, "GroupedMatmul": _grouped_inputs}
    validators.get(operator)(inputs, attrs)
