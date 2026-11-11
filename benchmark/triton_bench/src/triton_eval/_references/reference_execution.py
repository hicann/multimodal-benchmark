# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Strict execution policy; the reference function and equality gate stay intact."""
from __future__ import annotations

from contextlib import contextmanager


POLICY = 'strict-npu-cross-entropy-determinism-v1'


def assign_token_pool_reference(requests, pool, starts, ends, locations, batch):
    result = pool.clone()
    previous = 0
    for index in range(batch):
        start = int(starts[index])
        end = int(ends[index])
        request = int(requests[index])
        length = end - start
        result[request, start:end] = locations[previous:previous + length]
        previous += int(ends[index] - starts[index])
    return result


def binned_gather_reference(values, indices, bins, top_k, capacity):
    experts = int(bins.numel())
    columns = values.shape[1]
    result = values.new_zeros((experts, capacity, columns))
    start = 0
    for expert in range(experts):
        end = int(bins[expert])
        count = min(capacity, end - start)
        if count > 0:
            rows = (indices[start:start + count] // top_k).long()
            result[expert, :count, :] = values[rows, :]
        start = end
    return result


def padded_gather_reference(values, indices, bins, padded_bins, top_k):
    columns = values.shape[1]
    result = values.new_zeros((int(padded_bins[-1]), columns))
    previous = 0
    for expert in range(int(bins.numel())):
        end = int(bins[expert])
        destination = 0 if expert == 0 else int(padded_bins[expert - 1])
        count = end - previous
        if count > 0:
            rows = (indices[previous:end] // top_k).long()
            result[destination:destination + count, :] = values[rows, :]
        previous = end
    return result


@contextmanager
def reference_execution(operator, inputs):
    import torch

    enabled = (operator == 'CrossEntropyLoss' and inputs
               and str(inputs[0].device).startswith('npu'))
    if not enabled:
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
        torch.npu.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warning)
