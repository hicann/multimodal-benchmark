# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from triton_eval.reference_execution import binned_gather_reference
from triton_eval.inputs import expert_routing_inputs
import torch


def make_inputs(seed, dtype, shape, device, attrs):
    routing = expert_routing_inputs(seed, shape, device, attrs, legacy_shape=False)
    indices = routing.indices.to(torch.int32)
    counts = torch.bincount(routing.experts.long(), minlength=routing.expert_count).to(torch.int32)
    bins = torch.cumsum(counts, 0).to(torch.int32)
    weights = torch.rand(routing.experts.shape, device=routing.values.device).to(torch.float16)
    return [routing.values, indices, weights, bins, routing.bins]


def reference(x, indices, weights, bins, bin_ids, **attrs):
    top_k = int(attrs["top_k"])
    capacity = int(attrs["expert_capacity"])
    return binned_gather_reference(x, indices, bins, top_k, capacity)
