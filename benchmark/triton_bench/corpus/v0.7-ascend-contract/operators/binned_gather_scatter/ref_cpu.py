# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch


def reference(x, indices, weights, bins, bin_ids, **attrs):
    capacity = int(attrs["expert_capacity"])
    experts = bins.numel()
    expanded = torch.repeat_interleave(x, int(attrs["top_k"]), dim=0)
    routed = expanded.index_select(0, indices.long())
    output = x.new_zeros((experts, capacity, x.shape[1]))
    starts = torch.cat((bins.new_zeros(1), bins[:-1])).long()
    slots = torch.arange(indices.numel()) - starts[bin_ids.long()]
    valid = slots < capacity
    output[bin_ids[valid].long(), slots[valid]] = routed[valid]
    return output
