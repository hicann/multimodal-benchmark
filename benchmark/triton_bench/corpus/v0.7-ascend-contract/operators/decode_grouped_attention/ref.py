# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch


def make_inputs(seed, dtype, shape, device, attrs):
    torch.manual_seed(seed)
    batch, heads, dimension = shape
    length = 128
    kv_heads = 2
    splits = int(attrs.get("max_kv_splits", 4))
    q = torch.randn(shape, dtype=torch.bfloat16, device=device)
    keys = torch.randn((batch * length, kv_heads, dimension), dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    indptr = torch.arange(batch + 1, device=device, dtype=torch.int32) * length
    indices = torch.randperm(batch * length, device=device).to(torch.int32)
    counts = torch.full((batch,), splits, dtype=torch.int32, device=device)
    return [q, keys, values, indptr, indices, counts]


def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, **attrs):
    splits = int(attrs.get("max_kv_splits", 4))
    scale = float(attrs.get("sm_scale", 0.0)) or q.shape[-1] ** -0.5
    batch, heads, dimension = q.shape
    output = torch.zeros((batch, heads, splits, v_buffer.shape[-1]), dtype=torch.float32, device=q.device)
    lse = torch.zeros((batch, heads, splits), dtype=torch.float32, device=q.device)
    for request in range(batch):
        begin, end = int(kv_indptr[request]), int(kv_indptr[request + 1])
        count = int(num_kv_splits[request])
        chunk = (((end - begin + count - 1) // count + 31) // 32) * 32
        for split in range(count):
            start, stop = split * chunk, min((split + 1) * chunk, end - begin)
            if start >= stop:
                continue
            indices = kv_indices[begin + start:begin + stop].long()
            for head in range(heads):
                kv_head = head // (heads // k_buffer.shape[1])
                keys = k_buffer[indices, kv_head].float()
                values = v_buffer[indices, kv_head].float()
                scores = (keys * q[request, head].float()).sum(-1) * scale
                output[request, head, split] = (torch.softmax(scores, 0)[:, None] * values).sum(0)
                lse[request, head, split] = torch.logsumexp(scores, 0)
    return output, lse
