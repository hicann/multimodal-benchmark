# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch


def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, **attrs):
    batch, heads, dimension = q.shape
    splits = int(attrs.get("max_kv_splits", 4))
    scale = float(attrs.get("sm_scale", 0.0)) or dimension ** -0.5
    output = torch.zeros((batch, heads, splits, v_buffer.shape[-1]), dtype=torch.float32)
    lse = torch.zeros((batch, heads, splits), dtype=torch.float32)
    mapping = torch.arange(heads) // (heads // k_buffer.shape[1])
    for request in range(batch):
        start, end = int(kv_indptr[request]), int(kv_indptr[request + 1])
        count = int(num_kv_splits[request])
        chunk = (((end - start + count - 1) // count + 31) // 32) * 32
        route = kv_indices[start:end].long()
        keys = k_buffer[route].double()[:, mapping].transpose(0, 1)
        values = v_buffer[route].double()[:, mapping].transpose(0, 1)
        scores = torch.einsum("hd,hnd->hn", q[request].double(), keys) * scale
        for split in range(count):
            begin, stop = split * chunk, min((split + 1) * chunk, end - start)
            if begin < stop:
                selected = scores[:, begin:stop]
                normalizer = torch.logsumexp(selected, -1)
                probabilities = torch.exp(selected - normalizer[:, None])
                output[request, :, split] = torch.einsum(
                    "hn,hnd->hd", probabilities, values[:, begin:stop]).float()
                lse[request, :, split] = normalizer.float()
    return output, lse
