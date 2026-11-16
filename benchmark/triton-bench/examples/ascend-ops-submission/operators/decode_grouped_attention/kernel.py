# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(q, k, v, indptr, indices, splits, out, lse, configuration: tl.constexpr):
    hq, hkv, d, dv, max_splits, scale, bd, bv = configuration
    batch = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    begin = tl.load(indptr + batch)
    end = tl.load(indptr + batch + 1)
    count = tl.load(splits + batch)
    chunk = tl.cdiv(tl.cdiv(end - begin, count), 32) * 32
    start = split * chunk
    stop = tl.minimum(start + chunk, end - begin)
    columns = tl.arange(0, bd)
    value_columns = tl.arange(0, bv)
    rows = tl.arange(0, 32)
    q_value = tl.load(q + (batch * hq + head) * d + columns, columns < d, other=0).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    numerator = tl.full((bv,), 0, tl.float32)
    kv_head = head // (hq // hkv)
    if split < count and start < stop:
        for block in range(start, stop, 32):
            positions = block + rows
            valid = positions < stop
            indices_value = tl.load(indices + begin + positions, valid, other=0)
            keys = tl.load(
                k + (indices_value[:, None] * hkv + kv_head) * d + columns[None, :],
                valid[:, None] & (columns[None, :] < d),
                other=0,
            ).to(tl.float32)
            scores = tl.sum(keys * q_value[None, :], 1) * scale
            scores = tl.where(valid, scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, 0))
            alpha = tl.exp(maximum - new_maximum)
            probabilities = tl.exp(scores - new_maximum)
            values = tl.load(
                v + (indices_value[:, None] * hkv + kv_head) * dv + value_columns[None, :],
                valid[:, None] & (value_columns[None, :] < dv),
                other=0,
            ).to(tl.float32)
            numerator = numerator * alpha + tl.sum(probabilities[:, None] * values, 0)
            denominator = denominator * alpha + tl.sum(probabilities, 0)
            maximum = new_maximum
        result = numerator / denominator
        logsumexp = maximum + tl.log(denominator)
    else:
        result = tl.full((bv,), 0, tl.float32)
        logsumexp = tl.full((), 0, tl.float32)
    offset = (batch * hq + head) * max_splits + split
    tl.store(out + offset * dv + value_columns, result, value_columns < dv)
    tl.store(lse + offset, logsumexp)


def decode_grouped_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, **attrs):
    splits = int(attrs.get("max_kv_splits", 4))
    scale = float(attrs.get("sm_scale", 0.0)) or q.shape[-1] ** -0.5
    batch, heads, dimension = q.shape
    values = v_buffer.shape[-1]
    output = torch.empty((batch, heads, splits, values), dtype=torch.float32, device=q.device)
    lse = torch.empty((batch, heads, splits), dtype=torch.float32, device=q.device)
    _attention[batch, heads, splits](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        output,
        lse,
        (
            heads,
            k_buffer.shape[1],
            dimension,
            values,
            splits,
            scale,
            triton.next_power_of_2(dimension),
            triton.next_power_of_2(values),
        ),
        enable_fp_fusion=False,
    )
    return output, lse
