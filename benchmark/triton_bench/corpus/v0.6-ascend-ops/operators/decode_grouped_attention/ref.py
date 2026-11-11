# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tut import load_tutorial

_t = load_tutorial("best_practice/002-decode_grouped_attention.py")

# ⚠️ 真机验证提示：本参考实现是对 decode_grouped_attention_stage1 的「split-K 部分注意力」
# 的 torch 移植（每 split 覆盖 32 个 KV token，计算块内 e_max / 部分分子 / logsumexp）。
# 因无法在本机 NPU 验证，请在上线真机后首轮专门检查 DecodeGroupedAttention 的 20 条用例；
# 若参考与提交实现不一致，优先核对此处 split 边界与 e_max 计算顺序。


def make_inputs(seed, dtype, shape, device, attrs):
    torch.manual_seed(seed)
    b, hq, d = shape
    sequence_length = 128
    kv_heads = 1
    value_width = d
    max_kv_splits = int(attrs.get("max_kv_splits", 4))
    dev = device if isinstance(device, torch.device) else torch.device(device)
    q = torch.randn((b, hq, d), dtype=torch.bfloat16, device=dev)
    k_buffer = torch.randn((b * sequence_length, kv_heads, d), dtype=torch.bfloat16, device=dev)
    v_buffer = torch.randn((b * sequence_length, kv_heads, value_width), dtype=torch.bfloat16, device=dev)
    b_seq_len = torch.full((b,), sequence_length, dtype=torch.int32, device=dev)
    kv_indptr = torch.zeros((b + 1,), dtype=torch.int32, device=dev)
    kv_indptr[1:b + 1] = torch.cumsum(b_seq_len[:b], 0)
    kv_indices = torch.arange(b * sequence_length, dtype=torch.int32, device=dev)
    num_kv_splits = torch.full((b,), max_kv_splits, dtype=torch.int32, device=dev)
    return [q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits]


def _reference_head(query, keys, values, max_kv_splits, sm_scale, output, logsumexp):
    kv_len = keys.shape[0]
    for split in range(max_kv_splits):
        start = split * 32
        end = min(start + 32, kv_len)
        if start >= kv_len:
            break
        key_block = keys[start:end, 0, :].float()
        value_block = values[start:end, 0, :].float()
        scores = (key_block @ query) * sm_scale
        e_max = scores.max()
        exponentials = torch.exp(scores - e_max)
        e_sum = exponentials.sum()
        accumulated = (exponentials[:, None] * value_block).sum(0)
        output[split] = accumulated / e_sum
        logsumexp[split] = e_max + torch.log(e_sum)


def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, **attrs):
    max_kv_splits = int(attrs.get("max_kv_splits", 4))
    sm_scale = float(attrs.get("sm_scale", 0.0)) or (1.0 / (q.shape[-1] ** 0.5))
    batch_size, query_heads, _head_width = q.shape
    value_width = v_buffer.shape[-1]
    # 必须显式带 device：默认落在 CPU，与 NPU 上的提交输出比对会抛
    # "Expected all tensors to be on the same device"，整算子 20 例 ORACLE_FAIL。
    att_out = torch.zeros((batch_size, query_heads, max_kv_splits, value_width), dtype=torch.float32, device=q.device)
    att_lse = torch.zeros((batch_size, query_heads, max_kv_splits), dtype=torch.float32, device=q.device)
    for b in range(batch_size):
        s0 = int(kv_indptr[b])
        s1 = int(kv_indptr[b + 1])
        for h in range(query_heads):
            qh = q[b, h].float()
            _reference_head(qh, k_buffer[s0:s1], v_buffer[s0:s1], max_kv_splits, sm_scale,
                            att_out[b, h], att_lse[b, h])
    return att_out, att_lse
