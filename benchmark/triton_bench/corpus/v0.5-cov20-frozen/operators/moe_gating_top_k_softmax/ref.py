# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoeGatingTopKSoftmax 的参考实现（评测方资产）。

来源池 tasks/level3/moe_gating_top_k_softmax（公式与 row_idx 规则忠实于
池 golden）：
    softmaxOut = softmax(x, axis=-1)
    values, expertIdx = topK(softmaxOut, k)
    rowIdx = arange(N*k).reshape(k, N).transpose(0, 1)   # 展平后的全局位置
池的可选输入 finished（布尔掩码）不收录（proto 标 optional）。

输出契约：三输出的元组，执行器按现行契约只比对第一个张量（values）——
与 Cummin 的已披露缺口同源：indices 路正确性的判定待多输出比对机制。

并列值防歧义：dtype 不含 fp16（注册表已限定 fp32/bf16）——fp16 下 softmax
概率易量化出并列，topk 选中下标在实现与参考间会合法地分歧；fp32/bf16
下由随机种子驱动的连续分布 logits 并列概率可忽略。
"""

from __future__ import annotations

import torch


def _rand(seed: int, n: int, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, generator=g).to(dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    n = 1
    for d in shape:
        n *= d
    return [_rand(seed, n, dtype).reshape(shape).to(device)]


def reference(x, k):
    k = int(k)
    sm = torch.softmax(x.to(torch.float32), dim=-1)
    values, expert_idx = torch.topk(sm, k, dim=-1)

    out_shape = expert_idx.shape
    if len(out_shape) == 2:
        # (N, k)：row_idx[i, j] = j*N + i（输出位置的展平全局索引）
        # device 必须显式取自结果张量：2026-10-13 真机实测，默认建在 CPU
        # 上的 row_idx 与 NPU 输出比对时抛设备不匹配、杀死评测子进程
        rng = torch.arange(out_shape[0] * out_shape[1], dtype=torch.int64,
                           device=values.device)
        row_idx = rng.reshape(out_shape[1], out_shape[0]).transpose(0, 1)
    else:
        # (B, N, k)：先按 (B*N, k) 展平计算，再还原形状
        b, nn = out_shape[0], out_shape[1]
        rng = torch.arange(b * nn * k, dtype=torch.int64,
                           device=values.device)
        row_idx = rng.reshape(k, b * nn).transpose(0, 1).reshape(out_shape)
    return values.to(x.dtype), expert_idx.to(torch.int64), row_idx
