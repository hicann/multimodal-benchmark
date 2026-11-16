# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoeGatingTopKSoftmax 的 triton 实现（合法提交样例）。

softmax(x, -1) → topK：门控值、专家下标、行位置一次算完。
三输出 (values, expert_indices, row_indices)；row_indices 是纯位置量
（与输入无关），host 侧直接构造。

要点：
  - 每个 program 处理一行（专家维 E 单块装下，BLOCK=next_pow2(E)，
    覆盖主流路由规模至 E=256）：
    稳定 softmax（先减行最大值），再 k 轮选择——每轮取当前行最大值
    及其下标，写输出后把该位置置 -inf
  - k 轮选择在 kernel 内完成：与 softmax 的中间结果同在寄存器，
    不落显存——这正是该融合算子的考点
  - 并列值：tl.argmax 的并列取最低下标（与 torch.topk 的 CPU 行为
    一致）；冻结用例限定 fp32/bf16，随机连续分布下并列概率可忽略
  - 按行 stride 寻址（x 可非连续）；输出自建为连续
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_gating_kernel(x_ptr, v_ptr, e_ptr, geometry, configuration: tl.constexpr):
    experts, k, a1, a2, ac, r1, r2 = geometry
    (block,) = configuration
    p = tl.program_id(0)
    i2 = p % r2
    t = p // r2
    i1 = t % r1
    base = i1 * a1 + i2 * a2
    offs = tl.arange(0, block)
    valid = offs < experts
    x = tl.load(x_ptr + base + offs * ac, mask=valid, other=float("-inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    psum = tl.sum(tl.exp(x - m), axis=0)
    probs = tl.exp(x - m) / psum
    probs = tl.where(valid, probs, float("-inf"))
    for j in range(k):
        val = tl.max(probs, axis=0)
        idx = tl.argmax(probs, axis=0)
        tl.store(v_ptr + p * k + j, val.to(v_ptr.dtype.element_ty))
        tl.store(e_ptr + p * k + j, idx.to(tl.int64))
        probs = tl.where(offs == idx, float("-inf"), probs)


def _outer(shape, strides):
    """最内维以外的维度，前补齐到 2 个（与 add_rms_norm/kernel.py 同款）。"""
    os_, ob = tuple(shape[:-1]) or (1,), tuple(strides[:-1]) or (0,)
    r2 = os_[-1] if len(os_) >= 1 else 1
    r1 = os_[-2] if len(os_) >= 2 else 1
    b2 = ob[-1] if len(ob) >= 1 else 0
    b1 = ob[-2] if len(ob) >= 2 else 0
    return r1, r2, b1, b2


def moe_gating_top_k_softmax(x, k):
    if x.dim() > 3:
        raise NotImplementedError(f"样例只实现 rank≤3，实际 rank={x.dim()}")
    experts = x.shape[-1]
    k = int(k)
    if not (0 < k <= experts):
        raise NotImplementedError(f"k={k} 超出 (0, {experts}]")
    rows = x.numel() // experts
    device = x.device
    # 参考按输入形状出 topk 结果（(T,k) / (B,T,k)）：直接按该形状分配，
    # 内核按扁平行号 p 写入——行主序展平偏移与 (rows,k) 完全一致
    out_shape = tuple(x.shape[:-1]) + (k,)
    values = torch.empty(out_shape, dtype=x.dtype, device=device)
    expert_idx = torch.empty(out_shape, dtype=torch.int64, device=device)
    # 行位置下标：与输入无关的纯位置量（池 golden 公式的展平规则：
    # row_idx[..., j] = 行展平号 + rows·j）。用加法广播直接构造，不经
    # reshape/transpose/view（FusedComposite 类静态扫描禁止布局变换调用，
    # 即便它只是输出构造；广播加法无需搬运）
    if x.dim() == 2:
        row_idx = (torch.arange(rows, dtype=torch.int64, device=device)
                   .unsqueeze(1)
                   + rows * torch.arange(k, dtype=torch.int64,
                                         device=device).unsqueeze(0))
    else:
        b_, t_ = x.shape[0], x.shape[1]
        row_idx = (torch.arange(b_, dtype=torch.int64, device=device)
                   .unsqueeze(1).unsqueeze(2) * t_
                   + torch.arange(t_, dtype=torch.int64, device=device)
                   .unsqueeze(0).unsqueeze(2)
                   + rows * torch.arange(k, dtype=torch.int64,
                                         device=device)
                   .unsqueeze(0).unsqueeze(0))

    r1, r2, a1, a2 = _outer(x.shape, x.stride())
    block = max(16, triton.next_power_of_2(experts))
    _moe_gating_kernel[rows,](x, values, expert_idx, (experts, k, a1, a2, x.stride(-1), r1, r2), (block,))
    return values, expert_idx, row_idx
