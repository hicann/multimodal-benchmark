# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ApplyRotaryPosEmb 的参考实现（评测方资产）。

来源池 tasks/level2/apply_rotary_pos_emb（公式与旋转模式忠实于池 golden）：
    rotate_half(x) = concat(-x[D/2:], x[:D/2])          # half（LLaMA 连续半分式）
    rotate_half(x) = stack(-x[1::2], x[::2]).flatten     # interleaved（GPT-J 交错式）
    y = x * cos_full + rotate_half(x) * sin_full

cos/sin 的形状约定（本体系的简化契约，数值语义与池一致）：
    cos/sin 形如 x.shape[:-1] + [D/2]，即除最后一维外与 x 同形；
    展开到 D 维的方式：half 模式为 concat(c, c)，interleaved 模式为
    repeat_interleave(c, 2)（池 golden 同款）。
    池的 layout 属性只影响 [B,S,N,D] 与 [B,N,S,D] 的排布约定，不影响末维
    旋转的数学（本体系对任意 rank 按末维旋转），故不收录。

计算精度：fp32 域计算后转回目标 dtype（与全语料约定一致；池 golden 对
fp16/bf16 同样升 fp32 计算）。
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
    d_last = shape[-1]
    x = _rand(seed, n, dtype).reshape(shape).to(device)
    # 角度在 [-π, π]：均匀覆盖旋转的完整周期，避免极端角度的精度放大
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    angles = (torch.rand(n // d_last * (d_last // 2), generator=g) * 2 - 1) \
        * torch.pi
    half = angles.reshape(list(shape[:-1]) + [d_last // 2]).to(dtype).to(device)
    cos = torch.cos(half.to(torch.float32)).to(dtype)
    sin = torch.sin(half.to(torch.float32)).to(dtype)
    return [x, cos, sin]


def _rotate_half(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "interleaved":
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    half_dim = x.shape[-1] // 2
    return torch.cat([-x[..., half_dim:], x[..., :half_dim]], dim=-1)


def _expand_tables(t: torch.Tensor, mode: str) -> torch.Tensor:
    """D/2 → D：half 模式两半同值，interleaved 模式相邻成对。"""
    if mode == "interleaved":
        return t.repeat_interleave(2, dim=-1)
    return torch.cat([t, t], dim=-1)


def reference(x, cos, sin, rotary_mode="half"):
    x32 = x.to(torch.float32)
    cos_full = _expand_tables(cos.to(torch.float32), rotary_mode)
    sin_full = _expand_tables(sin.to(torch.float32), rotary_mode)
    y = x32 * cos_full + _rotate_half(x32, rotary_mode) * sin_full
    return y.to(x.dtype)
