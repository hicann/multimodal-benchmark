# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Gelu 的 triton 实现（合法提交样例）。

要点：
  - kernel 定义在模块级：@triton.jit 依赖 inspect.getsourcelines
  - 不调用 torch 内置 gelu：那会被判为代算（SUB-STR-001）
  - tanh 用 exp 表达：本环境已验证 tl.exp 可用，而 tanh 内建未经验证
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_kernel(x_ptr, o_ptr, n, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    # gelu 的 tanh 近似：0.5x(1 + tanh(0.7978845608 * (x + 0.044715x^3)))
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


def gelu(x, approximate="tanh"):
    """只在 tanh 近似下实现。

    exact（erf）路径未实现：erf 需要 libdevice 支持，未在本环境验证过。
    宁可明确报 NotImplementedError，也不用未经验证的近似去凑一个"通过"。
    """
    if approximate != "tanh":
        raise NotImplementedError(
            f"样例只实现 approximate='tanh'，实际收到 {approximate!r}"
        )
    # 非连续输入先挡住：kernel 线性寻址，硬跑是静默错误（2026-10-12 真机
    # 实测：gelu-f32-S85 静默算错 rel 6e+31）。.contiguous() 属布局搬运，
    # 逐元素类不允许用——见 exp/kernel.py 的说明。
    if not x.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    out = torch.empty_like(x)
    n = x.numel()
    block = 1024
    _gelu_tanh_kernel[(triton.cdiv(n, block),)](x, out, n, block=block)
    return out
