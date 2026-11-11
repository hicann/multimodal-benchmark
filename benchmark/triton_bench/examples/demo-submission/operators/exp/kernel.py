# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exp 的 triton 实现（合法提交样例）。

要点：
  - kernel 定义在模块级：@triton.jit 依赖 inspect.getsourcelines
  - 不调用 torch 内置 exp/pow 等：那会被判为代算（SUB-STR-001）
  - 不传 num_warps / num_stages：昇腾不认这两个 NVIDIA 概念
  - base != -1 时用 exp(z * ln(base)) 算幂，而不是调 torch.pow——
    模板自己必须经得起它要教的那套规则
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _exp_kernel(x_ptr, o_ptr, n, scale, shift, log_base, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    z = (x * scale + shift) * log_base
    # 显式转回输出 dtype：隐式转换在部分后端会报错，
    # 且与参考实现「fp32 计算 → 转回目标 dtype」的路径保持一致。
    tl.store(o_ptr + offs, tl.exp(z).to(o_ptr.dtype.element_ty), mask=mask)


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    # 非连续输入先挡住：kernel 按连续内存线性寻址（x_ptr + offs），
    # 硬跑会读错内存——那是静默错误（最坏的失败形态），宁可明确报未实现。
    # 2026-10-12 真机实测教训：v0.4 的内存布局维度（contiguous=false）
    # 恰好考住这一点，exp-f32-S85 曾静默算错（rel 3.7e+15）。
    # 注：逐元素类不允许用 .contiguous() 绕道（布局搬运仅布局类算子可用）。
    if not x.is_contiguous():
        raise NotImplementedError(
            "样例内核按连续内存线性寻址，未实现 stride 寻址；非连续输入请自行实现")
    # pow(base, z) == exp(z * ln(base))；base 为 -1（自然指数）时 log_base=1
    log_base = 1.0 if base in (None, -1.0) else math.log(float(base))
    out = torch.empty_like(x)
    n = x.numel()
    block = 1024
    _exp_kernel[(triton.cdiv(n, block),)](
        x, out, n, float(scale), float(shift), float(log_base), block=block
    )
    return out
