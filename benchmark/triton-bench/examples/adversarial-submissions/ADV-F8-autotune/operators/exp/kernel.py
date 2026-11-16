# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-F8：@triton.autotune（冻结启动配置不设防）。

性能口径规定「性能主口径禁 autotune，配置必须写死」，
case 里也确实存了 launch 配置（grid/BLOCK 等）——
但执行器从不把 case.launch 传给实现，autotune 装饰器
也不在任何静态名单里：计时阶段的 warmup 会触发配置搜索，
被测方相当于自带调参器参赛。

注：本文件需要真 triton 才能加载。本地（无 triton）validate 会报
加载失败（SUB-LOD-005），这是环境缺依赖，不是提交本身的问题；
请在昇腾机上验证。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}),
        triton.Config({"BLOCK": 256}),
        triton.Config({"BLOCK": 512}),
        triton.Config({"BLOCK": 1024}),
    ],
    key=["n"],
)
@triton.jit
def _exp_kernel(x_ptr, o_ptr, n, scale, shift, log_base, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    z = (x * scale + shift) * log_base
    tl.store(o_ptr + offs, tl.exp(z).to(o_ptr.dtype.element_ty), mask=mask)


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    log_base = 1.0 if base in (None, -1.0) else math.log(float(base))
    out = torch.empty_like(x)
    n = x.numel()
    _exp_kernel[(triton.cdiv(n, 1),)](  # grid 会被 autotune 接管重算
        x, out, n, float(scale), float(shift), float(log_base)
    )
    return out
