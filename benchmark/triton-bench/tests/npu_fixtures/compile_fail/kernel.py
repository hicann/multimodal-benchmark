# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负向用例：应当编译失败。

用途：验证系统能把「编译不过」正确判为 COMPILE_FAIL，
而不是笼统地归为 RUNTIME_FAIL 或直接崩掉。

刻意让 kernel 使用 1D 张量调用 tl.dot（tl.dot 要求 2D tile），
这在 Triton 上是确定性的语义错误。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def bad_dot_kernel(x_ptr, o_ptr, n, block: tl.constexpr):
    offs = tl.arange(0, block)
    x = tl.load(x_ptr + offs, mask=offs < n)
    # 语义错误：tl.dot 需要二维 tile，这里传了一维
    y = tl.dot(x, x)
    tl.store(o_ptr + offs, y, mask=offs < n)


def make_inputs(seed: int, dtype, shape: list[int], device: str) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(shape[0])
    return [torch.randn(n, generator=g).to(dtype).to(device)]


def reference(inputs: list[torch.Tensor]) -> torch.Tensor:
    return inputs[0] * 2.0


def launch(inputs: list[torch.Tensor], out: torch.Tensor, cfg: dict) -> None:
    n = int(inputs[0].numel())
    block = max(1, triton.next_power_of_2(n))
    bad_dot_kernel[(1,)](inputs[0], out, n, block=block)
