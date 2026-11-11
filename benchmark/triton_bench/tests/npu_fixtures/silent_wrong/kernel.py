# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负向用例：能编译、能运行，但结果错误（静默错误）。

用途：验证评测系统真的能检出 WRONG_RESULT。
如果这套系统对它也判 PASS，那么「20/20 全 PASS」就毫无意义。

刻意设计为：
    reference()  算 x + y     ← 主 oracle，正确
    launch()     调 bad_add    ← kernel 实际算 x - y
两者不一致，且都能跑通 → 应当判 WRONG_RESULT，而非 PASS。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def bad_add_kernel(x_ptr, y_ptr, o_ptr, n, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    # 故意算错：应当是 x + y
    tl.store(o_ptr + offs, x - y, mask=mask)


def make_inputs(seed: int, dtype, shape: list[int], device: str) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(shape[0])
    x = torch.randn(n, generator=g).to(dtype)
    y = torch.randn(n, generator=g).to(dtype)
    return [x.to(device), y.to(device)]


def reference(inputs: list[torch.Tensor]) -> torch.Tensor:
    """主 oracle：正确的加法。"""
    return inputs[0] + inputs[1]


def launch(inputs: list[torch.Tensor], out: torch.Tensor, cfg: dict) -> None:
    n = int(inputs[0].numel())
    block = int(cfg.get("BLOCK", 1024))
    bad_add_kernel[(triton.cdiv(n, block),)](
        inputs[0], inputs[1], out, n, block=block,
        num_warps=int(cfg.get("num_warps", 4)),
        num_stages=int(cfg.get("num_stages", 3)),
    )
