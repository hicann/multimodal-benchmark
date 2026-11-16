# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AddRmsNorm 的参考实现（评测方资产）。

来源池 tasks/level3/add_rms_norm_dynamic_quant 的主干（add + rms_norm 段）：
    y = rms_norm(x1 + x2) * gamma
归一化段复用 RmsNorm 原语参考（单一代码事实来源）；加法段是逐元素主路径。
池公式为 quantize(rmsnorm(x1+x2)*gamma)——量化段（y/xOut/scaleOut 中的
量化两路输出）待量化批次另行收录，本参考只产出归一化主输出。

确定性要点：x1 与 x2 用不同随机种子构造——两个同分布但不同的输入
（双输入比较类用例的合法性规则：相同输入会让比较失去区分度）。
"""

from __future__ import annotations

from triton_eval.refcheck import load_reference_segment

import torch


def _seg(entry: str):
    return load_reference_segment(__file__, entry)


_RMS = _seg("rms_norm")


def _rand(seed: int, n: int, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, generator=g).to(dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    n = 1
    for d in shape:
        n *= d
    x1 = _rand(seed, n, dtype).reshape(shape).to(device)
    x2 = _rand(seed + 2, n, dtype).reshape(shape).to(device)
    gamma = _rand(seed + 1, shape[-1], dtype).to(device)
    return [x1, x2, gamma]


def reference(x1, x2, gamma, epsilon=1e-6):
    summed = x1.to(torch.float32) + x2.to(torch.float32)
    y = _RMS.reference(summed, gamma, epsilon)
    return y.to(x1.dtype)
