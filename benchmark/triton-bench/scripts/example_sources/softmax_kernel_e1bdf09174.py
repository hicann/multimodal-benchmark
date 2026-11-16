# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-F3：输出 dtype 与规格不符（Softmax）。

规格声明输出 dtype 随输入（fp32/fp16/bf16）；
本实现以 float64 计算（比参考还准）并**返回 float64 张量**。
比对在 float64 域进行，数值全对 → PASS；
输出 dtype 从未被校验——
「以更高精度类型偷渡输出」与「以错误类型降级输出」在终态上不可区分。
"""

from __future__ import annotations


from scripts.example_sources.scalar_dependencies import math as math, torch as torch

from scripts.example_sources._shared_escapee_10c7261575 import _nest as _nest, _walk as _walk
from scripts.example_sources.scalar_support import softmax_values as softmax_values










def softmax(x, dim=-1):
    shape, values = softmax_values(x, dim)
    return torch.tensor(values[0] if not shape else _nest(shape, iter(values)),
                        dtype=torch.float64, device=x.device)
