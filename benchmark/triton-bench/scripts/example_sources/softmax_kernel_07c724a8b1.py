# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-F7：忽略属性 dim（语料属性盲区）。

Softmax 的 3 个冻结用例 dim 分别是 0/1/2，而它们的 shape 恰好让
dim == 最后一维：((1024,) dim=0)、((32,128) dim=1)、((4,16,64) dim=2)。
一个**永远按最后一维归一**、完全无视 dim 参数的实现能全部通过。

属性覆盖是语料设计问题而非校验器问题：
其余属性（exp 的 base/scale/shift、gelu 的 approximate、argmax 的
keepdim、transpose 的 perm）都有非默认变体兜底，唯独 softmax 的 dim
全部退化到默认语义。补一个 dim=中间维 的用例即可封死。
"""

from __future__ import annotations


from scripts.example_sources.scalar_dependencies import math as math, torch as torch

from scripts.example_sources._shared_escapee_10c7261575 import _nest as _nest, _walk as _walk
from scripts.example_sources.scalar_support import softmax_values as softmax_values










def softmax(x, dim=-1):
    shape, values = softmax_values(x, -1)
    return torch.tensor(values[0] if not shape else _nest(shape, iter(values)),
                        dtype=x.dtype, device=x.device)
