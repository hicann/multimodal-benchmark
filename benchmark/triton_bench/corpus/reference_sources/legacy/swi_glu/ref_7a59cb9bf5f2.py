# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SwiGlu 的参考实现（评测方资产）：output = silu(a) * b。

在 dim 上把输入拆成两等份，输出在该维的长度减半——
该 dim 的 size 必须是偶数，由算子规格的派生逻辑保证。
"""

from __future__ import annotations

import torch


def reference(input_value, dim=-1):
    x32 = input_value.to(torch.float32)
    d = dim if dim >= 0 else dim + x32.dim()
    a, b = x32.chunk(2, dim=d)
    return (torch.nn.functional.silu(a) * b).to(input_value.dtype)
