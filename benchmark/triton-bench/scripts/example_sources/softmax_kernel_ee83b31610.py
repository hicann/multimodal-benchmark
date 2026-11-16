# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D9：返回张量子类，验证按 isinstance 接纳子类的输出策略。

数值完全正确（手写 softmax），但返回类型是 torch.Tensor 的子类。
当前策略不再仅因子类身份判 SUBMISSION_INVALID；
设备、布局、精度、委托计算与其他违规检查继续独立生效。
"""

from __future__ import annotations


from scripts.example_sources.scalar_dependencies import math as math, torch as torch

from scripts.example_sources._shared_escapee_10c7261575 import _nest as _nest, _walk as _walk
from scripts.example_sources.scalar_support import softmax_values as softmax_values






class _Sub(torch.Tensor):
    """真实 Tensor 子类：按当前实例检查策略允许进入后续验证。"""






def softmax(x, dim=-1):
    shape, values = softmax_values(x, dim)
    output = torch.tensor(values[0] if not shape else _nest(shape, iter(values)),
                          dtype=x.dtype, device=x.device)
    return output.as_subclass(_Sub)
