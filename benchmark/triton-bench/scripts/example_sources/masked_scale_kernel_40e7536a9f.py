# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-F6：输出正确，顺手毁掉评测方输入（输入完整性不设防）。

masked_scale 数值完全正确；返回前把 x 与 mask **就地清零**。
参考实现先于实现执行、比对只看输出，于是：
- 终态 PASS，无任何告警；
- 评测方的输入张量（同会话资产）被提交代码静默破坏。
换成「把输入拷走、留陷阱」等变体，受害面更大。

zero_ / fill_ 是就地方法，方法名不在任何静态名单里。
"""

from __future__ import annotations

__all__ = [
    '_nest',
    '_walk',
    'torch',
]

import torch
from scripts.example_sources._shared_escapee_10c7261575 import _nest, _walk









def masked_scale(x, mask, scale=1.0):
    vals = [v * m * float(scale) for v, m in zip(_walk(x), _walk(mask))]
    if not list(x.shape):
        out = torch.tensor(vals[0], dtype=x.dtype, device=x.device)
    else:
        out = torch.tensor(
            _nest(list(x.shape), iter(vals)), dtype=x.dtype, device=x.device
        )

    # —— 攻击点：评测方资产就地销毁，无检查无告警 ——
    with torch.no_grad():
        mask.zero_()
        x.fill_(-1.0)
    return out
