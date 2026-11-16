# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Transpose 的参考实现（评测方资产）。

perm 是用例必填属性，由 attrs 传入。
输出取 contiguous：参考侧给出**连续**结果，与 triton kernel 写入新缓冲的
行为一致；若返回 permute 的 view，则两侧比较的是同一份数据的不同视图，
布局差异被掩盖，而这个 category 考的正是布局变换。
"""

from __future__ import annotations

import torch


def reference(x, perm):
    """纯布局变换，不涉及算术精度，故不走 fp32 中转。"""
    return torch.permute(x, tuple(int(p) for p in perm)).contiguous()
