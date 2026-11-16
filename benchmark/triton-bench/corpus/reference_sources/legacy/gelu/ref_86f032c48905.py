# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Gelu 的参考实现（评测方资产）。

精度约定（全 corpus 统一）：
    fp32 计算 → 转回目标 dtype。昇腾不支持 fp64，若直接以 fp16/bf16 计算，
    参考与实现的差距会混入「参考自身精度不足」的部分，那不是被测方的问题。
"""

from __future__ import annotations

import torch


def reference(x, approximate="none"):
    """gelu：approximate 仅 "none" / "tanh" 两种合法取值。"""
    x32 = x.to(torch.float32)
    return torch.nn.functional.gelu(x32, approximate=approximate).to(x.dtype)
