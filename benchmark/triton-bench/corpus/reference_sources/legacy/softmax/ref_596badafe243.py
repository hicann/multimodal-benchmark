# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Softmax 的参考实现（评测方资产，纯 CPU 可执行）。"""

from __future__ import annotations

import torch


def reference(x, dim=-1):
    # float32 而非 float64：昇腾不支持 fp64（会被静默降级为 float32），
    # 用 float64 会让交叉校验把精度差异误判为实现差异。
    return torch.softmax(x.to(torch.float32), dim=dim).to(x.dtype)
