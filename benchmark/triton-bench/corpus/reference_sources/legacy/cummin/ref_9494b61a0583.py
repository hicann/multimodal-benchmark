# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Cummin 的参考实现（评测方资产）。

输出是 (values, indices)：执行器按契约取**第一个张量**参与比对
（见 harness.npu_exec._as_tensor）。即当前只判定 values，
indices 的正确性尚未纳入——这是已知的覆盖缺口，不是实现错误。
"""

from __future__ import annotations

import torch


def reference(input_value, dim):
    """沿 dim 的累积最小值。比较在 fp32 域进行：
    dtype 转换是单调映射，故比较结果与在目标 dtype 上比较一致。
    """
    x32 = input_value.to(torch.float32)
    values, indices = torch.cummin(x32, dim=dim)
    return values.to(input_value.dtype), indices
