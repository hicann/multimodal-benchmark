# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exp 的参考实现（评测方资产）。

参考实现必须独立于任何提交——否则就成了自己验证自己。
它可在纯 CPU 上执行，用于交叉校验。

计算精度统一为 float32（见 spec/tolerance.yaml 的 reference_dtype）：
    昇腾不支持 float64，且若直接以目标 dtype（如 fp16）计算，
    参考与 kernel 的差距会混入「参考自身精度不足」的部分，
    那不是被测方的问题。故一律 fp32 计算后再转回目标 dtype。
"""

from __future__ import annotations

import math

import torch


def reference(x, base=-1.0, scale=1.0, shift=0.0):
    """exp(x)：base 为 -1 时按自然指数，否则按 base 为底。"""
    x32 = x.to(torch.float32)
    if base is None or math.isclose(base, -1.0, rel_tol=0.0, abs_tol=0.0):
        y = torch.exp(x32 * scale + shift)
    else:
        y = torch.pow(float(base), x32 * scale + shift)
    return y.to(x.dtype)
