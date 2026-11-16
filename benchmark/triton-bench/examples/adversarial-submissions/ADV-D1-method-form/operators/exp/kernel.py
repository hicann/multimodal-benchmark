# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D1：方法调用形态的框架代算（SUB-STR-001 绕过）。

规则意图：Exp 的核心计算不得交给框架内置接口。
静态扫描只拦截「框架前缀」的调用：torch.exp(x) / F.exp(x) 会被抓，
但张量**方法形态** x.exp() 的调用前缀是变量名，不在框架前缀表里——
整个核心计算由一行 .exp() 代劳，静态阶段零告警。

对照：examples/invalid-submissions/SUB-STR-001（torch.exp 直调）会被拒，
本样例与它语义完全等价，只是换了个写法。
"""

from __future__ import annotations

import math

import torch


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    lb = 1.0 if base in (None, -1.0) else math.log(float(base))
    # 乘加是运算符重载（允许），最后一步 .exp() 把核心计算整个交给 torch
    return (x * float(scale) + float(shift) * lb).exp()
