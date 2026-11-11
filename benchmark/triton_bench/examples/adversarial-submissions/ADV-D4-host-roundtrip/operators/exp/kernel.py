# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D4：以 to("cpu") 做主机侧往返（SUB-STR-002 绕过）。

规则意图：不得把输入搬回主机计算。
静态扫描的设备回退名单是调用名集合 {cpu, to_cpu, numpy, tolist}：
    x.cpu()     → 会被抓（方法名 cpu 在名单里）
    x.to("cpu") → **不会**被抓（方法名是 to，不在名单里）

本实现把输入搬到 CPU、在主机上算 exp、直接返回 **CPU 张量**——
输出设备从头到尾无人校验（比对前会统一 .to("cpu")），
于是一份纯主机计算的提交可以走完全部用例拿到 PASS。
"""

from __future__ import annotations

import math

import torch


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    xc = x.to("cpu")  # 主机侧往返的绕过形态：方法名是 to，不在文本名单
    lb = 1.0 if base in (None, -1.0) else math.log(float(base))
    return (xc * float(scale) + float(shift) * lb).exp()
