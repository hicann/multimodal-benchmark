# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""LinearGelu 的 triton 实现（合法提交样例）——当前整算子显性不支持。

y = gelu(grouped_matmul(x))：本算子的 gelu 段取**默认精确语义**
（approximate="none"，即 erf 路径）——签名里没有 approximate 参数，
全部用例都考 erf 精确 gelu。

样例实现不了 erf：它需要 libdevice 支持，未在本环境验证过
（口径同 gelu/kernel.py 的既有决策：宁可明确报 NotImplementedError，
也不用未经验证的近似去凑一个"通过"——tanh 近似与 erf 精确路径的
偏差 ~3e-4，超出 f32 容差，硬凑会是更糟的静默错误）。

后果（如实披露）：该算子的锚点（性能基准）无法由示范提交冻结，
待支持 erf 的真实提交出现后按算子锚点机制自然冻结。
"""

from __future__ import annotations


def linear_gelu(x, weight, bias, group_list):
    raise NotImplementedError(
        "样例内核不支持 LinearGelu：其 gelu 段为 erf 精确语义，"
        "erf 依赖 libdevice、未在本环境验证（口径同 gelu/kernel.py）。"
        "请自行实现 erf 路径后再提交本算子")
