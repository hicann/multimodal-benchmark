# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-F5：运行期错误伪装成编译失败（终态归因错置）。

_is_compile_error 靠在异常名/文本里找 "compil" / "syntax" 字样。
本实现的**运行期**错误消息里故意带上 "compile"——
终态会被判 COMPILE_FAIL（编译器指标）而非 RUNTIME_FAIL（提交方问题）。

COMPILE_FAIL 是被测编译器的质量信号：提交方可以把自己的任何运行期
缺陷伪装成编译器失败，把责任转嫁给被测对象；反之亦然。
"""

from __future__ import annotations


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    raise RuntimeError(
        "triton compile failed: unsupported dtype in lowering stage"
    )
