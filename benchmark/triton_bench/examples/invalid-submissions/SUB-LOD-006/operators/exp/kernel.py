# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-LOD-006：参数个数与规格不符。

规格声明 exp(x, base, scale, shift) 四个参数，这里只有两个。
调用约定是「位置传输入、关键字传属性」，少参数会让属性被当成输入张量传入。

刻意不 import triton、也不做实际计算：
这条规则只关乎签名，这样在任何装了 torch 的机器上都能验证。
"""


def exp(x, base=-1.0):
    return x
