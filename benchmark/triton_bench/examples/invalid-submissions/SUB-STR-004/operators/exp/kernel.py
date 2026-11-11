# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-STR-004：非布局类算子做布局搬运。

Exp 是 Elementwise，核心语义是逐元素计算而非搬运。
用 permute/contiguous 绕道属于越权——
布局变换只对 LayoutTransform 类（如 Transpose）放行。
"""


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    return x.permute(0).contiguous()
