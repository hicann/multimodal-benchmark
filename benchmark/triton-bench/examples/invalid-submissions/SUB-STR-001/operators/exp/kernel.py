# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-STR-001：把核心计算交给框架内置接口。

最典型的作弊形态恰恰是**同名代算**——softmax 算子里直接调 torch.softmax。
因此判定不能按「调用名 != 入口名」放过，只看调用是否位于 @triton.jit 内部。
"""

import torch


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    return torch.exp(x * scale + shift)
