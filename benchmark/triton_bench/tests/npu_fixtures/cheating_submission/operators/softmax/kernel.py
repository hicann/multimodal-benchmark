# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""作弊样例：直接把核心计算交给 torch 内置接口。

用途：验证静态校验能在**执行前**就拒绝它（SUB-STR-001）。
这个文件不属于任何 corpus，只用于测试，不参与评分。
"""

import torch


def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)
