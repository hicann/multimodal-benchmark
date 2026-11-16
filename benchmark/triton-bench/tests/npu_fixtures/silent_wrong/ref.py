# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""silent_wrong 的第二参考实现（CPU 路径）。

与主 oracle 一致（都是 x + y），因此交叉校验应当通过，
最终判定由「kernel 输出 vs 参考」决定 → WRONG_RESULT。
"""

from __future__ import annotations

import torch


def reference(inputs: list[torch.Tensor]) -> torch.Tensor:
    cpu = [t.detach().to("cpu") for t in inputs]
    return cpu[0] + cpu[1]
