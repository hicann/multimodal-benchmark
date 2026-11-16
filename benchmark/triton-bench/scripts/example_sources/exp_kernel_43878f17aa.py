# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-F4：吞掉硬超时信号（SIGALRM 一次性 → 超时机制失效）。

deadline() 靠 SIGALRM 在超时时刻向主线程注入 TimeoutError_。
它是**一次性**定时器，且异常可以被实现捕获：
`except BaseException` 把超时异常吃掉后，再没有任何机制能打断计算。

本样例：睡 70s（超过 case_timeout_s=60s），信号在 60s 到达时被吞，
再赖 5s 后正常返回**正确**结果——
结果记录的 duration_ms 会超过 60000ms 而终态照常判定。
把睡眠换成无限循环就是永久挂死评测进程（本样例不挂死，方便跑完全轮）。
"""

from __future__ import annotations

__all__ = [
    '_nest',
    '_walk',
    'math',
    'time',
    'torch',
]

import math
import time

import torch
from scripts.example_sources._shared_escapee_10c7261575 import _nest, _walk










def exp(x, base=-1.0, scale=1.0, shift=0.0):
    try:
        time.sleep(70.0)  # 超过单 case 硬超时 60s
    except BaseException:  # SIGALRM 注入的 TimeoutError_ 被吞
        time.sleep(5.0)   # 再赖 5s：证明超时窗口已失效

    lb = 1.0 if base in (None, -1.0) else math.log(float(base))
    vals = [math.exp((v * float(scale) + float(shift)) * lb) for v in _walk(x)]
    if not list(x.shape):
        return torch.tensor(vals[0], dtype=x.dtype, device=x.device)
    return torch.tensor(
        _nest(list(x.shape), iter(vals)), dtype=x.dtype, device=x.device
    )
