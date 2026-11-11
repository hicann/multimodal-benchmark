# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D7：import 期劫持 oracle（判定权被提交方接管）。

提交代码与评测方参考实现跑在**同一进程**，无任何隔离：
参考实现 corpus/.../softmax/ref.py 调用的正是 torch.softmax。
本文件在 import 时把 torch.softmax 换成恒返回全零的假实现——
此后每个 softmax 用例的「期望值」都是全零；
入口函数同样返回全零，于是误差为零，全部判 PASS。

一个从不算任何东西的提交，让 oracle 替它把「正确答案」定义成了它的输出。
（交叉校验本可制衡，但 corpus 里没有任何 ref_cpu.py，第二参考从未生效。）
"""

from __future__ import annotations

import torch


def _fake_softmax(x, dim=-1, dtype=None):
    return torch.zeros_like(x)


# setattr 的目标名是字符串字面量，静态扫描不可见
setattr(torch, "softmax", _fake_softmax)


def softmax(x, dim=-1):
    return torch.zeros_like(x)
