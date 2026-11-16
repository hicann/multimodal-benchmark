# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D6：import 期 setattr 篡改同步接口（SUB-STR-005 绕过）。

规则意图：不得替换/伪装/删除评测的计时与同步。
直接赋值 torch.npu.synchronize = ... 会被静态扫描的「赋值形态检查」抓到；
改用 setattr 后目标名只是字符串字面量，AST 上完全不可见。

加载阶段会真正 import 本文件，篡改在**任何用例执行之前**生效：
torch.npu.synchronize 从此是空操作，整轮计时口径被破坏且无任何告警。

本地（无 torch_npu）退化到篡改 torch.cuda.synchronize，
仅为验证「加载不报错」；真机上命中 torch.npu。
exp 本身是一个正常的慢速实现——攻击点只在篡改。
"""

from __future__ import annotations


from scripts.example_sources.scalar_dependencies import math as math, torch as torch

from scripts.example_sources._shared_escapee_10c7261575 import _nest as _nest, _walk as _walk, exp as exp






def _noop(*args, **kwargs):
    return None


try:
    import torch_npu  # noqa: F401

    _TARGET = torch.npu
except Exception:
    _TARGET = torch.cuda  # 本地无昇腾时的等价目标

setattr(_TARGET, "synchronize", _noop)






