# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D3：别名导入的框架代算（SUB-STR-001 绕过）。

静态扫描的框架前缀表是硬编码白名单：torch / F / nn / functional /
tf / np / numpy / mindspore。`import torch as _t` 之后 _t.maximum(...)
的前缀是 "_t"，不在表里，代算不报警。

扫描器不解析 import 语句，所以无法知道 _t 就是 torch。
"""

from __future__ import annotations

import torch as _t


def maximum(x1, x2):
    return _t.maximum(x1, x2)
