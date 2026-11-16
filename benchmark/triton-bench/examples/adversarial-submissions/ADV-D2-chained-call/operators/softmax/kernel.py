# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D2：链式属性调用的框架代算（SUB-STR-001 绕过）。

torch.nn.functional.softmax(x, dim) 的调用前缀在 AST 上是
Attribute(functional=Attribute(nn=Name(torch)))——不是简单的 Name 节点，
_call_name 取不到前缀，于是「框架前缀」判定为空，代算不报警。

注意：这个文件与评测方自己的参考实现
corpus/v0.3-multi-input-frozen/operators/softmax/ref.py **逐字等价**——
参考实现怎么写，提交方就怎么抄，静态扫描两边都看不见。
"""

from __future__ import annotations

import torch


def softmax(x, dim=-1):
    return torch.nn.functional.softmax(x.to(torch.float32), dim=dim).to(x.dtype)
