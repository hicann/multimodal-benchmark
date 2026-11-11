# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-STR-003：以输入地址/指针为键的缓存。

输入轮换后，只要地址复用，陈旧结果就会被直接返回——
等价于根本没有计算。必须靠文本检测：AST 上它只是普通下标。

刻意不调用 torch.exp：否则会同时命中 SUB-STR-001（代算），
样例就不只对应一条规则了——那会让提交方不知道该先改哪个。
样例的原则是「一个目录只演示一条违规」。
"""

import torch

_CACHE: dict = {}


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    key = x.data_ptr()
    if key in _CACHE:
        return _CACHE[key]
    # 用乘法占位，避免引入任何框架内置计算调用
    out = x * scale + shift
    _CACHE[key] = out
    return out
