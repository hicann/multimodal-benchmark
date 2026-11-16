# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-D5：以 id() 为键的缓存（SUB-STR-003 绕过 + 计时虚高）。

规则意图：不得以输入地址为键缓存（输入轮换后复用陈旧结果）。
静态检测是**文本**级的：source 里出现那个取指针地址的方法名（此处
不直接写出，见 anticheat.detect_address_keyed_cache）就报。
把键换成 id(x)——语义完全等价（都是对象身份）——文本扫描即失效。

（注意：本文件的说明文字里也**不能**出现那个方法名：文本检测命中
注释与 docstring 同样会报，见 ADV-E5 的反向证据。）

行为：
  1. 正确性阶段：每个用例的输入都是新造的对象，id 不同 → 真计算 → 结果正确。
  2. 计时阶段（--profile standard）：measure 对**同一批输入**反复调用实现，
     全部命中缓存 → 计时测的是一次 dict 查找，加速比被极度虚高。

附带风险：Python 回收旧输入后 id 可能复用，跨用例返回陈旧结果——
这正是该规则要防的事故，只是换了个写法就防不住了。
"""

from __future__ import annotations


from scripts.example_sources.scalar_dependencies import math as math, torch as torch

from scripts.example_sources._shared_escapee_10c7261575 import _nest as _nest, _walk as _walk
from scripts.example_sources.scalar_support import softmax_values as softmax_values





_CACHE = {}






def _compute(x, dim):
    shape, values = softmax_values(x, dim)
    return torch.tensor(values[0] if not shape else _nest(shape, iter(values)),
                        dtype=x.dtype, device=x.device)


def softmax(x, dim=-1):
    key = (id(x), int(dim))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    out = _compute(x, dim)
    _CACHE[key] = out
    return out
