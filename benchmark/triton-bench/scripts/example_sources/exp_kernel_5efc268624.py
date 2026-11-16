# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-E5：合法实现 + 注释里提到被禁词（SUB-STR-003 的误伤证据）。

实现与 ADV-C1 的 Exp 完全一致（合法、正确、无任何缓存），
唯一的区别是**注释**里出现了那个取指针地址的方法名。

SUB-STR-003 的检测是全文本搜索，命中注释 / docstring 同样报违规——
本样例预期被拒（实际预检已复现：报在第 4 行，即 docstring），
证明「宁可误报」的代价是：提交方在注释里写个词就会被拒，
且报错位置指向文档行，定位体验差。
"""

from __future__ import annotations


from scripts.example_sources.scalar_dependencies import math as math, torch as torch

from scripts.example_sources._shared_escapee_10c7261575 import _nest as _nest, _walk as _walk, exp as exp





# 提示：不要用 data_ptr 做缓存键（本实现没有用它，只是提一句）
# —— 上面这行注释就是被静态文本检测命中的全部「证据」。






