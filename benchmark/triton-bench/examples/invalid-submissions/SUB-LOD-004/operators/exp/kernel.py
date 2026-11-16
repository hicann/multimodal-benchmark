# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-LOD-004：入口函数缺失。

文件能导入，但里面没有清单声明的入口名 exp。

刻意不含任何框架计算调用：否则会先被静态阶段的代算规则拦下，
静态失败会短路加载阶段，这条规则就永远测不到。
"""


def compute(x, base=-1.0, scale=1.0, shift=0.0):
    return x
