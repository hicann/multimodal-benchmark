# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-LOD-007：参数名与规格不符。

个数对、名字错（scale 写成 factor）。属性以关键字传入，
名字不对会让属性静默丢失——实现按默认值算，看起来"通过"了，
实际根本没测到该测的配置。
"""


def exp(x, base=-1.0, factor=1.0, shift=0.0):
    return x
