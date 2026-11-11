# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tut import load_tutorial

_t = load_tutorial("best_practice/006-padded_gather_scatter.py")
_padded_gather = _t.padded_gather


def padded_gather_scatter(x, indices, weights, bin_ids, bins, padded_bins, **attrs):
    """适配教程参数顺序，保持注册表签名和原有的 None 占位。

    注册表将 weights 放在第三位；教程在 bin_ids、bins、padded_bins 之后
    才接收 weights。此处仍按教程顺序传参，不改变现有计算路径。
    """
    top_k = int(attrs["top_k"])
    return _padded_gather(x, indices, bin_ids, bins, padded_bins, None, top_k)
