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

_t = load_tutorial("best_practice/005-binned_gather_scatter.py")
_binned_gather = _t.binned_gather_npu


def binned_gather_scatter(x, indices, weights, bins, bin_ids, **attrs):
    top_k = int(attrs["top_k"])
    ec = int(attrs.get("expert_capacity") or (x.shape[0] * top_k) // bins.numel())
    # 教程 NPU 路径忽略 weights（与 numpy 参考一致）
    return _binned_gather(x, indices, None, bins, ec, top_k, bin_ids)
