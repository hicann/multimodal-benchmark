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

_t = load_tutorial("best_practice/001-assign_req_to_token_pool.py")
_kernel = _t.npu_assign_req_to_token_pool


def assign_req_to_token_pool(req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, **attrs):
    pool_len = int(attrs["pool_len"])
    bs_upper = int(attrs["bs_upper"])
    # 原地写回 req_to_token：克隆避免与 reference 共享输入相互污染
    rt = req_to_token.clone()
    _kernel[(bs_upper,)](
        req_pool_indices, rt, start_offset, end_offset, out_cache_loc,
        pool_len, bs_upper)
    return rt
