# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import os

from triton_eval.reference_execution import assign_token_pool_reference
from triton_eval.inputs import token_pool_inputs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tut import load_tutorial

_t = load_tutorial("best_practice/001-assign_req_to_token_pool.py")


def make_inputs(seed, dtype, shape, device, attrs):
    return token_pool_inputs(seed, device, attrs, unique_requests=False)


def reference(req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, **attrs):
    batch = int(attrs["bs_upper"])
    return assign_token_pool_reference(req_pool_indices, req_to_token, start_offset,
                                       end_offset, out_cache_loc, batch)
