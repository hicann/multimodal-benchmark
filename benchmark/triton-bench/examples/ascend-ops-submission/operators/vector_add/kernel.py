# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

import triton
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tut import load_tutorial

_t = load_tutorial("basic/001-vector_add.py")
_kernel = _t.npu_vector_add_kernel


def vector_add(x, y, **attrs):
    z = torch.empty_like(x)
    n = x.numel()
    block_size = 1024
    # grid 必须按 n 计算：教程是 grid=(32,) x BLOCK=512（覆盖 16384），
    # 而此前写死 grid=(1,)，于是只算前 BLOCK 个元素，后面留的是 empty_like
    # 的未初始化值 —— 表现为 shape<=1024 的 6 例 PASS、更大的 14 例全 WRONG。
    # kernel 内部有 `offset < vector_len` 的 mask，多出来的 program 不会越界。
    grid = (triton.cdiv(n, block_size),)
    _kernel[grid](x, y, z, n, block_size)
    return z
