# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tut import load_tutorial

_t = load_tutorial("basic/004-discrete_memory_access.py")


def make_inputs(seed, dtype, shape, device, attrs):
    torch.manual_seed(seed)
    n = shape[0]
    m = int(attrs["m"])
    dev = device if isinstance(device, torch.device) else torch.device(device)
    # 同 vector_add：调用方传的是 torch.dtype 对象，兼容字符串写法
    dt = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    x = torch.randn(m, device=dev).to(dt)
    # 索引范围与官方教程一致 [0, M)，保证 Ascend pick kernel 不越界
    indices = torch.randint(0, m, (n,), device=dev).to(torch.int32)
    return [x, indices]


def reference(x, indices, **attrs):
    m = int(attrs["m"])
    y = torch.zeros(indices.shape, dtype=x.dtype, device=x.device)
    valid = indices < m
    y[valid] = x[indices[valid].long()]
    return y
