# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import os

from triton_eval.inputs import permutation_inputs

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tut import load_tutorial

_t = load_tutorial("op_extension/002-extract_slice.py")


def make_inputs(seed, dtype, shape, device, attrs):
    return permutation_inputs(seed, shape, device)


def reference(x, indices, **attrs):
    # extract_slice：out[indices[i]] = x[i]（insert_slice 的逆重排）
    out = torch.empty_like(x)
    out[indices.long()] = x
    return out
