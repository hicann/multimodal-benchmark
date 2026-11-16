# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import os

from triton_eval.inputs import expert_routing_inputs

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tut import load_tutorial

_t = load_tutorial("best_practice/004-gather_scatter.py")


def make_inputs(seed, dtype, shape, device, attrs):
    routing = expert_routing_inputs(seed, shape, device, attrs, legacy_shape=True)
    indices = routing.indices
    weights = torch.rand(routing.experts.shape, device=routing.values.device).to(torch.float16)
    return [routing.values, indices, weights]


def reference(x, indices, weights, **attrs):
    # 教程的 gather_numpy 定义在 test_fn 内部（不是模块级），取不到；
    # 而模块级的 gather() 内部跑的正是被测的 triton kernel，拿来当参考等于自证。
    # 这里按 numpy 参考的语义用纯 torch 复写：out[i] = x[indices[i] // top_k]。
    top_k = int(attrs["top_k"])
    return x[(indices // top_k).long()]
