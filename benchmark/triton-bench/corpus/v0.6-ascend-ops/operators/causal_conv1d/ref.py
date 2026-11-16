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

_t = load_tutorial("best_practice/003-fused-cat-slice-conv1d.py")


def make_inputs(seed, dtype, shape, device, attrs):
    torch.manual_seed(seed)
    b, d, sl = shape
    width = 4
    state_len = width - 1  # conv_state 最后一维需 >= width-1
    dev = device if isinstance(device, torch.device) else torch.device(device)
    x = torch.randn((b, d, sl), device=dev).to(torch.bfloat16)
    conv_state = torch.randn((b, d, state_len), device=dev).to(torch.bfloat16)
    weight = torch.randn((d, width), device=dev).to(torch.bfloat16)
    bias = torch.zeros((d,), device=dev).to(torch.bfloat16)
    conv_state_indices = torch.arange(b, dtype=torch.int32, device=dev)
    return [x, conv_state, weight, bias, conv_state_indices]


def reference(x, conv_state, weight, bias, conv_state_indices, **attrs):
    act = None if str(attrs.get("activation", "none")).lower() in ("none", "null") else attrs["activation"]
    # 原地修改 conv_state：先克隆，避免与 entry 共享输入相互污染
    conv_state = conv_state.clone()
    out = _t.causal_conv1d_update_ref(
        x, conv_state, weight, bias, act, cache_seqlens=None,
        conv_state_indices=conv_state_indices)
    return (out, conv_state)
