# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch.nn.functional as functional


def make_inputs(seed, dtype, shape, device, attrs):
    torch.manual_seed(seed)
    batch, channels = shape[:2]
    sequence = shape[2] if len(shape) == 3 else 1
    x = torch.randn((batch, channels, sequence), device=device).to(torch.bfloat16)
    state = torch.randn((batch + 1, channels, 3), device=device).to(torch.bfloat16)
    weight = torch.randn((channels, 4), device=device).to(torch.bfloat16)
    bias = torch.randn((channels,), device=device).to(torch.bfloat16)
    indices = torch.arange(batch - 1, -1, -1, device=device, dtype=torch.int32)
    return [x if len(shape) == 3 else x.squeeze(-1), state, weight, bias, indices]


def reference(x, conv_state, weight, bias, conv_state_indices, **attrs):
    sequence = x.shape[2] if x.ndim == 3 else 1
    values = x if x.ndim == 3 else x.unsqueeze(-1)
    history = torch.cat((conv_state[conv_state_indices.long()], values), dim=-1)
    state = conv_state.clone()
    state[conv_state_indices.long()] = history[:, :, -conv_state.shape[-1]:]
    output = functional.conv1d(history.float(), weight.float().unsqueeze(1),
                               bias.float(), groups=x.shape[1])[:, :, -sequence:].to(x.dtype)
    activation = str(attrs.get("activation", "silu")).lower()
    if activation in ("silu", "swish"):
        output = functional.silu(output.float()).to(x.dtype)
    elif activation not in ("none", "null"):
        raise ValueError("activation must be silu, swish or none")
    return output if x.ndim == 3 else output.squeeze(-1), state
