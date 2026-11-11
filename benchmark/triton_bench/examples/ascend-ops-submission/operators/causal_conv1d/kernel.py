# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _conv(x, state, weight, bias, indices, out, new_state, configuration: tl.constexpr):
    d, sequence, state_local, width, silu, block = configuration
    batch = tl.program_id(0)
    channels = tl.program_id(1) * block + tl.arange(0, block)
    valid = channels < d
    state_row = tl.load(indices + batch)
    bias_value = tl.load(bias + channels, valid, other=0).to(tl.float32)
    for step in tl.static_range(sequence):
        result = bias_value
        for tap in tl.static_range(width):
            position = state_local - width + 1 + step + tap
            if position < state_local:
                value = tl.load(state + (state_row * d + channels) * state_local + position, valid, other=0).to(
                    tl.float32
                )
            else:
                value = tl.load(x + (batch * d + channels) * sequence + position - state_local, valid, other=0).to(
                    tl.float32
                )
            weight_value = tl.load(weight + channels * width + tap, valid, other=0).to(tl.float32)
            result = result + value * weight_value
        result = result.to(out.dtype.element_ty).to(tl.float32)
        if silu:
            result = result * tl.sigmoid(result)
        tl.store(out + (batch * d + channels) * sequence + step, result, valid)
    for slot in tl.static_range(state_local):
        position = sequence + slot
        if position < state_local:
            state_value = tl.load(state + (state_row * d + channels) * state_local + position, valid, other=0)
        else:
            state_value = tl.load(x + (batch * d + channels) * sequence + position - state_local, valid, other=0)
        tl.store(new_state + (state_row * d + channels) * state_local + slot, state_value, valid)


def causal_conv1d_update(x, conv_state, weight, bias, conv_state_indices, **attrs):
    activation = str(attrs.get("activation", "silu")).lower()
    if activation not in ("silu", "swish", "none", "null"):
        raise ValueError("activation must be silu, swish or none")
    batch, channels = x.shape[:2]
    sequence = x.shape[2] if x.ndim == 3 else 1
    output = torch.empty_like(x)
    new_state = conv_state.clone()
    _conv[batch, triton.cdiv(channels, 128)](
        x,
        conv_state,
        weight,
        bias,
        conv_state_indices,
        output,
        new_state,
        (channels, sequence, conv_state.shape[-1], weight.shape[-1], activation in ("silu", "swish"), 128),
        enable_fp_fusion=False,
    )
    return output, new_state
