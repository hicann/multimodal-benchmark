# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch


def reference(x, conv_state, weight, bias, conv_state_indices, **attrs):
    values = x if x.ndim == 3 else x.unsqueeze(-1)
    history = torch.cat((conv_state.index_select(0, conv_state_indices.long()), values), -1)
    sequence = values.shape[-1]
    width = weight.shape[-1]
    windows = history.double().unfold(-1, width, 1)[:, :, -sequence:]
    output = (windows * weight.double()[None, :, None, :]).sum(-1) + bias.double()[None, :, None]
    output = output.to(x.dtype)
    if str(attrs.get("activation", "silu")).lower() in ("silu", "swish"):
        output = (output.double() * torch.sigmoid(output.double())).to(x.dtype)
    state = conv_state.clone()
    state.index_copy_(0, conv_state_indices.long(), history[:, :, -conv_state.shape[-1]:])
    return output if x.ndim == 3 else output.squeeze(-1), state
