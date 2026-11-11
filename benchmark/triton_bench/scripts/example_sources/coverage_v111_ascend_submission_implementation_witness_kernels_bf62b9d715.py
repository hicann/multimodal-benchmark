# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small audited kernels used to exercise implementation coverage obligations.

All arithmetic runs on device. Output allocations and launch dimensions depend
only on public metadata. No evidence declaration in this submission earns credit.
"""

from scripts.example_sources.device_dependencies import tl as tl, torch as torch, triton as triton

from scripts.templates.implementation_witness_kernels import (
    _witness_masked_softmax as _witness_masked_softmax,
    _witness_neg as _witness_neg,
    _witness_neg4 as _witness_neg4,
    masked_softmax as masked_softmax,
)








def neg(x):
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.ndim == 4 and x.is_contiguous():
        d0, d1, d2, d3 = x.shape
        _witness_neg4[(triton.cdiv(d3, 256), d2, d0 * d1)](
            x, out, d0, d1, d2, d3, 256)
    else:
        _witness_neg[(triton.cdiv(x.numel(), 256),)](
            x, out, x.numel(), x.stride(-1), 256)
    return out




