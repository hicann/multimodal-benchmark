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
    _witness_neg3 as _witness_neg3,
    _witness_neg4 as _witness_neg4,
    masked_softmax as masked_softmax,
    neg as neg,
)














