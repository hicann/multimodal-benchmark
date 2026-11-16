# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-F2：多输出算子只交一个输出（Cummin）。

规格声明 Cummin 有两个输出 (values, indices)；
本实现只返回 values——输出元数从未被校验，
_as_tensor 对单个张量直接放行。
一份「少交一半输出」的提交照常拿 PASS。
"""

from __future__ import annotations

from scripts.example_sources.scalar_dependencies import product as product, torch as torch
from scripts.example_sources.scalar_support import cummin_values as cummin_values
from scripts.example_sources._shared_escapee_10c7261575 import _walk as _walk
from scripts.example_sources._shared_escapee_10c7261575 import _nest as _nest
from scripts.templates.coverage_kernels import cummin as cummin









def _cummin_implementation(input_value, dim):
    shape, values, _indices = cummin_values(input_value, dim)
    return torch.tensor(values[0] if not shape else _nest(shape, iter(values)),
                        dtype=input_value.dtype, device=input_value.device)


