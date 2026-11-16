# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ADV-F1：多输出算子的第二输出造假（Cummin）。

values（第一个输出）完全正确；indices（第二个输出）全部置零。
执行器契约是「元组输出取第一个张量参与比对」（_as_tensor），
于是 indices 从不被比对——
ref.py 的 docstring 也承认这是已知覆盖缺口。

本样例把它钉成可执行证据：一份 indices 完全错误的提交拿 PASS。
"""

from __future__ import annotations

from scripts.example_sources.scalar_dependencies import product as product, torch as torch
from scripts.example_sources.scalar_support import cummin_values as cummin_values
from scripts.example_sources._shared_escapee_10c7261575 import _walk as _walk
from scripts.example_sources._shared_escapee_10c7261575 import _nest as _nest
from scripts.example_sources.adversarial_submissions_scalar_kernels_340c775b28 import _build as _build
from scripts.templates.coverage_kernels import cummin as cummin











def _cummin_implementation(input_value, dim):
    shape, values, indices = cummin_values(input_value, dim)
    output = _build(shape, values, input_value.dtype, input_value.device)
    positions = _build(shape, [0] * len(indices),
                       torch.int64, input_value.device)
    return output, positions


