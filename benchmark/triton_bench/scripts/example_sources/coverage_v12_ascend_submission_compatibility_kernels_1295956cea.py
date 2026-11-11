# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Signed-int32 division and scalar-metadata grouped matmul candidates."""
__all__ = [
    '_floor_div',
    '_floor_divide_int32',
    '_grouped',
    '_linear_stride',
    'floor_div',
    'grouped_matmul',
    'tl',
    'torch',
    'triton',
]

import torch
import triton
import triton.language as tl

from scripts.templates.compatibility_kernels import (
    _floor_div,
    _floor_divide_int32,
    _grouped,
    _linear_stride,
    floor_div,
    grouped_matmul,
)















