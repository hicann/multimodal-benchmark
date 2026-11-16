# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""RmsNormLinear 的参考实现（评测方资产）——组合参考。

参考 = 原语参考的顺序组合：先 rms_norm（按行归一化、乘 gamma），
后分组矩阵乘。段间数据流：归一化输出直接喂收缩的主输入——
这正是该融合算子考察的「中间结果不出内核」。

make_inputs 说明：x 与 gamma 由 RmsNorm 段的构造给出（主 shape 即 x），
weight 与 bias 由 GroupedMatmul 段按同一 shape 与 group_list 给出；
两个构造各自独立造主输入，此处只取 weight / bias，x 以 RmsNorm 段为准。
"""

from __future__ import annotations

from triton_eval.refcheck import load_reference_segment


def _seg(entry: str):
    return load_reference_segment(__file__, entry)


_RMS = _seg("rms_norm")
_GMM = _seg("grouped_matmul")


def make_inputs(*, seed, dtype, shape, device, attrs):
    x, gamma = _RMS.make_inputs(seed=seed, dtype=dtype, shape=shape,
                                device=device, attrs=attrs)
    _x2, weight, bias = _GMM.make_inputs(seed=seed + 11, dtype=dtype,
                                         shape=shape, device=device,
                                         attrs=attrs)
    return [x, gamma, weight, bias]


def reference(x, gamma, weight, bias, group_list, epsilon=1e-6):
    normalized = _RMS.reference(x, gamma, epsilon)
    return _GMM.reference(normalized, weight, bias, group_list)
