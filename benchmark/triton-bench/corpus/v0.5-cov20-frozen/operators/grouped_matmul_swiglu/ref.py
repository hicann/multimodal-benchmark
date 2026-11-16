# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GroupedMatmulSwiGlu 的参考实现（评测方资产）——组合参考。

参考 = 原语参考的顺序组合：先分组矩阵乘，后 swiglu 门控
（输出维减半）。主干对应来源池任务
`tasks/level4/grouped_matmul_swiglu_quant` 的矩阵乘 + swiglu 段
（量化段待量化批次另行收录）。

段间约束：swiglu 沿最后一维对半拆，故矩阵乘的输出维 N 必须为偶数
（GroupedMatmul 参考的 N 固定为 64，天然满足）。
"""

from __future__ import annotations

from triton_eval.refcheck import load_reference_segment


def _seg(entry: str):
    return load_reference_segment(__file__, entry)


_GMM = _seg("grouped_matmul")
_SWI_GLU = _seg("swi_glu")


def make_inputs(*, seed, dtype, shape, device, attrs):
    return _GMM.make_inputs(seed=seed, dtype=dtype, shape=shape, device=device,
                            attrs=attrs)


def reference(x, weight, bias, group_list):
    y = _GMM.reference(x, weight, bias, group_list)
    return _SWI_GLU.reference(y, -1)
