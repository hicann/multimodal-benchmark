# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""算子分类分级体系（L0，无 NPU 依赖）。

两个正交维度：
    category —— 算子是干什么的（语义族，主分类维度）
    level    —— 写起来多难（L1–L4，算子独立属性）

二者交叉而非包含：同一 category 可跨多个 level（Reduction 出现在 L1/L2/L3），
同一 level 也含多个 category（L2 有 8 个 category）。
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class Category(str, Enum):
    """13 个语义族。取值与 `reference/cann-bench` 保持一致。"""

    ELEMENTWISE = "Elementwise"
    REDUCTION = "Reduction"
    BROADCAST = "Broadcast"
    MASK_PREDICATE = "MaskPredicate"
    NORMALIZATION = "Normalization"
    INDEX_GATHER = "IndexGather"
    SCATTER_UPDATE = "ScatterUpdate"
    CONTRACTION = "Contraction"
    LAYOUT_TRANSFORM = "LayoutTransform"
    SORT_SELECT = "SortSelect"
    NUMERICAL_STABLE = "NumericalStable"
    FUSED_COMPOSITE = "FusedComposite"
    VV_FUSION = "VVFusion"


class Level(str, Enum):
    """难度等级。"""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class Feasibility(str, Enum):
    """在受支持实现语言（triton）下的可行性。

    参考仓库支持 AscendC / PyPTO / TileLang / Triton 多种语言，因此 53 个算子
    都能填。我们只支持 triton，部分 category 无可行实现——必须显式标注，
    保留占位而不能静默丢弃。
    """

    NATIVE = "native"        # triton 擅长
    FEASIBLE = "feasible"    # 可做，但需挑算子
    INFEASIBLE = "infeasible"  # 当前无可行实现，留占位


# 每个 category 的可行性标注。
# 依据：cann-bench 自带的 triton 示例只给了 L1 的 5 个算子
# （Exp / MaskedScale / Mish / Sigmoid / SwiGlu），全是 Elementwise 与 MaskPredicate。
CATEGORY_FEASIBILITY: Final[dict[Category, Feasibility]] = {
    Category.ELEMENTWISE: Feasibility.NATIVE,
    Category.REDUCTION: Feasibility.NATIVE,
    Category.BROADCAST: Feasibility.NATIVE,
    Category.MASK_PREDICATE: Feasibility.NATIVE,
    Category.NORMALIZATION: Feasibility.NATIVE,
    Category.INDEX_GATHER: Feasibility.NATIVE,
    Category.SCATTER_UPDATE: Feasibility.NATIVE,
    Category.CONTRACTION: Feasibility.FEASIBLE,      # 仅 matmul 类；卷积需 im2col
    Category.LAYOUT_TRANSFORM: Feasibility.FEASIBLE,
    Category.SORT_SELECT: Feasibility.FEASIBLE,      # 仅 ArgMax；TopK/Unique 需多 pass
    Category.NUMERICAL_STABLE: Feasibility.FEASIBLE,
    Category.FUSED_COMPOSITE: Feasibility.FEASIBLE,  # 仅 L1 简单融合；L4 的 MHA/MLA 不行
    Category.VV_FUSION: Feasibility.INFEASIBLE,
}

DEFAULT_CASES_PER_CATEGORY: Final[int] = 5


def all_categories() -> list[str]:
    return [c.value for c in Category]


def is_known_category(value: str) -> bool:
    try:
        Category(value)
    except ValueError:
        return False
    return True


def feasibility_of(value: str) -> Feasibility:
    return CATEGORY_FEASIBILITY[Category(value)]


def categories_missing_operators(
    present: set[str],
) -> list[str]:
    """列出没有任何算子的 category。空缺必须显式暴露，不得静默省略。"""
    return [c.value for c in Category if c.value not in present]
