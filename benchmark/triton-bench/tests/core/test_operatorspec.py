# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""算子规格测试（L0，无 NPU 依赖）。"""

from __future__ import annotations

import pytest

from triton_eval.operator_registry import ALL_OPERATORS, by_category, by_name
from triton_eval.operatorspec import AttrSpec, IOSpec, OperatorSpec, ShapeSupport
from triton_eval.taxonomy import Category, Level


def _op(**over) -> OperatorSpec:
    base = dict(
        name="Dummy", category=Category.ELEMENTWISE, level=Level.L1,
        signature="dummy(Tensor x, float s=1.0) -> Tensor y",
        inputs=[IOSpec(name="x", dtype=["float32", "float16"])],
        outputs=[IOSpec(name="y", dtype=["float32", "float16"])],
        attrs=[AttrSpec(name="s", type="float", default=1.0)],
        entry="dummy", params=["x", "s"],
    )
    base.update(over)
    return OperatorSpec(**base)  # type: ignore[arg-type]


# ---------- 任务 2.1 ----------


def test_valid_spec_passes() -> None:
    op = _op()
    assert op.name == "Dummy"
    assert op.entry == "dummy"


def test_empty_dtype_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IOSpec(name="x", dtype=[])


def test_empty_params_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _op(params=[])


# ---------- 任务 2.2 ----------


def test_out_of_range_dtype_rejected() -> None:
    op = _op()
    with pytest.raises(ValueError, match="不支持 dtype"):
        op.validate_case_params("int64", [8], {})


def test_declared_dtype_accepted() -> None:
    op = _op()
    op.validate_case_params("float16", [8], {})  # 不抛异常


# ---------- 任务 2.3 ----------


def test_rank_below_min_rejected() -> None:
    op = _op(shape_support=ShapeSupport(min_rank=2, max_rank=4))
    with pytest.raises(ValueError, match="维度数"):
        op.validate_case_params("float32", [8], {})


def test_rank_above_max_rejected() -> None:
    op = _op(shape_support=ShapeSupport(min_rank=1, max_rank=2))
    with pytest.raises(ValueError, match="维度数"):
        op.validate_case_params("float32", [2, 4, 8], {})


# ---------- 任务 2.4 ----------


def test_required_attr_missing_rejected() -> None:
    op = _op(attrs=[AttrSpec(name="dim", type="int64", required=True)])
    with pytest.raises(ValueError, match="缺少必填属性"):
        op.validate_case_params("float32", [8], {})


def test_undeclared_attr_rejected() -> None:
    op = _op()
    with pytest.raises(ValueError, match="未声明属性"):
        op.validate_case_params("float32", [8], {"nope": 1})


def test_attr_type_mismatch_rejected() -> None:
    op = _op(attrs=[AttrSpec(name="n", type="int64", default=1)])
    with pytest.raises(ValueError, match="类型不符"):
        op.validate_case_params("float32", [8], {"n": "not-an-int"})


def test_unknown_dtype_method_raises() -> None:
    op = _op()
    with pytest.raises(KeyError):
        op.dtype_of("nonexistent")


def test_resolve_attrs_fills_defaults() -> None:
    op = _op(attrs=[AttrSpec(name="s", type="float", default=2.5)])
    assert op.resolve_attrs({}) == {"s": 2.5}
    assert op.resolve_attrs({"s": 9.0}) == {"s": 9.0}


def test_list_attr_type_accepted() -> None:
    """int[] 类型必须被当作列表校验，不能按标量整型误报。

    曾经的 bug：`int[]`.startswith("int") 命中整型分支，
    导致 group_list=[1,1] 被误判为类型不符。
    """
    op = _op(attrs=[AttrSpec(name="perm", type="int[]", required=True)])
    op.validate_case_params("float32", [4, 8], {"perm": [1, 0]})


def test_list_attr_rejects_scalar() -> None:
    op = _op(attrs=[AttrSpec(name="perm", type="int[]", required=True)])
    with pytest.raises(ValueError, match="期望列表"):
        op.validate_case_params("float32", [4, 8], {"perm": 1})


# ---------- 任务 3.1 / 3.2：算子选型 ----------


def test_registered_operator_count() -> None:
    from triton_eval.official_catalog import CANDIDATES

    historical_names = {"Exp", "Gelu", "Maximum", "Softmax", "RmsNorm", "Gather", "Transpose", "ArgMax", "SwiGlu"}
    expected = 36 + len(CANDIDATES) - len(historical_names)
    assert len(ALL_OPERATORS) == expected
    assert len({operator.name for operator in ALL_OPERATORS}) == expected


def test_fusion_operators_have_composition() -> None:
    """融合算子与组合表一一对应：每个融合算子都声明了段落，反之亦然。"""
    from triton_eval.operator_registry import COMPOSITIONS, fusion_operators

    assert len(fusion_operators()) == 11
    assert set(fusion_operators()) == set(COMPOSITIONS)
    # 融合段落数：两段融合是本批的口径（多段融合待后续批次）
    assert all(len(COMPOSITIONS[op]) == 2 for op in fusion_operators())


def test_every_category_except_vvfusion_has_operator() -> None:
    covered = {op.category.value for op in ALL_OPERATORS}
    assert Category.VV_FUSION.value not in covered, "VVFusion 无可行算子"
    for cat in Category:
        if cat is Category.VV_FUSION:
            continue
        assert cat.value in covered, f"{cat.value} 缺少算子"


def test_contraction_selected_operators() -> None:
    """收缩类包含分组矩阵乘及新增的分组注意力。"""
    names = [op.name for op in by_category("Contraction")]
    assert names == ["GroupedMatmul", "DecodeGroupedAttention", "Matmul"]
    for banned in ("Conv2D", "DepthwiseConv2D", "Conv3DBackpropFilter", "Dilation2D"):
        assert banned not in names


def test_sortselect_official_candidates_registered() -> None:
    """Official selection candidates retain their distinct output contracts."""
    names = [op.name for op in by_category("SortSelect")]
    assert names == ["ArgMax", "ArgMin", "SortValues", "TopKValues"]
    for banned in ("TopK", "Unique", "NMS"):
        assert banned not in names


def test_fused_composite_only_simple_selected() -> None:
    """FusedComposite 只收录简单融合：L4 复杂融合（MHA/MLA/GRU/LSTM）不得进
    注册表；本批融合算子全部 ≤ L3（分级边界，见组合维度扩展方案 §2.2）。
    """
    ops = by_category("FusedComposite")
    names = [op.name for op in ops]
    for banned in ("Mha", "Mla", "Gru", "Lstm", "Gqa"):
        assert banned not in names, f"{banned} 属 L4 复杂融合，本批不收录"
    for op in ops:
        assert op.level.value in ("L1", "L2", "L3"), f"{op.name} 超出简单融合分级"


def test_operator_provenance_recorded() -> None:
    """每个算子必须能追溯：直收算子记来源池目录；组合算子记组合表。

    可追溯是硬要求，但形式有两种——来源池的 proto.yaml，或组合表里
    逐段引用的已注册原语（原语自身又可追溯到池）。两种都没有的才是洞。
    """
    from triton_eval.operator_registry import COMPOSITIONS, by_name as _by_name

    for op in ALL_OPERATORS:
        if op.name in COMPOSITIONS:
            stages = COMPOSITIONS[op.name]
            assert len(stages) >= 2, f"{op.name} 组合表段落不足两段"
            if op.source_dir:
                assert op.source_name, f"{op.name} 有来源目录但缺来源算子名"
                continue
            for stage in stages:
                assert _by_name(stage.name), \
                    f"{op.name} 的段落 {stage.name} 不是已注册原语——组合出处断链"
            continue
        assert op.source_dir, f"{op.name} 缺少来源目录"
        assert op.source_name, f"{op.name} 缺少来源算子名"


def test_by_name_lookup() -> None:
    assert by_name("Exp") is not None
    assert by_name("Nonexistent") is None


def test_entry_matches_params() -> None:
    """入口函数名与参数列表是提交校验的依据，必须齐备。"""
    for op in ALL_OPERATORS:
        assert op.entry, f"{op.name} 缺 entry"
        assert op.params, f"{op.name} 缺 params"
