# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""覆盖引导生成的测试（L0 标签器/选择器 + torch 可选的输入维度）。

守护什么：
    标签是「算子规格 + 用例五维」的**纯函数**——覆盖率的分母分子、
    覆盖引导的靶子，全都建立在这条边界上。标签器回归 = 覆盖率失真。
"""

from __future__ import annotations

from math import prod
from pathlib import Path

import pytest

from triton_eval.coverage import (
    coverage_stats,
    guided_selection,
    label_case,
    size_class,
)
from triton_eval.models import CaseSpec
from triton_eval.operator_registry import ALL_OPERATORS, by_name
from triton_eval.specload import load_feature_atoms

FA = load_feature_atoms()
UNIVERSE = [a["id"] for a in FA["atoms"]]
RESERVED = {a["id"] for a in FA.get("reserved", [])}


def _case(**kw) -> CaseSpec:
    base = dict(
        code="t-00", operator="Exp", source="vendor_selftest",
        dtype="float32", shape=[1024],
        attrs={}, seed=1,
    )
    base.update(kw)
    return CaseSpec(**base)


# ---------------- 标签器 ----------------


def test_label_is_pure_and_deterministic() -> None:
    c = _case(shape=[4096, 64], dtype="bfloat16")
    a = label_case(c, by_name("Exp"))
    b = label_case(_case(shape=[4096, 64], dtype="bfloat16"), by_name("Exp"))
    assert a == b, "同一用例两次打标必须一致（纯函数）"
    assert "FA-DT-BF16" in a
    assert "FA-SHAPE-SKINNY" in a
    assert "FA-LOAD-MEDIUM" in a


def test_label_covers_axis_positions() -> None:
    """中间维是 F7 类盲区的靶子：softmax 只认末维的实现必须被抓。"""
    spec = by_name("Softmax")
    last = label_case(_case(operator="Softmax", shape=[4, 16, 64],
                            attrs={"dim": 2}), spec)
    first = label_case(_case(operator="Softmax", shape=[4, 16, 64],
                             attrs={"dim": 0}), spec)
    inner = label_case(_case(operator="Softmax", shape=[4, 16, 64],
                             attrs={"dim": 1}), spec)
    assert "FA-AXIS-LAST" in last and "FA-AXIS-FIRST" not in last
    assert "FA-AXIS-FIRST" in first
    assert "FA-AXIS-INNER" in inner, "中间维标签缺失——F7 盲区没人盯"


def test_label_value_and_layout_dimensions() -> None:
    spec = by_name("Exp")
    zeros = label_case(_case(value_range={"kind": "zeros"}), spec)
    mixed = label_case(_case(value_range={"kind": "mixed"}), spec)
    extreme = label_case(_case(value_range={"kind": "extreme"}), spec)
    strided = label_case(_case(shape=[32, 128], contiguous=False), spec)
    assert "FA-NUM-ZEROS" in zeros and "FA-NUM-MIXED" in mixed
    assert "FA-NUM-EXTREME" in extreme
    assert "FA-LOAD-STRIDED" in strided and "FA-LOAD-CONTIG" not in strided


def test_label_alignment() -> None:
    spec = by_name("Exp")
    assert "FA-LOAD-ALIGNED" in label_case(_case(shape=[1024]), spec)
    assert "FA-LOAD-MASKED-EDGE" in label_case(_case(shape=[1000]), spec)


def test_labels_stay_within_universe() -> None:
    """标签器与 yaml 不得漂移：发射的原子必须在 active 或 reserved 里。"""
    from triton_eval.casegen import derive_pool

    for op in ALL_OPERATORS:
        for d in derive_pool(op, base_seed=99):
            c = _case(code=d.code, operator=d.operator, dtype=d.dtype,
                      shape=d.shape, attrs=d.attrs)
            for atom in label_case(c, by_name(d.operator)):
                assert atom in UNIVERSE or atom in RESERVED, \
                    f"标签器发射了清单外的原子：{atom}"


# ---------------- 覆盖统计 ----------------


def test_coverage_stats_denominator_excludes_reserved() -> None:
    class C:
        feature_atoms = ["FA-DT-FP32"]

    stats = coverage_stats([C()], UNIVERSE)
    assert stats["rate"] == 1 / len(UNIVERSE)
    assert "FA-DT-FP32" in stats["covered"]
    assert "FA-ATOMIC-ADD" not in UNIVERSE, "reserved 原子不得混进分母"


# ---------------- 引导选择 ----------------


def _pool_for_selection():
    from triton_eval.casegen import derive_pool

    pool = []
    for op in ALL_OPERATORS[:4]:  # 4 个算子足以验证选择器行为
        for d in derive_pool(op, base_seed=7):
            c = _case(code=d.code, operator=d.operator, dtype=d.dtype,
                      shape=d.shape, attrs=d.attrs)
            c.feature_atoms = label_case(c, by_name(d.operator))
            pool.append(c)
    return pool


def test_guided_selection_covers_all_reachable_atoms() -> None:
    pool = _pool_for_selection()
    # 池本身可达的原子，选择后必须全部覆盖（播种阶段的责任）
    pool_stats = coverage_stats(pool, UNIVERSE)
    selected, _ = guided_selection(
        pool, per_category=10, atom_universe=UNIVERSE,
        category_of=lambda c: by_name(c.operator).category.value,
    )
    sel_stats = coverage_stats(selected, UNIVERSE)
    assert set(sel_stats["covered"]) >= set(pool_stats["covered"]), \
        f"选择丢了池内可达原子：{sorted(set(pool_stats['covered']) - set(sel_stats['covered']))}"


def test_guided_selection_respects_category_quota() -> None:
    """配额语义：每类**补齐到**配额（或用尽候选）——
    覆盖播种阶段（保原子/保 dtype/保规模）允许超出配额，
    覆盖保证优先于类别均衡，这是设计而非缺陷。
    """
    from triton_eval.coverage import _case_signature

    pool = _pool_for_selection()
    selected, report = guided_selection(
        pool, per_category=8, atom_universe=UNIVERSE,
        category_of=lambda c: by_name(c.operator).category.value,
    )
    # 每类的去重候选数
    avail: dict[str, int] = {}
    for c in pool:
        k = by_name(c.operator).category.value
        avail[k] = avail.get(k, 0) + 0
    sigs: dict[str, set] = {}
    for c in pool:
        k = by_name(c.operator).category.value
        sigs.setdefault(k, set()).add(_case_signature(c))
    for category, v in report["per_category"].items():
        try:
            signatures = sigs[category]
        except KeyError as exc:
            raise AssertionError(f"Selection reported an unknown category: {category}") from exc
        assert v >= min(8, len(signatures)), \
            f"{category}: 配额是下限（覆盖播种可超出），实际 {v} < 8"
    assert len(selected) == len({id(c) for c in selected}), "不得重复选入"


def test_guided_selection_dedups_near_duplicates() -> None:
    """只差 seed 的两个候选是近重复——签名必须忽略 seed。"""
    a = _case(code="a", seed=1)
    b = _case(code="b", seed=999)
    a.feature_atoms = b.feature_atoms = ["FA-DT-FP32"]
    selected, _ = guided_selection(
        [a, b], per_category=10, atom_universe=["FA-DT-FP32"])
    assert len(selected) == 1


# ---------------- 池生成的合法性与规模 ----------------


@pytest.mark.torch
def test_derive_pool_legality_and_richness() -> None:
    """池子里不允许有非法用例（派生即校验），且五维都要真的出现。"""
    from triton_eval.casegen import derive_pool

    kinds, strided, sizes = set(), 0, set()
    total = 0
    for op in ALL_OPERATORS:
        for d in derive_pool(op, base_seed=3):
            total += 1
            op.validate_case_params(d.dtype, d.shape, d.attrs)  # 不抛即合法
            kinds.add((d.value_range.model_dump().get("kind")))
            strided += 0 if d.contiguous else 1
            sizes.add(size_class(prod(d.shape) or 1))
    assert total > 1000, f"候选池应有千级规模，实际 {total}"
    assert kinds >= {"normal", "zeros", "mixed", "extreme"}
    assert strided > 0, "池里必须有非连续候选"
    assert sizes == {"SMALL", "MEDIUM", "LARGE", "XLARGE"}, \
        f"四档规模必须齐（{sizes}）"


# ---------------- 输入维度落地（torch） ----------------


@pytest.mark.torch
def test_value_range_and_strided_inputs() -> None:
    """两个新维度必须真的作用到输入上，而不是只存字段。"""
    import torch

    from triton_eval.inputs import make_inputs_for

    z = make_inputs_for(_case(value_range={"kind": "zeros"}), torch.float32, "cpu")
    assert bool((z[0] == 0).all())

    m = make_inputs_for(_case(value_range={"kind": "mixed"}, shape=[8, 8]),
                        torch.float32, "cpu")
    flat = m[0].reshape(-1)
    assert bool((flat[flat.numel() // 2:] == 0).all()) and flat[: flat.numel() // 2].abs().sum() > 0

    e = make_inputs_for(_case(value_range={"kind": "extreme"}), torch.float32, "cpu")
    g = torch.Generator().manual_seed(1)
    assert e[0].abs().max() > 5.0, "extreme 应放大约 20 倍"

    s = make_inputs_for(_case(shape=[32, 128], contiguous=False), torch.float32, "cpu")
    assert not s[0].is_contiguous(), "跨步构造必须产生非连续输入"
    assert list(s[0].shape) == [32, 128], "逻辑 shape 必须不变"

    # extreme 安全阀：fp16 不该被放大
    h = make_inputs_for(_case(value_range={"kind": "extreme"}), torch.float16, "cpu")
    g2 = torch.Generator().manual_seed(1)
    ref = torch.randn(1024, generator=g2).to(torch.float16)
    assert torch.allclose(h[0], ref, atol=1e-2), "窄动态范围 dtype 必须跳过 extreme"


# ---------------- 融合原子（组合表驱动，2026-10-13） ----------------


def fusion_test_inputs():
    return {
        "LinearGelu": {"group_list": [32, 32]},
        "RmsNormLinear": {"group_list": [64]},
        "GroupedMatmulSwiGlu": {"group_list": [32, 32]},
        "TransposeLinear": {"group_list": [64, 64]},
        "MaskedSoftmax": {"dim": 1},
        "MaskedGelu": {},
        "GatherExp": {"dim": 1},
        "AddRmsNorm": {},
        "ApplyRotaryPosEmb": {},
        "MoeGatingTopKSoftmax": {"k": 2},
        "UnsortedSegmentSum": {"num_segments": 4},
    }


def test_fusion_atoms_derived_from_composition() -> None:
    """融合原子从组合表纯函数推导：两段融合必带 2STAGE + ELIDE；
    跨语义族组合带 CROSS-CATEGORY；全逐元素段带 ELEM-INTRA；
    含归约/归一化段带 RED-INTRA。非融合算子不得携带任何融合原子。
    """
    from triton_eval.operator_registry import (
        composition_of,
        fusion_operators,
    )

    # 每个融合算子一条合法属性（打标不校验属性，但保持与规格一致）
    fusion_attrs = fusion_test_inputs()
    assert set(fusion_attrs) == set(fusion_operators())
    for name, attrs in fusion_attrs.items():
        atoms = set(label_case(
            _case(operator=name, shape=[64, 128], attrs=attrs),
            by_name(name)))
        assert "FA-FUSE-2STAGE" in atoms, name
        assert "FA-FUSE-INTERM-ELIDE" in atoms, name

    # 跨语义族：收缩 × 逐元素 / 掩码 × 归一化 / 散射 × 归约
    for name, extra in (("LinearGelu", {"group_list": [32, 32]}),
                        ("MaskedSoftmax", {"dim": 1}),
                        ("UnsortedSegmentSum", {"num_segments": 4})):
        atoms = set(label_case(_case(operator=name, shape=[64, 128], attrs=extra),
                               by_name(name)))
        assert "FA-FUSE-CROSS-CATEGORY" in atoms, name

    # 逐元素族内融合：RoPE 的三角函数段 × 旋转段
    rope = set(label_case(_case(operator="ApplyRotaryPosEmb",
                                shape=[64, 64]), by_name("ApplyRotaryPosEmb")))
    assert "FA-FUSE-ELEM-INTRA" in rope

    # 含归约/归一化段：RmsNormLinear / MaskedSoftmax / AddRmsNorm
    for name, shape, extra in (
        ("RmsNormLinear", [64, 128], {"group_list": [64]}),
        ("MaskedSoftmax", [8, 128], {"dim": 1}),
        ("AddRmsNorm", [64, 128], {}),
    ):
        atoms = set(label_case(_case(operator=name, shape=shape, attrs=extra),
                               by_name(name)))
        assert "FA-FUSE-RED-INTRA" in atoms, name

    # 非融合算子：任何融合原子都不该出现
    for op in ALL_OPERATORS:
        if composition_of(op.name):
            continue
        atoms = set(label_case(_case(operator=op.name, shape=[1024],
                                     attrs={"dim": 0} if op.name in
                                     ("Softmax", "Cummin", "Scatter", "Gather")
                                     else {}), by_name(op.name)))
        assert not any(a.startswith("FA-FUSE-") for a in atoms), op.name


def test_fusion_labels_within_universe() -> None:
    """融合算子池的全部标签必须落在清单域内（标签器与 yaml 不许漂移）。"""
    from triton_eval.casegen import derive_pool
    from triton_eval.operator_registry import fusion_operators

    for name in fusion_operators():
        spec = by_name(name)
        for d in derive_pool(spec):
            case = _case(operator=d.operator, dtype=d.dtype,
                         shape=list(d.shape), attrs=dict(d.attrs),
                         contiguous=d.contiguous)
            for a in label_case(case, spec):
                assert a in UNIVERSE, f"{name} 打出清单外原子 {a}"


# ---------------- 优化手法原子（结构谓词，2026-10-13） ----------------


def test_pattern_atoms_targeted_and_gated() -> None:
    """每条手法原子都有定向载体（谓词命中），且小规模不误触发（谓词有门槛）。"""
    checks = [
        # (算子, shape, attrs, 期望原子)
        ("GroupedMatmul", [1024, 1024], {"group_list": [512, 512]},
         "FA-PAT-TILE-GEMM"),
        ("GroupedMatmul", [64, 65536], {"group_list": [64]},
         "FA-PAT-SPLIT-K"),
        ("GroupedMatmul", [3072, 128], {"group_list": [3072]},
         "FA-PAT-VEC-BROADCAST"),
        ("Exp", [65536], {}, "FA-PAT-VEC-LOAD"),
        ("Exp", [99971], {}, "FA-PAT-MASK-TAIL"),
        ("Cummin", [1048576], {"dim": 0}, "FA-PAT-MULTIPASS-REDUCE"),
        ("Scatter", [1024], {"dim": 0, "reduce": "add"}, "FA-CONCUR-ACCUM"),
        ("UnsortedSegmentSum", [1024], {"num_segments": 4},
         "FA-CONCUR-ACCUM"),
        ("GroupedMatmul", [96, 64], {"group_list": [12] * 8},
         "FA-CTRL-STATIC-UNROLL"),
        ("Gather", [65536], {"dim": 0}, "FA-PAT-COALESCE-GATHER"),
        ("Cummin", [1048576], {"dim": 0}, "FA-PAT-SCAN-BLOCKED"),
    ]
    for op, shape, attrs, want in checks:
        labels = label_case(_case(operator=op, shape=shape, attrs=attrs),
                            by_name(op))
        assert want in labels, f"{op} {shape} 未触发 {want}：{labels}"

    # 门槛反向：小负载/对齐形态不该触发对应手法原子
    small = label_case(_case(operator="GroupedMatmul", shape=[32, 32],
                             attrs={"group_list": [32]}),
                       by_name("GroupedMatmul"))
    assert "FA-PAT-TILE-GEMM" not in small
    assert "FA-PAT-SPLIT-K" not in small
    aligned = label_case(_case(operator="Exp", shape=[65536]), by_name("Exp"))
    assert "FA-PAT-MASK-TAIL" not in aligned
    misaligned = label_case(_case(operator="Exp", shape=[1000]), by_name("Exp"))
    assert "FA-PAT-VEC-LOAD" not in misaligned


def test_concur_accum_pool_int_only() -> None:
    """Scatter 并发累加变体只许宽整型进池——浮点的并发和不可判定，
    窄整型会溢出（跨维合法性 _combo_allowed）。
    """
    from triton_eval.casegen import derive_pool

    pool = derive_pool(by_name("Scatter"))
    add_cases = [c for c in pool if c.attrs.get("reduce") == "add"]
    assert add_cases, "并发累加变体必须存在（FA-CONCUR-ACCUM 载体）"
    assert all(c.dtype in ("int32", "int64") for c in add_cases), \
        [c.code for c in add_cases if c.dtype not in ("int32", "int64")]
    # 浮点用例全部走唯一索引路径（reduce 缺省）
    assert all("reduce" not in c.attrs for c in pool
               if c.dtype not in ("int32", "int64"))


def test_pattern_labels_ignore_seed() -> None:
    """手法谓词不得依赖 seed——近重复去重的签名不含 seed，
    若标签随 seed 变化，两个只差种子的用例会有不同语义。
    """
    a = label_case(_case(operator="GroupedMatmul", shape=[1024, 1024],
                         attrs={"group_list": [512, 512]}, seed=1),
                   by_name("GroupedMatmul"))
    b = label_case(_case(operator="GroupedMatmul", shape=[1024, 1024],
                         attrs={"group_list": [512, 512]}, seed=999983),
                   by_name("GroupedMatmul"))
    assert a == b
