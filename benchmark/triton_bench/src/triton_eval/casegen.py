# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""用例派生（L0，无 NPU 依赖）。

用例由**算子规格**派生，不依赖任何具体实现代码——这是支持外部提交的前提：
即使无人提交，用例也必须能被生成与校验。

五个派生维度：shape / dtype / attrs / value_range / 布局。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import CaseSource, CaseSpec, OperatorSpec as CorpusOperator
from .operatorspec import OperatorSpec
from .taxonomy import DEFAULT_CASES_PER_CATEGORY


class ValueRange(BaseModel):
    """输入数值范围。用于覆盖常规分布与边界值。"""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(default="normal", description="normal / zeros / large / mixed")
    low: float = Field(default=-1.0)
    high: float = Field(default=1.0)


class DerivedCase(BaseModel):
    """一个派生用例。只描述数据与判定参数，不含任何实现信息。"""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="业务唯一键，如 exp-f32-2d-0001")
    operator: str = Field(description="所属算子名")
    category: str = Field(description="所属语义族，冗余便于分组统计")
    level: str = Field(description="难度等级，冗余便于分组统计")
    dtype: str
    shape: list[int]
    attrs: dict[str, Any] = Field(default_factory=dict)
    value_range: ValueRange = Field(default_factory=ValueRange)
    seed: int = Field(description="输入数据随机种子")
    contiguous: bool = Field(default=True, description="输入是否连续")
    timeout_s: int = Field(default=60)
    content_hash: str = Field(default="")


def _case_code(op: OperatorSpec, dtype: str, idx: int) -> str:
    return f"{op.entry}-{dtype.replace('float', 'f').replace('bfloat', 'bf')}-{idx:02d}"


def derive_cases(
    op: OperatorSpec,
    *,
    count: int = DEFAULT_CASES_PER_CATEGORY,
    base_seed: int = 20261008,
) -> list[DerivedCase]:
    """为单个算子派生用例。所有派生参数 MUST 落在算子规格声明范围内。

    派生策略（五维）：
      - shape   ：覆盖 1D / 2D / 3D，含小 shape 与较大 shape
      - dtype   ：在算子支持的 dtype 内轮换，优先 fp32 → fp16 → 其余
      - attrs   ：默认值与非默认值混合，必填属性必给
      - value_range：normal / zeros / large / mixed 轮换
      - 布局    ：默认连续；LayoutTransform 类生成非连续用例
    """
    dtypes = _ordered_dtypes(op)
    shapes = _shapes_for(op)
    ranges = ["normal", "zeros", "large", "mixed", "normal"]

    cases: list[DerivedCase] = []
    for i in range(count):
        dtype = dtypes[i % len(dtypes)]
        shape = shapes[i % len(shapes)]
        attrs = _attrs_for(op, i, shape)
        # LayoutTransform 类的核心语义就是布局变换，故生成非连续用例
        contiguous = not (
            op.category.value == "LayoutTransform" and i == count - 1
        )

        case = DerivedCase(
            code=_case_code(op, dtype, i),
            operator=op.name,
            category=op.category.value,
            level=op.level.value,
            dtype=dtype,
            shape=shape,
            attrs=attrs,
            value_range=ValueRange(kind=ranges[i % len(ranges)]),
            seed=base_seed + i,
            contiguous=contiguous,
        )
        # 派生即校验：越界直接抛错，不产出非法用例
        op.validate_case_params(dtype, shape, attrs)
        cases.append(case)
    return cases


def _ordered_dtypes(op: OperatorSpec) -> list[str]:
    """dtype 轮换顺序：fp32 优先（精度判定最稳），再 fp16，再其余。"""
    allowed: list[str] = []
    declared = (op.case_dtypes if op.case_dtypes is not None else
                [dtype for item in op.inputs + op.outputs for dtype in item.dtype])
    for dtype in declared:
        if dtype not in allowed:
            allowed.append(dtype)
    priority = [d for d in ("float32", "float16", "bfloat16") if d in allowed]
    rest = [d for d in allowed if d not in priority]
    return priority + rest


def _shapes_for(op: OperatorSpec) -> list[list[int]]:
    """在 shape 约束内给出 1D / 2D / 3D 的代表形状。"""
    lo, hi = op.shape_support.min_rank, op.shape_support.max_rank
    candidates = [[1024], [32, 128], [4, 16, 64], [8, 256], [2, 4, 8, 16]]
    picked: list[list[int]] = []
    for s in candidates:
        if lo <= len(s) <= hi and s not in picked:
            picked.append(s)
    if not picked:
        picked = [[1] * max(1, lo)]
    return picked


def _attrs_for(op: OperatorSpec, idx: int, shape: list[int]) -> dict[str, Any]:
    """默认值与非默认值混合；必填属性必须给出。

    dim 类属性必须落在合法范围内：
      - 负值按 Python 惯例折算到 [0, rank)
      - SwiGlu 等还要求该 dim 的 size 为偶数，否则规格不成立
    """
    attrs: dict[str, Any] = {}
    for a in op.attrs:
        if a.required:
            attrs[a.name] = _non_default_value(a.type, idx, shape, a.name)
        elif op.attrs and list(op.attrs).index(a) == idx % len(op.attrs):
            attrs[a.name] = _non_default_value(a.type, idx, shape, a.name)

    for name in op.required_attrs():
        if name not in attrs:
            spec = op.attr(name)
            attrs[name] = _non_default_value(
                spec.type if spec else "int64", idx, shape, name
            )

    # dim 类属性：按 shape 归位，避免出现 dim=3 而 shape 只有 2 维
    for name in list(attrs):
        if name not in ("dim", "axis"):
            continue
        value = attrs[name]
        if not isinstance(value, int):
            continue
        rank = len(shape)
        resolved = value if value >= 0 else value + rank
        if not (0 <= resolved < rank):
            resolved = rank - 1
        attrs[name] = resolved

    # 取值为 None 的属性不入清单：那等同「用默认值」，
    # 留在清单里会让 validate_case_params 报类型不符（期望 string，实际 NoneType）。
    return {k: v for k, v in attrs.items() if v is not None}


def _non_default_value(
    type_: str, idx: int, shape: list[int] | None = None, name: str = ""
) -> Any:
    """给属性一个**合法**的非默认值。

    「合法」是硬约束：曾经对所有字符串属性统一返回 "sum"/"mean"，
    于是 gelu 收到 approximate="sum"、scatter 收到 reduce="mean"——
    两者都不是合法取值，参考实现会直接抛错，而错被记在被测方头上。
    """
    t = (type_ or "").lower()
    if "[]" in t or t.startswith("list"):
        # 列表型属性按语义分两类，不能一律当置换：
        #   perm       —— 必须是 rank 的合法置换（0..rank-1 各出现一次）
        #   group_list —— 分组大小，其**和必须等于被切分维的长度**，
        #                 否则 weight 的组数 E 与 x 的行数 M 对不上
        rank = len(shape) if shape else 2
        if name == "group_list":
            return _group_list(shape)
        if rank <= 1:
            return [0]
        perm = list(range(rank))
        perm[0], perm[-1] = perm[-1], perm[0]
        return perm
    if t.startswith("int"):
        # dim 取最后一维（最常用且语义稳定），再由调用方按 shape 归位
        return -1
    if t.startswith("float"):
        return 2.0 + idx
    if t.startswith("bool"):
        return bool(idx % 2)
    if t.startswith("str"):
        return _string_variant(name, idx)
    return None


def _group_list(shape: list[int] | None) -> list[int]:
    """分组大小：均分 3 组，余数并入最后一组。"""
    m = shape[0] if shape else 1
    groups = 3 if m >= 3 else 1
    base, extra = divmod(m, groups)
    out = [base] * groups
    out[-1] += extra
    return [g for g in out if g > 0]


def _string_variant(name: str, idx: int) -> Any:
    """字符串属性的取值域按属性名限定。"""
    if name == "approximate":
        return "tanh"          # gelu 唯一的非默认取值
    if name == "reduction":
        return "sum" if idx % 2 else "mean"
    if name == "reduce":
        # scatter 的归约方式：非默认取值会改变语义（add / mul），
        # 且当前参考实现先覆盖「不归约」这一主路径，故保留默认。
        return None
    return "sum" if idx % 2 else "mean"


def generate_all(
    operators: list[OperatorSpec],
    *,
    per_category: int = DEFAULT_CASES_PER_CATEGORY,
) -> list[DerivedCase]:
    """按 **category** 生成用例，每个 category 绑定固定数量。

    额度按 category 分配，而非按算子：一个 category 下若有多个算子，
    则均分额度（余数给前几个）。这样总数是
    「category 数 × per_category」，而不是「算子数 × per_category」。
    """
    grouped: dict[str, list[OperatorSpec]] = {}
    for op in operators:
        grouped.setdefault(op.category.value, []).append(op)

    out: list[DerivedCase] = []
    for _category, ops in sorted(grouped.items()):
        base, extra = divmod(per_category, len(ops))
        for i, op in enumerate(ops):
            share = base + (1 if i < extra else 0)
            if share:
                out.extend(derive_cases(op, count=share))
    return out


# ==================== 候选池生成（覆盖引导生成的前半段） ====================
#
# derive_cases 是「按配方轮换」的 MVP 生成；derive_pool 是参数化生成 2.0：
# 把每个算子的参数空间摊开成大候选池，交给 coverage.guided_selection
# 做覆盖引导选择。池子只管**合法**（派生即校验），不管取舍。


def _timeout_for(numel: int) -> int:
    """规模越大窗口越宽：XL 的参考实现与冷编译都需要更多时间。"""
    if numel > 2 * 1024 * 1024:
        return 120
    if numel > 256 * 1024:
        return 90
    return 60


def _shapes_pool(op: OperatorSpec) -> list[list[int]]:
    """形态 × 规模的形状池。对齐/非对齐、方/扁/细长/立方都占位。

    每个算子的形状必须落进其 rank 支持范围；分组矩阵乘固定 rank-2，
    CEL 是 (N,C) 语义，给出类目数有意义的形状。
    收缩类融合（LinearGelu 等）沿用 GroupedMatmul 的形态池——
    融合算子的负载形态与原语收缩一致，另加一条高瘦形态（M≤64、K≥2^16）
    定向触达 split-K 手法原子。
    """
    lo, hi = op.shape_support.min_rank, op.shape_support.max_rank

    def rank_ok(r: int) -> bool:
        return lo <= r <= hi

    candidates = _special_shape_pool(op)
    if candidates is None:
        candidates = _rank_shape_pool(rank_ok)

    # SwiGlu 需要末维为偶（拆两半）；奇数末维微调为最近的偶数
    if op.name in ("SwiGlu", "GroupedMatmulSwiGlu"):
        fixed = []
        for s in candidates:
            if s and s[-1] % 2 == 1:
                s = list(s)
                s[-1] += 1
            fixed.append(s)
        candidates = fixed
    return candidates


def _gemm_shape_pool():
    return [
        [32, 32],            # S 方（收缩类最小有意义的乘法规模）
        [96, 64],            # S 方
        [128, 1024],         # S 细长（行矮列宽）
        [1024, 128],         # M 细长（行高列窄）
        [3072, 128],         # M
        [1024, 1024],        # L 方
        [64, 65536],         # XL 高瘦（split-K 载体：M 并行度不足拆 K 补）
        [8192, 256],         # L 细长
        [2048, 2048],        # XL 方
        [16384, 128],        # XL 细长
    ]


def _special_shape_pool(op):
    if op.name in ("GroupedMatmul", "LinearGelu", "RmsNormLinear",
                   "GroupedMatmulSwiGlu", "TransposeLinear"):
        return _gemm_shape_pool()
    if op.name == "CrossEntropyLoss":
        return [
            [32, 100],           # S
            [128, 1000],         # M 方偏
            [1000, 128],         # M 细长
            [1024, 1000],        # L
            [4096, 512],         # L 细长
            [2048, 2048],        # XL
        ]
    if op.name == "ApplyRotaryPosEmb":
        # (S, D) / (B, S, D)，head_dim 取主流值 64/128（必须为偶）。
        # 档位按 size_class 边界（4K/256K/2M）标注——四档保底是方案 §2.2
        # 的设计规则，XL（超长序列）不可缺
        return [
            [64, 64],            # S（边界值：恰在 4K 上限）
            [1000, 64],          # M 非对齐
            [999, 64],           # M 非对齐（素数行数）
            [8, 256, 64],        # M 立方（B,S,D）
            [2, 1024, 128],      # M 立方（边界值：恰在 256K 上限）
            [4096, 128],         # L
            [32768, 128],        # XL 超长序列（四档保底）
        ]
    if op.name == "MoeGatingTopKSoftmax":
        # (T, E) / (B, T, E)：E 为专家数，取主流路由规模（32→256）。
        # S 与 XL 是方案 §2.2 的四档保底，此前缺失、2026-10-13 补齐
        return [
            [64, 32],            # S（四档保底）
            [1024, 32],          # M
            [999, 128],          # M 非对齐
            [4096, 64],          # M（边界值：恰在 256K 上限）
            [2, 2048, 64],       # M 立方（边界值：恰在 256K 上限）
            [8192, 128],         # L
            [16384, 256],        # XL（DeepSeek 级专家数，四档保底）
        ]
    if op.name == "UnsortedSegmentSum":
        # 主维 N 是段元素数；num_segments 由属性变体给出（≤ 64 有界，
        # 参考实现逐段掩码求和的循环开销由它决定）
        return [
            [1024],              # S
            [1000],              # S 非对齐
            [65536],             # M
            [99971],             # M 素数
            [4096, 64],          # M 带特征列
            [65536, 32],         # L 带特征列
            [262144, 16],        # L
            [1048576],           # XL
        ]
    return None


def _rank_shape_pool(rank_ok):
    candidates = []
    if rank_ok(0):
        candidates.append([])            # 标量：0 维边界
    if rank_ok(1):
        candidates += [
            [1024],           # S 对齐
            [1000],           # S 非对齐（块尾 mask）
            [65536],          # M
            [99971],          # M 素数（非对齐极端）
            [1048576],        # L
            [4194304],        # XL
        ]
    if rank_ok(2):
        candidates += [
            [32, 32],         # S 方
            [1000, 999],      # S 非对齐
            [4096, 64],       # M 细长
            [128, 2048],      # M 细长（反向）
            [512, 512],       # M 方
            [1024, 1024],     # L 方
            [8192, 256],      # L 细长
            [2048, 2048],     # XL 方
        ]
    if rank_ok(3):
        candidates += [
            [4, 16, 64],      # S 立方
            [8, 64, 64],      # M
            [16, 64, 128],    # M
            [40, 40, 40],     # M 真立方
        ]
    if rank_ok(4):
        candidates += [
            [2, 4, 8, 16],    # S
            [4, 8, 16, 32],   # M
        ]
    if rank_ok(5):
        candidates.append([2, 4, 4, 4, 4])
    return candidates


def _attr_variants(op: OperatorSpec, shape: list[int]) -> list[dict[str, Any]]:
    """每个算子的属性变体：默认档 + 定向变体（打轴位置与非默认属性盲区）。"""
    rank = len(shape)
    variants: list[dict[str, Any]] = []

    def dim_variants(*, required: bool) -> list[int]:
        """默认档=末维；变体=首维；rank≥3 加中间维。"""
        out = []
        if rank == 0:
            return [0]
        out.append(rank - 1)
        if rank >= 2:
            out.append(0)
        if rank >= 3:
            out.append(1)
        return out

    variants = _operator_attr_variants(op, shape, rank, dim_variants)

    # 兜底：必填属性缺失的变体补上合法值
    for name in op.required_attrs():
        spec = op.attr(name)
        if all(name in v for v in variants):
            continue
        for v in variants:
            v.setdefault(name, rank - 1 if rank else 0)
    return [v for v in variants if v is not None]


def _axis_attr_variants(op, rank, dim_variants):
    variants = []
    if op.name in ("Softmax", "Cummin", "ArgMax", "SwiGlu", "Gather", "Scatter"):
        for d in dim_variants(required=True):
            v: dict[str, Any] = {"dim": d}
            if op.name == "ArgMax" and d == 0 and rank >= 1:
                v = {"dim": d, "keepdim": True}  # 顺带一个非默认 keepdim
            variants.append(v)
        if op.name == "ArgMax":
            variants.append({"dim": max(rank - 1, 0)})  # keepdim 默认档
        if op.name == "Scatter":
            # 并发累加变体：reduce=add 允许索引重复——「和」与写入顺序无关
            # 是语义要求（FA-CONCUR-ACCUM 载体）。仅整型合法（int32/int64，
            # 由 _combo_allowed 限定）：浮点加法不满足结合律，实现与参考的
            # 累加顺序差异会变成 ULP 级噪声；窄整型会溢出（99*N 超出 int16）。
            for d in dim_variants(required=True):
                variants.append({"dim": d, "reduce": "add"})
    elif op.name == "MaskedSoftmax":
        for d in dim_variants(required=True):
            variants.append({"dim": d})
    elif op.name == "GatherExp":
        for d in dim_variants(required=True):
            variants.append({"dim": d})
        # exp 段的非默认属性档（底数/缩放/偏移一起换）
        variants.append({"dim": max(rank - 1, 0),
                         "base": 2.0, "scale": 0.5, "shift": 1.0})
    return variants


def _fused_attr_variants(op, shape, rank):
    variants = []
    grouped_dim = shape[1] if op.name == "TransposeLinear" else shape[0]
    groups = 3 if grouped_dim >= 3 else 1
    base, extra = divmod(grouped_dim, groups)
    gl = [base] * groups
    gl[-1] += extra
    variants.append({"group_list": [g for g in gl if g > 0]})
    if grouped_dim >= 16:
        groups8 = 8
        b8, e8 = divmod(grouped_dim, groups8)
        gl8 = [b8] * groups8
        gl8[-1] += e8
        variants.append({"group_list": [g for g in gl8 if g > 0]})
    if op.name == "RmsNormLinear":
        variants.append({**variants[0], "epsilon": 1e-3})
    return variants


def _transpose_attr_variants(op, shape, rank):
    variants = []
    if rank >= 2:
        swap = list(range(rank))
        swap[0], swap[-1] = swap[-1], swap[0]
        variants.append({"perm": swap})
    if rank >= 3:
        rotate = list(range(1, rank)) + [0]
        variants.append({"perm": rotate})
    return variants


def _grouped_attr_variants(op, shape, rank):
    variants = []
    m = shape[0] if shape else 96
    groups = 3 if m >= 3 else 1
    base, extra = divmod(m, groups)
    gl = [base] * groups
    gl[-1] += extra
    variants.append({"group_list": [g for g in gl if g > 0]})
    if m >= 16:
        groups8 = 8
        b8, e8 = divmod(m, groups8)
        gl8 = [b8] * groups8
        gl8[-1] += e8
        variants.append({"group_list": [g for g in gl8 if g > 0]})
    return variants


def _fixed_attr_variants(name):
    variants = {
        'Gelu': [{}, {'approximate': 'tanh'}],
        'Exp': [{}, {'base': 2.0, 'scale': 0.5, 'shift': 1.0}],
        'MaskedScale': [{}, {'scale': 2.5}],
        'RmsNorm': [{}, {'epsilon': 0.001}],
        'AddRmsNorm': [{}, {'epsilon': 0.001}],
        'ApplyRotaryPosEmb': [{}, {'rotary_mode': 'interleaved'}],
        'MoeGatingTopKSoftmax': [{'k': 2}, {'k': 8}],
        'UnsortedSegmentSum': [{'num_segments': 4}, {'num_segments': 16}, {'num_segments': 64}],
        'CrossEntropyLoss': [{}, {'reduction': 'sum'}],
        'MaskedGelu': [{}, {'approximate': 'tanh'}],
    }
    return variants.get(name, [{}])


def _operator_attr_variants(op, shape, rank, dim_variants):
    if op.name in ("Softmax", "Cummin", "ArgMax", "SwiGlu", "Gather", "Scatter", "MaskedSoftmax", "GatherExp"):
        return _axis_attr_variants(op, rank, dim_variants)
    if op.name in ("LinearGelu", "RmsNormLinear", "GroupedMatmulSwiGlu", "TransposeLinear"):
        return _fused_attr_variants(op, shape, rank)
    if op.name == "Transpose":
        return _transpose_attr_variants(op, shape, rank)
    if op.name == "GroupedMatmul":
        return _grouped_attr_variants(op, shape, rank)
    return _fixed_attr_variants(op.name)


_VALUE_KINDS_ALL = ("normal", "zeros", "mixed", "extreme")
_VALUE_KINDS_NARROW = ("normal", "zeros", "mixed")  # fp16 等窄动态范围禁止 extreme


def _combo_allowed(op: OperatorSpec, dtype: str, attrs: dict[str, Any]) -> bool:
    """维度间的组合合法性——validate_case_params 只查单维越界，
    查不到「某属性变体只在某类 dtype 下语义可判定」这类跨维约束。

    已知规则：
      Scatter reduce=add（并发累加）仅整型 int32/int64 合法——
      浮点加法不满足结合律，重复索引下「和」依赖写入顺序，实现与参考
      的顺序差异会把语义差异伪装成噪声；窄整型（int8/int16/uint8）会
      溢出（updates+data 的和超出表示范围）。整型宽位的「和」精确且
      顺序无关——这才是可判定的并发累加。

      GatherExp / MoeGatingTopKSoftmax 的 case dtype 必须落在**主输入**
      声明的 dtype 集：validate_case_params 的合法域取的是全部输入输出
      的并集，这两个算子的索引位（GatherExp.index）与索引输出
      （MoeGating 的 expert/row indices）会把整型混进并集——于是生成出
      「对整数做 exp」（e^99 ≈ 1e43 远超 int64，转回整型即饱和截断，
      语义病态）与「整型 logits 过 softmax」（截断出大量并列值，topk
      选中下标在实现与参考间合法分歧）的用例。两算子没有 masked_scale
      那种「整型 dtype 路由给掩码/索引位」的构造约定，故直接限定。
    """
    if op.name == "Scatter" and attrs.get("reduce") == "add":
        return dtype in ("int32", "int64")
    if op.name in ("GatherExp", "MoeGatingTopKSoftmax"):
        x_dtypes = {d for d in op.inputs[0].dtype}
        return dtype in x_dtypes
    return True


def derive_pool(
    op: OperatorSpec,
    *,
    base_seed: int = 20261009,
) -> list[DerivedCase]:
    """摊开一个算子的参数空间成候选池（覆盖引导选择的上游）。

    池只保证**合法**（派生即校验），不做取舍；取舍在 coverage.guided_selection。
    候选量级：形状（8-16）× dtype（3-6）× 属性变体（1-3），
    数值分布与布局按轮转混入，避免池子被单一形态占据。
    """
    from itertools import product

    dtypes = _ordered_dtypes(op)
    shapes = _shapes_pool(op)
    out: list[DerivedCase] = []
    idx = 0
    for si, shape in enumerate(shapes):
        rank = len(shape)
        attr_variants = _attr_variants(op, shape)
        numel = 1
        for d in shape:
            numel *= d
        for (di, dtype), (ai, attrs) in product(enumerate(dtypes), enumerate(attr_variants)):
            kinds = _VALUE_KINDS_ALL if dtype in ("float32", "bfloat16") else _VALUE_KINDS_NARROW
            # 数值分布按轮转混入：同一 (形状, dtype) 的不同属性变体
            # 拿到不同分布，池子的每个角落都有边界值候选
            kind = kinds[(si + di + ai) % len(kinds)]
            # 布局：rank≥2 的候选里轮转混入非连续
            contiguous = not (rank >= 2 and (idx + si) % 4 == 3)
            case = DerivedCase(
                code=f"{op.entry}-{_dtype_tag(dtype)}-{_size_tag(numel)}{idx:02d}",
                operator=op.name,
                category=op.category.value,
                level=op.level.value,
                dtype=dtype,
                shape=list(shape),
                attrs=dict(attrs),
                value_range=ValueRange(kind=kind),
                seed=base_seed + idx * 7919,
                contiguous=contiguous,
                timeout_s=_timeout_for(numel),
            )
            # 派生即校验：越界直接跳过（池子宁缺毋滥）
            if not _valid_pool_parameters(op, dtype, shape, attrs):
                continue
            if not _combo_allowed(op, dtype, dict(attrs)):
                continue
            out.append(case)
            idx += 1
    return out


def _valid_pool_parameters(op, dtype, shape, attrs):
    try:
        op.validate_case_params(dtype, list(shape), dict(attrs))
    except Exception:  # noqa: BLE001
        return False
    return True


def _dtype_tag(dtype: str) -> str:
    return (dtype.replace("float", "f").replace("bfloat", "bf")
                 .replace("int", "i").replace("uint8", "u8"))


def _size_tag(numel: int) -> str:
    if numel <= 4 * 1024:
        return "S"
    if numel <= 256 * 1024:
        return "M"
    if numel <= 2 * 1024 * 1024:
        return "L"
    return "X"


def count_by_category(cases: list[DerivedCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.category] = counts.get(c.category, 0) + 1
    return counts


def count_by_level(cases: list[DerivedCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.level] = counts.get(c.level, 0) + 1
    return counts


def manifest_operators(operators: list[OperatorSpec]) -> list[CorpusOperator]:
    """Build the corpus declarations shared by the historical builders."""
    declarations = []
    for operator in operators:
        declarations.append(CorpusOperator(
            name=operator.name, source=CaseSource.VENDOR_SELFTEST, weight=1.0,
            ref_path=f"operators/{operator.entry}/ref.py",
            desc=f"{operator.category.value} / {operator.level.value}",
        ))
    return declarations


def manifest_candidates(operators: list[OperatorSpec]) -> list[CaseSpec]:
    """Convert the ordered derived pool to labeled manifest cases."""
    from .coverage import label_case

    cases = []
    for operator in operators:
        for derived in derive_pool(operator):
            case = CaseSpec(
                code=derived.code, operator=derived.operator, source=CaseSource.VENDOR_SELFTEST,
                dtype=derived.dtype, shape=derived.shape, attrs=derived.attrs, seed=derived.seed,
                launch={}, timeout_s=derived.timeout_s, feature_atoms=[],
                value_range=dict(derived.value_range.model_dump()), contiguous=derived.contiguous,
            )
            case.feature_atoms = label_case(case, operator)
            cases.append(case)
    return cases
