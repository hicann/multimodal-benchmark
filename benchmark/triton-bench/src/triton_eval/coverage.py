# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""覆盖度量与覆盖引导选择（L0，无 NPU 依赖）。

范式核心——「标签是算子规格与用例五维的**纯函数**」：

    label_case(case, spec) → 一组 FA 原子 id

这条硬边界保证：
  1. 客观性：标签与任何实现、任何提交方无关，谁算都一样；
  2. 可冻结：标签随用例进 manifest，受内容哈希保护，分母分子不漂移；
  3. 可引导：「缺哪个原子」能反推「补哪类参数」——覆盖引导的靶子明确。

选择算法（guided_selection）三阶段：
  A 原子播种——每个未覆盖原子挑一个最廉价的候选（优先小负载、常规数值）；
  B 配额保底——每个算子在每档规模、每个 dtype 上至少有代表，
    防止「覆盖了原子但只剩单一形态」的畸形解；
  C 多样性补齐——轮转补足目标数，优先引入新的 (算子, 原子) 组合。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, NamedTuple

from .models import CaseSpec
from .operatorspec import OperatorSpec

# 规模档位边界（主输入元素数）
SIZE_SMALL = 4 * 1024              # ≤ 4K：启动开销主导
SIZE_MEDIUM = 256 * 1024           # ≤ 256K
SIZE_LARGE = 2 * 1024 * 1024       # ≤ 2M
# > 2M 为 XLARGE


def size_class(numel: int) -> str:
    if numel <= SIZE_SMALL:
        return "SMALL"
    if numel <= SIZE_MEDIUM:
        return "MEDIUM"
    if numel <= SIZE_LARGE:
        return "LARGE"
    return "XLARGE"


# ---------------- 标签器 ----------------

_OP_ALWAYS_ATOMS: dict[str, list[str]] = {
    # 算子级原子：该算子的任何用例都隐含的语言能力
    # Exp：y = exp(base*x*scale + shift) —— 乘缩放与加偏移都在语义里
    "Exp": ["FA-MATH-EXP", "FA-MATH-MUL", "FA-MATH-ADD"],
    "Gelu": ["FA-MATH-GELU"],
    "Cummin": ["FA-REDUCE-MIN", "FA-REDUCE-ARG", "FA-MULTI-OUTPUT"],
    "Maximum": ["FA-MATH-MAXMIN", "FA-MULTI-INPUT"],
    "MaskedScale": ["FA-IDX-WHERE", "FA-MATH-MUL", "FA-MULTI-INPUT"],
    "Softmax": ["FA-NORM-SOFTMAX", "FA-REDUCE-SUM", "FA-REDUCE-MAX",
                "FA-MATH-EXP", "FA-STABLE-MAXSUB"],
    "RmsNorm": ["FA-NORM-RMS", "FA-REDUCE-SUM", "FA-MATH-SQRT",
                "FA-MULTI-INPUT"],
    "Gather": ["FA-IDX-GATHER", "FA-MULTI-INPUT"],
    "Scatter": ["FA-IDX-SCATTER", "FA-MULTI-INPUT"],
    "GroupedMatmul": ["FA-LINALG-DOT", "FA-LINALG-BIAS", "FA-MULTI-INPUT"],
    "Transpose": ["FA-IDX-PERMUTE"],
    "ArgMax": ["FA-REDUCE-MAX", "FA-REDUCE-ARG"],
    "CrossEntropyLoss": ["FA-STABLE-LOGSUMEXP", "FA-REDUCE-SUM",
                         "FA-MATH-EXP", "FA-MULTI-INPUT"],
    "SwiGlu": ["FA-MATH-SIGMOID", "FA-MATH-MUL"],
    # ---- 融合算子（第 3 批）：算子级原子 = 段落语言能力的并集 ----
    # （语义上两段能力都必然用到；融合结构本身由 _fusion_atoms 另行打标）
    "LinearGelu": ["FA-LINALG-DOT", "FA-LINALG-BIAS", "FA-MULTI-INPUT",
                   "FA-MATH-GELU"],
    "RmsNormLinear": ["FA-NORM-RMS", "FA-REDUCE-SUM", "FA-MATH-SQRT",
                      "FA-LINALG-DOT", "FA-LINALG-BIAS", "FA-MULTI-INPUT"],
    "GroupedMatmulSwiGlu": ["FA-LINALG-DOT", "FA-LINALG-BIAS",
                            "FA-MULTI-INPUT", "FA-MATH-SIGMOID",
                            "FA-MATH-MUL"],
    "TransposeLinear": ["FA-IDX-PERMUTE", "FA-LINALG-DOT", "FA-LINALG-BIAS",
                        "FA-MULTI-INPUT"],
    "MaskedSoftmax": ["FA-IDX-WHERE", "FA-MULTI-INPUT", "FA-NORM-SOFTMAX",
                      "FA-REDUCE-SUM", "FA-REDUCE-MAX", "FA-MATH-EXP",
                      "FA-STABLE-MAXSUB"],
    "MaskedGelu": ["FA-IDX-WHERE", "FA-MULTI-INPUT", "FA-MATH-GELU"],
    "GatherExp": ["FA-IDX-GATHER", "FA-MULTI-INPUT", "FA-MATH-EXP",
                  "FA-MATH-MUL", "FA-MATH-ADD"],
    "AddRmsNorm": ["FA-MATH-ADD", "FA-NORM-RMS", "FA-REDUCE-SUM",
                   "FA-MATH-SQRT", "FA-MULTI-INPUT"],
    "ApplyRotaryPosEmb": ["FA-MATH-SINCOS", "FA-MATH-MUL", "FA-MATH-ADD",
                          "FA-MULTI-INPUT"],
    "MoeGatingTopKSoftmax": ["FA-NORM-SOFTMAX", "FA-REDUCE-SUM",
                             "FA-REDUCE-MAX", "FA-MATH-EXP",
                             "FA-STABLE-MAXSUB", "FA-REDUCE-ARG",
                             "FA-MULTI-OUTPUT"],
    "UnsortedSegmentSum": ["FA-IDX-SCATTER", "FA-REDUCE-SUM",
                           "FA-MULTI-INPUT"],
}

_DTYPE_ATOMS = {
    "float32": "FA-DT-FP32",
    "float16": "FA-DT-FP16",
    "bfloat16": "FA-DT-BF16",
    "int8": "FA-DT-INT8",
    "uint8": "FA-DT-UINT8",
    "int16": "FA-DT-INT16",
    "int32": "FA-DT-INT32",
    "int64": "FA-DT-INT64",
}


def _axis_atom(rank: int, dim: Any) -> str | None:
    """轴位置原子。dim 已按规格归一为非负。"""
    if not isinstance(dim, int) or rank <= 0:
        return None
    resolved = dim if dim >= 0 else dim + rank
    if not (0 <= resolved < rank):
        return None
    if resolved == rank - 1:
        return "FA-AXIS-LAST"
    if resolved == 0:
        return "FA-AXIS-FIRST"
    return "FA-AXIS-INNER"  # 中间维——只认末维的实现必错（ADV-F7 类盲区）


def _shape_atoms(shape: list[int]) -> list[str]:
    atoms: list[str] = []
    rank = len(shape)
    if rank == 1:
        atoms.append("FA-SHAPE-FLAT")
    elif rank == 2:
        r, c = shape[0], shape[1]
        if r == c:
            atoms.append("FA-SHAPE-SQUARE")
        if max(r, c) >= 8 * max(1, min(r, c)):
            atoms.append("FA-SHAPE-SKINNY")
    elif rank == 3:
        atoms.append("FA-SHAPE-CUBE")
    if rank >= 4:
        atoms.append("FA-PROG-GRID4D")
    elif rank == 3:
        atoms.append("FA-PROG-GRID3D")
    elif rank == 2:
        atoms.append("FA-PROG-GRID2D")
    else:
        atoms.append("FA-PROG-GRID1D")
    return atoms


def _fusion_atoms(operator: str) -> list[str]:
    """融合原子：从组合表（operator_registry.COMPOSITIONS）纯函数推导。

    组合表是算子侧的规格声明（段落 + 语义族），不依赖任何实现——
    满足标签纯函数边界。两段融合隐含「中间结果不出内核」的考察，
    故 INTERM-ELIDE 与 2STAGE 一起携带。
    """
    from .operator_registry import composition_of

    comp = composition_of(operator)
    if not comp:
        return []
    atoms = ["FA-FUSE-2STAGE", "FA-FUSE-INTERM-ELIDE"]
    cats = [s.category.value for s in comp]
    if len(set(cats)) >= 2:
        atoms.append("FA-FUSE-CROSS-CATEGORY")
    if all(c == "Elementwise" for c in cats):
        atoms.append("FA-FUSE-ELEM-INTRA")
    if any(c in ("Reduction", "Normalization") for c in cats):
        atoms.append("FA-FUSE-RED-INTRA")
    return atoms


# 优化手法原子的结构谓词阈值。这些常数是「收益显著」的结构声明——
# 每条原子的收益须经真机探针实证（docs/composition-and-pattern-dimensions.md
# §3.3），实证不达标的原子在披露中如实降级为「声明性考察」。
_TILE_MIN_ELEMENTS = 2 ** 18      # 分块：朴素全局访存开始慢一个量级
_SPLITK_MAX_M = 64                # split-K：M/N 并行度不足
_SPLITK_MIN_K = 2 ** 16
_BIAS_BROADCAST_MIN_M = 2048      # 偏置广播驻留：行数多才有摊销价值
_VECLOAD_MIN_LAST = 4096          # 向量化访存：最内维足够长且对齐
_VECLOAD_ALIGN = 64
_MASKTAIL_MIN_LAST = 2048         # 边界掩码：尾部不满块开始有代价
_MULTIPASS_MIN_REDUCED = 2 ** 20  # 多趟归约：归约维超出单趟寄存器容量
_COALESCE_MIN_ELEMENTS = 2 ** 16  # 聚集访存：大表查访的局部性收益
_SCAN_MIN_ELEMENTS = 2 ** 20      # 分块扫描：两趟策略的分水岭


def _pattern_atoms(case: CaseSpec, spec: OperatorSpec) -> list[str]:
    """优化手法原子（性能负载型）：结构谓词纯函数推导。

    只看算子语义族 + 形状/规模/布局/属性——不看任何实现。语义：
    该用例的结构使「使用手法 X」与「不使用」存在可测量的性能差距；
    谁都能算对，手法水平体现在计时上。
    """
    cat = spec.category.value
    shape = list(case.shape)
    numel = 1
    for d in shape:
        numel *= d
    last = shape[-1] if shape else 0
    atoms: list[str] = []

    if cat == "Contraction":
        _contraction_pattern_atoms(shape, numel, atoms)

    if cat in ("Elementwise", "MaskPredicate"):
        if case.contiguous and last >= _VECLOAD_MIN_LAST \
                and last % _VECLOAD_ALIGN == 0:
            atoms.append("FA-PAT-VEC-LOAD")
        if last >= _MASKTAIL_MIN_LAST and last % _VECLOAD_ALIGN != 0:
            atoms.append("FA-PAT-MASK-TAIL")

    if cat in ("Reduction", "SortSelect", "NumericalStable"):
        _reduction_pattern_atoms(case, shape, last, atoms)

    # 并发累加（正确性负载）：载体算子 + 归约属性
    if case.operator == "UnsortedSegmentSum" or (
        case.operator == "Scatter" and case.attrs.get("reduce") == "add"
    ):
        atoms.append("FA-CONCUR-ACCUM")

    # 静态展开：小循环次数可由用例定义推导
    if _has_static_group_loop(case):
        atoms.append("FA-CTRL-STATIC-UNROLL")

    if cat == "IndexGather" and numel >= _COALESCE_MIN_ELEMENTS:
        atoms.append("FA-PAT-COALESCE-GATHER")

    if case.operator == "Cummin" and numel >= _SCAN_MIN_ELEMENTS:
        atoms.append("FA-PAT-SCAN-BLOCKED")

    return atoms


def _contraction_pattern_atoms(shape, numel, atoms):
    if numel >= _TILE_MIN_ELEMENTS:
        atoms.append("FA-PAT-TILE-GEMM")
    if len(shape) >= 2:
        m, k = shape[0], shape[1]
        if m <= _SPLITK_MAX_M and k >= _SPLITK_MIN_K:
            atoms.append("FA-PAT-SPLIT-K")
        if m >= _BIAS_BROADCAST_MIN_M:
            atoms.append("FA-PAT-VEC-BROADCAST")


def _reduction_pattern_atoms(case, shape, last, atoms):
    dim = case.attrs.get("dim", case.attrs.get("axis"))
    if isinstance(dim, int) and shape:
        resolved = dim if dim >= 0 else dim + len(shape)
        reduced = shape[resolved] if 0 <= resolved < len(shape) else last
    else:
        reduced = last
    if reduced >= _MULTIPASS_MIN_REDUCED:
        atoms.append("FA-PAT-MULTIPASS-REDUCE")


def _has_static_group_loop(case):
    return case.operator == "GroupedMatmul" and len(case.attrs.get("group_list", []) or []) <= 8


def label_case(case: CaseSpec, spec: OperatorSpec | None) -> list[str]:
    """给一个用例打语言特性标签。纯函数、确定性、不依赖任何实现。"""
    if spec is None:
        return []

    atoms: list[str] = list(_OP_ALWAYS_ATOMS.get(case.operator, []))
    from .official_catalog import SEMANTIC_ATOMS

    atoms.extend(SEMANTIC_ATOMS.get(case.operator, []))
    numel = 1
    for d in case.shape:
        numel *= d

    # 融合结构原子（组合表驱动）与优化手法原子（结构谓词驱动）
    atoms.extend(_fusion_atoms(case.operator))
    atoms.extend(_pattern_atoms(case, spec))

    # dtype
    if case.dtype in _DTYPE_ATOMS:
        atoms.append(_DTYPE_ATOMS[case.dtype])

    # 形态与网格
    atoms.extend(_shape_atoms(case.shape))

    # 规模档位
    sc = size_class(numel)
    atoms.append(f"FA-LOAD-{sc}")

    # 对齐：块尾是否需要 mask
    atoms.append("FA-LOAD-ALIGNED" if numel % 1024 == 0 else "FA-LOAD-MASKED-EDGE")

    # 布局
    atoms.append("FA-LOAD-CONTIG" if case.contiguous else "FA-LOAD-STRIDED")

    # 数值分布
    kind = (case.value_range or {}).get("kind", "normal")
    value_atom = {
        "zeros": "FA-NUM-ZEROS",
        "mixed": "FA-NUM-MIXED",
        "extreme": "FA-NUM-EXTREME",
    }.get(kind)
    if value_atom:
        atoms.append(value_atom)

    # 轴位置（dim / axis 型属性）
    axis = _axis_atom(len(case.shape), case.attrs.get("dim", case.attrs.get("axis")))
    if axis:
        atoms.append(axis)

    # 非默认属性与列表型属性
    defaults = {a.name: a.default for a in spec.attrs}
    if any(case.attrs.get(k, d) != d for k, d in defaults.items()):
        atoms.append("FA-ATTR-NONDEFAULT")
    if any(isinstance(v, list) for v in case.attrs.values()):
        atoms.append("FA-ATTR-LIST")

    # 去重且保序
    seen: set[str] = set()
    out: list[str] = []
    for a in atoms:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


# ---------------- 覆盖统计 ----------------


def coverage_stats(
    cases: Iterable[Any], atom_universe: list[str]
) -> dict[str, Any]:
    """统计一组（已打标）用例对原子清单的覆盖。

    返回：每个原子的命中次数、未覆盖清单、覆盖率。
    分母是 atom_universe（active 原子）——reserved 不进分母，
    把够不着的条目放进分母，覆盖率就成了自欺。
    """
    counts: Counter[str] = Counter()
    for c in cases:
        for a in getattr(c, "feature_atoms", []) or []:
            counts[a] += 1
    universe = set(atom_universe)
    unknown = sorted(set(counts) - universe)
    covered = sorted(a for a in universe if counts[a] > 0)
    missing = sorted(universe - set(covered))
    rate = len(covered) / len(universe) if universe else 1.0
    return {
        "counts": dict(counts),
        "covered": covered,
        "missing": missing,
        "unknown_atoms": unknown,
        "rate": rate,
    }


# ---------------- 覆盖引导选择 ----------------


class _CaseSignature(NamedTuple):
    operator: str
    dtype: str
    shape: tuple
    attrs: tuple
    value_range: str
    contiguous: bool


def _case_signature(c) -> tuple:
    """去重签名：内容维度（不含 seed——只差种子的两个用例是近重复）。"""
    return _CaseSignature(
        operator=c.operator, dtype=c.dtype, shape=tuple(c.shape),
        attrs=tuple(sorted((k, str(v)) for k, v in (c.attrs or {}).items())),
        value_range=json.dumps(c.value_range or {"kind": "normal"}, sort_keys=True, separators=(",", ":")),
        contiguous=bool(c.contiguous),
    )


def _cost(c) -> tuple:
    """候选成本：播种阶段优先便宜的（小负载、常规数值、连续、fp32）。"""
    numel = 1
    for d in c.shape:
        numel *= d
    kind = (c.value_range or {}).get("kind", "normal")
    return (
        numel,
        0 if c.contiguous else 1,
        0 if kind == "normal" else 1,
        0 if c.dtype == "float32" else 1,
    )


def guided_selection(
    candidates: list[Any],
    *,
    per_category: int,
    target_total: int | None = None,
    atom_universe: list[str],
    category_of: Any = None,
) -> tuple[list[Any], dict[str, Any]]:
    """从候选池选出题库。

    per_category：每个 category 的用例数目标（quota 保底 + 覆盖引导共用）；
    target_total：可选的全局上限；
    category_of：case → category 的取值函数（默认读 case.category）。

    返回 (选中用例列表, 选择报告)。
    """
    # 去重（近重复只留一个：seed 不同不算差异）
    by_sig: dict[tuple, Any] = {}
    for c in candidates:
        sig = _case_signature(c)
        if sig not in by_sig or _cost(c) < _cost(by_sig[sig]):
            by_sig[sig] = c
    pool = list(by_sig.values())

    def cat(c):
        return category_of(c) if category_of else getattr(c, "category", "")

    selected: list[Any] = []
    sel_sigs: set[tuple] = set()
    covered: set[str] = set()

    def take(c) -> bool:
        sig = _case_signature(c)
        if sig in sel_sigs:
            return False
        sel_sigs.add(sig)
        selected.append(c)
        covered.update(getattr(c, "feature_atoms", []) or [])
        return True

    # ---- A 原子播种：每个未覆盖原子挑一个最廉价的候选 ----
    _seed_atoms(pool, atom_universe, covered, take)
    _seed_diversity(pool, selected, take)

    # ---- C 多样性补齐：按 category 配额轮转，优先新的 (算子, 原子) 组合 ----
    selected_pairs: set[tuple[str, str]] = set()

    def marginal(c) -> tuple:
        atoms = set(c.feature_atoms or [])
        new_pairs = sum(1 for a in atoms if (c.operator, a) not in selected_pairs)
        return (-new_pairs,) + _cost(c)

    per_cat = _fill_category_quotas(
        pool, selected, sel_sigs, take, cat, per_category, target_total, marginal, selected_pairs)

    stats = coverage_stats(selected, atom_universe)
    report = {
        "pool_size": len(pool),
        "selected": len(selected),
        "per_category": dict(per_cat),
        "coverage": stats,
    }
    return selected, report


def _numel(c) -> int:
    n = 1
    for d in c.shape:
        n *= d
    return n


def _seed_atoms(pool, atom_universe, covered, take):
    for atom in sorted(atom_universe):
        if atom in covered:
            continue
        hits = [c for c in pool if atom in (getattr(c, "feature_atoms", []) or [])]
        if not hits:
            continue
        take(min(hits, key=_cost))


def _seed_diversity(pool, selected, take):
    by_op: dict[str, list[Any]] = {}
    for c in pool:
        by_op.setdefault(c.operator, []).append(c)

    for op_cases in by_op.values():
        need: list[tuple] = []
        dts = {c.dtype for c in op_cases}
        scs = {size_class(_numel(c)) for c in op_cases}
        for dt in sorted(dts):
            need.append(("dtype", dt))
        for sc in sorted(scs):
            need.append(("size", sc))
        for key, val in need:
            have = any(
                (c.dtype == val if key == "dtype" else size_class(_numel(c)) == val)
                for c in selected if c.operator == op_cases[0].operator
            )
            if have:
                continue
            hits = [
                c for c in op_cases
                if (c.dtype == val if key == "dtype" else size_class(_numel(c)) == val)
            ]
            if hits:
                take(min(hits, key=_cost))


def _fill_category_quotas(
        pool, selected, sel_sigs, take, cat, per_category, target_total, marginal, selected_pairs):
    per_cat: Counter[str] = Counter()
    for c in selected:
        per_cat[cat(c)] += 1

    cat_order = sorted({cat(c) for c in pool})
    progress = True
    while progress and (target_total is None or len(selected) < target_total):
        progress = False
        for category in cat_order:
            if per_cat[category] >= per_category:
                continue
            remaining = _category_candidates(pool, sel_sigs, cat, category)
            if not remaining:
                continue
            # 边际覆盖最大者优先：给本 category 的算子带来新 (算子, 原子) 组合
            selected_pairs.clear()
            selected_pairs.update(_category_pairs(pool, selected, cat, category))

            best = min(remaining, key=marginal)
            if take(best):
                per_cat[category] += 1
                progress = True
            if target_total is not None and len(selected) >= target_total:
                break

    return per_cat


def _category_candidates(pool, signatures, category_of, category):
    candidates = []
    for case in pool:
        if category_of(case) == category and _case_signature(case) not in signatures:
            candidates.append(case)
    return candidates


def _category_pairs(pool, selected, category_of, category):
    operators = {case.operator for case in pool if category_of(case) == category}
    pairs = set()
    for case in selected:
        if case.operator in operators:
            pairs.update((case.operator, atom) for atom in (case.feature_atoms or []))
    return pairs
