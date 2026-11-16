# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Official-source admission catalogue; historical registrations are not qualifications."""

from __future__ import annotations

from dataclasses import dataclass

from .operatorspec import AttrSpec, IOSpec, OperatorSpec, ShapeSupport
from .taxonomy import Category, Level


SOURCE_REPOSITORY = "https://github.com/triton-lang/triton-ascend"
SOURCE_COMMIT = "5f1d99a87432c09815b6ded57624c04c00bbe0aa"
SOURCE_TESTS = "third_party/ascend/unittest/pytest_ut/"
VERSION = "v0.9-official-candidates"

ELEMENTWISE_FAMILIES = frozenset(("unary", "binary", "comparison", "logical",
                                  "bitwise", "bitwise_unary", "where"))
COMPILER_ONLY = frozenset({"Exp2", "Log2", "Log10", "XorSum"})


def submission_entry_name(operator, default):
    """Choose a submission entry without changing registry or reference paths."""
    return {"Abs": "absolute", "Pow": "power"}.get(operator, default)


@dataclass(frozen=True)
class Candidate:
    name: str
    entry: str
    family: str
    source: str
    expression: str
    category: Category = Category.ELEMENTWISE
    dtypes: tuple[str, ...] = ("float32", "float16")


def _candidate(name, entry, family, filename, expression, category=Category.ELEMENTWISE,
               dtypes=("float32", "float16")):
    return Candidate(name, entry, family, SOURCE_TESTS + filename, expression, category, dtypes)


CANDIDATES = (
    _candidate("Abs", "abs", "unary", "test_abs.py", "abs(x)"),
    _candidate("Neg", "neg", "unary", "test_neg.py", "-x"),
    _candidate(
        "Exp",
        "exp",
        "unary",
        "test_exp.py",
        "exp(x * scale + shift) if base == -1 else base ** (x * scale + shift)"),
    _candidate("Exp2", "exp2", "unary", "test_exp2.py", "2 ** x"),
    _candidate("Expm1", "expm1", "unary", "test_expm1.py", "exp(x) - 1"),
    _candidate("Log", "log", "unary", "test_log.py", "ln(x), x > 0"),
    _candidate("Log2", "log2", "unary", "test_log2.py", "log2(x), x > 0"),
    _candidate("Log10", "log10", "unary", "test_log10.py", "log10(x), x > 0"),
    Candidate("Log1p", "log1p", "unary", "docs/zh/python-api/_examples/triton.language.extra.cann.libdevice.log1p.py",
              "ln(1 + x), x > -1"),
    _candidate("Sqrt", "sqrt", "unary", "test_sqrt.py", "sqrt(x), x >= 0"),
    _candidate("Rsqrt", "rsqrt", "unary", "test_rsqrt.py", "1 / sqrt(x), x > 0"),
    _candidate("Sin", "sin", "unary", "test_sin.py", "sin(x)"),
    _candidate("Cos", "cos", "unary", "test_cos.py", "cos(x)"),
    _candidate("Tan", "tan", "unary", "test_tan.py", "tan(x), excluding poles"),
    _candidate("Tanh", "tanh", "unary", "test_tanh.py", "tanh(x)"),
    _candidate("Sinh", "sinh", "unary", "test_sinh.py", "sinh(x)"),
    _candidate("Cosh", "cosh", "unary", "test_cosh.py", "cosh(x)"),
    _candidate("Floor", "floor", "unary", "test_floor.py", "floor(x)"),
    _candidate("Ceil", "ceil", "unary", "test_ceil.py", "ceil(x)"),
    _candidate("Trunc", "trunc", "unary", "test_trunc.py", "truncate toward zero"),
    _candidate("Relu", "relu", "unary", "test_relu.py", "max(x, 0)"),
    _candidate("Sigmoid", "sigmoid", "unary", "test_sigmoid.py", "1 / (1 + exp(-x))"),
    _candidate(
        "Signbit",
        "signbit",
        "unary",
        "test_signbit.py",
        "IEEE sign bit, including negative zero",
        Category.MASK_PREDICATE),
    _candidate("Add", "add", "binary", "test_add.py", "x + y"),
    _candidate("Sub", "sub", "binary", "test_sub.py", "x - y"),
    _candidate("Mul", "mul", "binary", "test_mul.py", "x * y"),
    _candidate("Div", "div", "binary", "test_div.py", "x / y, y != 0"),
    _candidate(
        "FloorDiv",
        "floor_div",
        "binary",
        "test_floordiv.py",
        "floor(x / y), signed operands",
        dtypes=(
            "int32",
        )),
    _candidate("Mod", "mod", "binary", "test_mod.py", "x - trunc(x / y) * y"),
    _candidate("Pow", "pow", "binary", "test_pow.py", "x ** y, x > 0"),
    _candidate("Maximum", "maximum", "binary", "test_maximum.py", "elementwise maximum"),
    _candidate("Minimum", "minimum", "binary", "test_minimum.py", "elementwise minimum"),
    _candidate("Equal", "equal", "comparison", "test_eq.py", "x == y", Category.MASK_PREDICATE),
    _candidate("NotEqual", "not_equal", "comparison", "test_ne.py", "x != y", Category.MASK_PREDICATE),
    _candidate("Greater", "greater", "comparison", "test_gt.py", "x > y", Category.MASK_PREDICATE),
    _candidate("GreaterEqual", "greater_equal", "comparison", "test_ge.py", "x >= y", Category.MASK_PREDICATE),
    _candidate("Less", "less", "comparison", "test_lt.py", "x < y", Category.MASK_PREDICATE),
    _candidate("LessEqual", "less_equal", "comparison", "test_le.py", "x <= y", Category.MASK_PREDICATE),
    _candidate("BitwiseAnd", "bitwise_and", "bitwise", "test_and.py", "x & y", dtypes=("int32",)),
    _candidate("BitwiseOr", "bitwise_or", "bitwise", "test_or.py", "x | y", dtypes=("int32",)),
    _candidate("BitwiseXor", "bitwise_xor", "bitwise", "test_xor.py", "x ^ y", dtypes=("int32",)),
    _candidate("BitwiseNot", "bitwise_not", "bitwise_unary", "test_invert.py", "~x", dtypes=("int32",)),
    _candidate(
        "LogicalAnd",
        "logical_and",
        "logical",
        "test_logical_and.py",
        "bool(x) and bool(y)",
        Category.MASK_PREDICATE),
    _candidate(
        "LogicalOr",
        "logical_or",
        "logical",
        "test_logical_or.py",
        "bool(x) or bool(y)",
        Category.MASK_PREDICATE),
    _candidate("Clamp", "clamp", "unary", "test_clamp.py", "min(max(x, lower), upper)"),
    _candidate("Where", "where", "where", "test_where_mask.py", "condition ? x : y", Category.MASK_PREDICATE),
    _candidate("ReduceSum", "reduce_sum", "reduce", "test_sum_dim1.py", "sum over last axis", Category.REDUCTION),
    _candidate("ReduceMean", "reduce_mean", "reduce", "test_mean_dim1.py", "mean over last axis", Category.REDUCTION),
    _candidate("ReduceMax", "reduce_max", "reduce", "test_max_dim1.py", "maximum over last axis", Category.REDUCTION),
    _candidate("ReduceMin", "reduce_min", "reduce", "test_min_dim1.py", "minimum over last axis", Category.REDUCTION),
    _candidate(
        "ArgMax",
        "arg_max",
        "arg",
        "test_argmax.py",
        "first index of maximum, including ties",
        Category.SORT_SELECT),
    _candidate(
        "ArgMin",
        "arg_min",
        "arg",
        "test_argmin.py",
        "first index of minimum, including ties",
        Category.SORT_SELECT),
    _candidate("XorSum", "xor_sum", "reduce", "test_xor_sum.py",
               "bitwise XOR reduction", Category.REDUCTION, ("int32",)),
    _candidate("CumSum", "cum_sum", "scan", "test_cumsum.py", "inclusive sum over last axis", Category.REDUCTION),
    _candidate(
        "CumProd",
        "cum_prod",
        "scan",
        "test_cumprod.py",
        "inclusive product over last axis",
        Category.REDUCTION),
    _candidate(
        "Transpose",
        "transpose",
        "transpose",
        "test_permute.py",
        "2D axes permutation [1, 0]",
        Category.LAYOUT_TRANSFORM),
    _candidate(
        "Concat",
        "concat",
        "concat",
        "test_cat_dim.py",
        "concatenate along last axis",
        Category.LAYOUT_TRANSFORM),
    _candidate("Flip", "flip", "flip", "test_flip.py", "reverse last axis", Category.LAYOUT_TRANSFORM),
    Candidate("Gather", "gather", "gather",
              "docs/zh/python-api/_examples/triton.language.gather.py",
              "2D gather along last axis; repeated indices allowed", Category.INDEX_GATHER),
    _candidate(
        "SortValues",
        "sort_values",
        "sort",
        "test_sort.py",
        "sort values along last axis, ascending or descending",
        Category.SORT_SELECT),
    _candidate(
        "TopKValues",
        "top_k_values",
        "topk",
        "test_topk.py",
        "largest k values, descending; value output only",
        Category.SORT_SELECT),
    _candidate("Softmax", "softmax", "softmax", "test_02_fused_softmax.py",
               "exp(x - max(x)) / sum(exp(x - max(x)))", Category.NORMALIZATION),
    _candidate(
        "RmsNorm",
        "rms_norm",
        "rms",
        "test_rms_norm.py",
        "x * rsqrt(mean(x*x) + epsilon) * gamma",
        Category.NORMALIZATION),
    _candidate(
        "LayerNorm",
        "layer_norm",
        "layer",
        "test_layernorm.py",
        "(x - mean(x)) * rsqrt(var(x) + epsilon) * weight + bias",
        Category.NORMALIZATION),
    _candidate("Matmul", "matmul", "matmul", "test_03_matrix_multiplication.py",
               "x[M,K] @ weight[K,N]", Category.CONTRACTION, ("float16",)),
    Candidate("Mish", "mish", "unary", "reference/cann-bench/tasks/level1/mish/proto.yaml",
              "x * tanh(softplus(x))"),
    _candidate(
        "Gelu",
        "gelu",
        "unary",
        "test_gelu.py",
        "0.5*x*(1+erf(x/sqrt(2))) or explicitly requested tanh approximation"),
    _candidate(
        "SwiGlu",
        "swi_glu",
        "swiglu",
        "test_swiglu.py",
        "silu(first half) * second half",
        Category.FUSED_COMPOSITE),
)

BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}

SEMANTIC_ATOMS = {
    "Add": ["FA-MATH-ADD", "FA-MULTI-INPUT"],
    "Mul": ["FA-MATH-MUL", "FA-MULTI-INPUT"],
    "Exp2": ["FA-MATH-EXP"], "Expm1": ["FA-MATH-EXP"],
    "Sqrt": ["FA-MATH-SQRT"], "Rsqrt": ["FA-MATH-SQRT"],
    "Sin": ["FA-MATH-SINCOS"], "Cos": ["FA-MATH-SINCOS"],
    "Sigmoid": ["FA-MATH-SIGMOID"],
    "Minimum": ["FA-MATH-MAXMIN", "FA-MULTI-INPUT"],
    "Clamp": ["FA-MATH-MAXMIN"], "Relu": ["FA-MATH-MAXMIN"],
    "Where": ["FA-IDX-WHERE", "FA-MULTI-INPUT"],
    "ReduceSum": ["FA-REDUCE-SUM"], "ReduceMean": ["FA-REDUCE-SUM"],
    "ReduceMax": ["FA-REDUCE-MAX"], "ReduceMin": ["FA-REDUCE-MIN"],
    "ArgMin": ["FA-REDUCE-MIN", "FA-REDUCE-ARG"],
    "CumSum": ["FA-REDUCE-SUM"], "CumProd": ["FA-MATH-MUL"],
    "LayerNorm": ["FA-REDUCE-SUM", "FA-MATH-SQRT", "FA-MATH-ADD", "FA-MATH-MUL", "FA-MULTI-INPUT"],
    "Matmul": ["FA-LINALG-DOT", "FA-MULTI-INPUT"],
}


def specification(candidate):
    floating = list(candidate.dtypes)
    family = candidate.family
    first_name = "input" if candidate.name in ("ArgMax", "SwiGlu") else "x1" if candidate.name == "Maximum" else "x"
    inputs = _specification_inputs(candidate, family, floating, first_name)

    attrs = _specification_attrs(candidate, family)

    boolean = family in ("comparison", "logical") or candidate.name == "Signbit"
    output_dtype = ["bool"] if boolean else ["int64"] if family == "arg" else floating
    return OperatorSpec(
        name=candidate.name, category=candidate.category,
        level=Level.L3 if family == "matmul" else Level.L2 if family not in
        ("unary", "binary", "comparison", "logical", "bitwise", "bitwise_unary", "where") else Level.L1,
        signature=f"{candidate.entry}({', '.join(item.name for item in inputs)}) -> y",
        inputs=inputs, outputs=[IOSpec(name="y", dtype=output_dtype)], attrs=attrs,
        case_dtypes=list(candidate.dtypes),
        shape_support=ShapeSupport(
            min_rank=1 if family in ELEMENTWISE_FAMILIES else 2,
            max_rank=4 if family in ELEMENTWISE_FAMILIES else 2,
            notes=("v0.9 candidate domain: rank 1–4 elementwise; main input contiguous or "
                   "uniform flat stride=2; auxiliary tensors contiguous; equal operand shapes"
                   if family in ELEMENTWISE_FAMILIES else
                   "v0.9 candidate domain: contiguous nonempty 2D; last-axis operations")),
        source_name=candidate.name, source_dir=candidate.source,
        entry=candidate.entry, params=[item.name for item in inputs] + [item.name for item in attrs])


def additional_specs(existing):
    registered = {spec.name for spec in existing}
    return [specification(candidate) for candidate in CANDIDATES if candidate.name not in registered]


def _specification_inputs(candidate, family, floating, first_name):
    inputs = [IOSpec(name=first_name, dtype=floating)]
    if family in ("binary", "comparison", "bitwise", "logical", "concat"):
        inputs.append(IOSpec(name="x2" if candidate.name == "Maximum" else "y", dtype=floating))
    if family == "where":
        inputs = [IOSpec(name="condition", dtype=["bool"]),
                  IOSpec(name="x", dtype=floating), IOSpec(name="y", dtype=floating)]
    if family == "gather":
        inputs.append(IOSpec(name="index", dtype=["int64"]))
    if family in ("rms", "layer"):
        inputs.append(IOSpec(name="gamma" if family == "rms" else "weight", dtype=floating))
    if family == "layer":
        inputs.append(IOSpec(name="bias", dtype=floating))
    if family == "matmul":
        inputs.append(IOSpec(name="weight", dtype=floating))
    return inputs


def _specification_attrs(candidate, family):
    if candidate.name == "Exp":
        return [AttrSpec(name="base", type="float", default=-1.0),
                AttrSpec(name="scale", type="float", default=1.0),
                AttrSpec(name="shift", type="float", default=0.0)]
    if candidate.name == "Gelu":
        return [AttrSpec(name="approximate", type="string", default="none")]
    if candidate.name == "Clamp":
        return [AttrSpec(name="lower", type="float", default=-0.5),
                AttrSpec(name="upper", type="float", default=0.5)]
    if family in ("reduce", "arg", "scan", "softmax", "gather", "swiglu", "concat", "flip", "sort", "topk"):
        attrs = [AttrSpec(name="dim", type="int64", default=-1)]
        if family in ("reduce", "arg"):
            attrs.append(AttrSpec(name="keepdim", type="bool", default=False))
        if family == "sort":
            attrs.append(AttrSpec(name="descending", type="bool", default=False))
        if family == "topk":
            attrs.append(AttrSpec(name="k", type="int64", default=1))
        return attrs
    if family == "transpose":
        return [AttrSpec(name="perm", type="int[]", default=[1, 0])]
    if family in ("layer", "rms"):
        return [AttrSpec(name="epsilon", type="float", default=1e-6)]
    return []
