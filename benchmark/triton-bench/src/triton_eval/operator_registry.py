# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""算子注册表（L0，无 NPU 依赖）。

算子规格的来源与选型依据见 `docs/operator-selection.md`。
所有字段取自 `reference/cann-bench/tasks/<level>/<op>/proto.yaml`，
此处固化选型与规格；Python 入口将上游 input 映射为 input_value，避免遮盖内置函数。
"""

from __future__ import annotations

from .official_catalog import additional_specs
from .operatorspec import AttrSpec, IOSpec, OperatorSpec, ShapeSupport
from .taxonomy import Category, Level
from .operator_registry_ascend_ops import ASCEND_OPS  # noqa: E402

FP = ["float16", "bfloat16", "float32"]
FP_INT = ["float16", "bfloat16", "float32", "int8", "int32", "int64"]


def _io(name: str, dtype: list[str]) -> IOSpec:
    return IOSpec(name=name, dtype=dtype)


def _attr(name: str, type_: str, default=None, required: bool = False) -> AttrSpec:
    return AttrSpec(name=name, type=type_, default=default, required=required)


# ---------------- 第 1 批：triton 擅长（7 个 category） ----------------

EXP = OperatorSpec(
    name="Exp", category=Category.ELEMENTWISE, level=Level.L1,
    signature="exp(Tensor x, float base=-1.0, float scale=1.0, float shift=0.0) -> Tensor y",
    inputs=[_io("x", FP)], outputs=[_io("y", FP)],
    attrs=[
        _attr("base", "float", -1.0),
        _attr("scale", "float", 1.0),
        _attr("shift", "float", 0.0),
    ],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="输入输出 shape 一致"),
    source_name="Exp", source_dir="level1/exp",
    entry="exp", params=["x", "base", "scale", "shift"],
)

GELU = OperatorSpec(
    name="Gelu", category=Category.ELEMENTWISE, level=Level.L1,
    signature="gelu(Tensor x, str approximate=\"none\") -> Tensor y",
    inputs=[_io("x", FP)], outputs=[_io("y", FP)],
    attrs=[_attr("approximate", "string", "none")],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="输入输出 shape 一致"),
    source_name="Gelu", source_dir="level1/gelu",
    entry="gelu", params=["x", "approximate"],
)

CUMMIN = OperatorSpec(
    name="Cummin", category=Category.REDUCTION, level=Level.L2,
    signature="cummin(Tensor input_value, int dim) -> (Tensor values, Tensor indices)",
    inputs=[_io("input_value", ["float16", "bfloat16", "float32", "int32"])],
    outputs=[_io("values", ["float16", "bfloat16", "float32", "int32"]),
             _io("indices", ["int64"])],
    attrs=[_attr("dim", "int64", None, required=True)],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="输入输出 shape 相同"),
    source_name="Cummin", source_dir="level2/cummin",
    entry="cummin", params=["input_value", "dim"],
)

MAXIMUM = OperatorSpec(
    name="Maximum", category=Category.BROADCAST, level=Level.L2,
    signature="maximum(Tensor x1, Tensor x2) -> Tensor y",
    inputs=[_io("x1", FP_INT), _io("x2", FP_INT)],
    outputs=[_io("y", FP_INT)],
    attrs=[],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="输入支持广播"),
    source_name="Maximum", source_dir="level2/maximum",
    entry="maximum", params=["x1", "x2"],
)

MASKED_SCALE = OperatorSpec(
    name="MaskedScale", category=Category.MASK_PREDICATE, level=Level.L1,
    signature="masked_scale(Tensor x, Tensor mask, float scale=1.0) -> Tensor y",
    inputs=[_io("x", FP), _io("mask", ["int8", "uint8"] + FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("scale", "float", 1.0)],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="x、mask、y 的 shape 需一致"),
    source_name="MaskedScale", source_dir="level1/masked_scale",
    entry="masked_scale", params=["x", "mask", "scale"],
)

SOFTMAX = OperatorSpec(
    name="Softmax", category=Category.NORMALIZATION, level=Level.L2,
    signature="softmax(Tensor x, int dim=-1) -> Tensor y",
    inputs=[_io("x", FP)], outputs=[_io("y", FP)],
    attrs=[_attr("dim", "int64", -1)],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="输入输出 shape 一致"),
    source_name="Softmax", source_dir="level2/softmax",
    entry="softmax", params=["x", "dim"],
)

RMS_NORM = OperatorSpec(
    name="RmsNorm", category=Category.NORMALIZATION, level=Level.L2,
    signature="rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> Tensor y",
    inputs=[_io("x", FP), _io("gamma", FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("epsilon", "float", 1e-6)],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="gamma shape 与 x 最后一维一致"),
    source_name="RmsNorm", source_dir="level2/rms_norm",
    entry="rms_norm", params=["x", "gamma", "epsilon"],
)

GATHER = OperatorSpec(
    name="Gather", category=Category.INDEX_GATHER, level=Level.L2,
    signature="gather(Tensor x, Tensor index, int dim=0) -> Tensor y",
    inputs=[_io("x", FP_INT), _io("index", ["int8", "int32", "int64"])],
    outputs=[_io("y", FP_INT)],
    attrs=[_attr("dim", "int64", 0)],
    shape_support=ShapeSupport(
        min_rank=1, max_rank=8,
        notes="x 与 index 维度数相同；输出 shape 与 index.shape 一致",
    ),
    source_name="Gather", source_dir="level2/gather",
    entry="gather", params=["x", "index", "dim"],
)

SCATTER = OperatorSpec(
    name="Scatter", category=Category.SCATTER_UPDATE, level=Level.L2,
    signature="scatter(Tensor data, Tensor indices, Tensor updates, int dim, str? reduce=None) -> Tensor y",
    inputs=[_io("data", ["float16", "bfloat16", "float32", "int32", "int64"]),
            _io("indices", ["int32", "int64"]),
            _io("updates", ["float16", "bfloat16", "float32", "int32", "int64"])],
    outputs=[_io("y", ["float16", "bfloat16", "float32", "int32", "int64"])],
    attrs=[_attr("dim", "int64", None, required=True),
           _attr("reduce", "string", None)],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="输入输出 shape 一致"),
    source_name="Scatter", source_dir="level2/scatter",
    # 参数序必须「张量在前、属性在后」——见 operatorspec.validate_params_order。
    # 曾经写成 (data, dim, indices, updates, reduce)：执行器按 make_inputs 顺序
    # 位置传张量、属性走关键字，第 2 个位置实参绑到形参 dim，再遇关键字 dim
    # 必然 TypeError: got multiple values for argument 'dim'。
    # 结果：任何与规格一致的提交都跑不起来，该算子无可行签名（实测 5/5 RUNTIME_FAIL）。
    entry="scatter", params=["data", "indices", "updates", "dim", "reduce"],
)

# ---------------- 第 2 批：可做但需挑算子（5 个 category） ----------------

# Contraction 只选 matmul 类：Conv2D / DepthwiseConv2D / Conv3DBackpropFilter
# 需显式 im2col，triton 极难，不选
GROUPED_MATMUL = OperatorSpec(
    name="GroupedMatmul", category=Category.CONTRACTION, level=Level.L3,
    signature="grouped_matmul(Tensor x, Tensor weight, Tensor? bias, int[] group_list) -> Tensor y",
    inputs=[_io("x", FP), _io("weight", FP), _io("bias", FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("group_list", "int[]", None, required=True)],
    shape_support=ShapeSupport(min_rank=2, max_rank=2, notes="x:[M,K], weight:[E,K,N], y:[M,N]"),
    source_name="GroupedMatmul", source_dir="level3/grouped_matmul",
    entry="grouped_matmul", params=["x", "weight", "bias", "group_list"],
)

TRANSPOSE = OperatorSpec(
    name="Transpose", category=Category.LAYOUT_TRANSFORM, level=Level.L3,
    signature="transpose(Tensor x, int[] perm) -> Tensor y",
    inputs=[_io("x", ["float16", "bfloat16", "float32", "int8", "int16", "int32", "int64"])],
    outputs=[_io("y", ["float16", "bfloat16", "float32", "int8", "int16", "int32", "int64"])],
    attrs=[_attr("perm", "int[]", None, required=True)],
    shape_support=ShapeSupport(min_rank=1, max_rank=8, notes="输入维度 <= 8"),
    source_name="Transpose", source_dir="level3/transpose",
    entry="transpose", params=["x", "perm"],
)

# SortSelect 只选 ArgMax：TopK / Unique / NMS 需多 pass，暂不选
ARG_MAX = OperatorSpec(
    name="ArgMax", category=Category.SORT_SELECT, level=Level.L2,
    signature="arg_max(Tensor input_value, int dim, bool keepdim=false) -> Tensor indices",
    inputs=[_io("input_value", ["float16", "bfloat16", "float32", "int32", "int64"])],
    outputs=[_io("indices", ["int64"])],
    attrs=[_attr("dim", "int64", None, required=True), _attr("keepdim", "bool", False)],
    shape_support=ShapeSupport(min_rank=0, max_rank=8, notes="输入支持 0-8 维"),
    source_name="ArgMax", source_dir="level2/arg_max",
    entry="arg_max", params=["input_value", "dim", "keepdim"],
)

CROSS_ENTROPY_LOSS = OperatorSpec(
    name="CrossEntropyLoss", category=Category.NUMERICAL_STABLE, level=Level.L2,
    signature=("cross_entropy_loss(Tensor input_value, Tensor target, "
               "str reduction=\"mean\", int ignore_index=-100) -> Tensor loss"),
    inputs=[_io("input_value", FP),
            _io("target", ["int32", "int64"] + FP)],
    outputs=[_io("loss", FP)],
    attrs=[_attr("reduction", "string", "mean"), _attr("ignore_index", "int", -100)],
    shape_support=ShapeSupport(
        min_rank=1, max_rank=8,
        notes="input_value:(N,C,...) channel_first；target:(N,...) 硬标签或 (N,C) 软标签",
    ),
    source_name="CrossEntropyLoss", source_dir="level2/cross_entropy_loss",
    entry="cross_entropy_loss", params=["input_value", "target", "reduction", "ignore_index"],
)

# FusedComposite 只选 L1 简单融合：L4 的 MHA / MLA / GRU / LSTM 暂不选
SWI_GLU = OperatorSpec(
    name="SwiGlu", category=Category.FUSED_COMPOSITE, level=Level.L1,
    signature="swi_glu(Tensor input_value, int dim=-1) -> Tensor output",
    inputs=[_io("input_value", FP)], outputs=[_io("output", FP)],
    attrs=[_attr("dim", "int64", -1)],
    shape_support=ShapeSupport(
        min_rank=1, max_rank=8,
        notes="在指定 dim 上拆分成两等份；该 dim 的 size 必须是偶数",
    ),
    source_name="SwiGlu", source_dir="level1/swi_glu",
    entry="swi_glu", params=["input_value", "dim"],
)


# ---------------- 第 3 批：融合算子（算子组合维度，2026-10-13） ----------------
#
# 设计依据 docs/composition-and-pattern-dimensions.md：
#   - Tier-2（7 个）：由已注册原语组合而成，参考实现 = 原语参考的顺序调用，
#     不新写 oracle；source_name/source_dir 留空（无池任务对应，出处是组合表）。
#   - Tier-1（4 个）：直接收录 cann-bench 来源池中已存在的融合任务（字段忠实
#     于其 proto.yaml；与池的差异逐条写在算子注释里）。池中第 5 个融合任务
#     grouped_matmul_swiglu_quant 的主干与 Tier-2 的 GroupedMatmulSwiGlu 同构，
#     量化段待量化批次另行收录，故不重复注册。

# ---- Tier-2：原语组合 ----

LINEAR_GELU = OperatorSpec(
    name="LinearGelu", category=Category.FUSED_COMPOSITE, level=Level.L3,
    signature="linear_gelu(Tensor x, Tensor weight, Tensor bias, int[] group_list) -> Tensor y",
    inputs=[_io("x", FP), _io("weight", FP), _io("bias", FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("group_list", "int[]", None, required=True)],
    shape_support=ShapeSupport(min_rank=2, max_rank=2, notes="x:[M,K]，y:[M,N]；同 GroupedMatmul"),
    entry="linear_gelu", params=["x", "weight", "bias", "group_list"],
)

RMS_NORM_LINEAR = OperatorSpec(
    name="RmsNormLinear", category=Category.FUSED_COMPOSITE, level=Level.L3,
    signature=("rms_norm_linear(Tensor x, Tensor gamma, Tensor weight, Tensor bias, int[] group_list, float "
               "epsilon=1e-6) -> Tensor y"),
    inputs=[_io("x", FP), _io("gamma", FP), _io("weight", FP), _io("bias", FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("group_list", "int[]", None, required=True),
           _attr("epsilon", "float", 1e-6)],
    shape_support=ShapeSupport(min_rank=2, max_rank=2, notes="x:[M,K]，gamma:[K]，y:[M,N]"),
    entry="rms_norm_linear", params=["x", "gamma", "weight", "bias", "group_list", "epsilon"],
)

GROUPED_MATMUL_SWI_GLU = OperatorSpec(
    name="GroupedMatmulSwiGlu", category=Category.FUSED_COMPOSITE, level=Level.L3,
    signature="grouped_matmul_swiglu(Tensor x, Tensor weight, Tensor bias, int[] group_list) -> Tensor y",
    inputs=[_io("x", FP), _io("weight", FP), _io("bias", FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("group_list", "int[]", None, required=True)],
    shape_support=ShapeSupport(min_rank=2, max_rank=2, notes="x:[M,K]，y:[M,N/2]；N 为偶数"),
    entry="grouped_matmul_swiglu", params=["x", "weight", "bias", "group_list"],
)

TRANSPOSE_LINEAR = OperatorSpec(
    name="TransposeLinear", category=Category.FUSED_COMPOSITE, level=Level.L3,
    signature="transpose_linear(Tensor x, Tensor weight, Tensor bias, int[] group_list) -> Tensor y",
    inputs=[_io("x", FP), _io("weight", FP), _io("bias", FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("group_list", "int[]", None, required=True)],
    shape_support=ShapeSupport(
        min_rank=2, max_rank=2,
        notes="x:[M,K] 先转置为 [K,M] 再收缩；group_list 之和必须等于 K（shape[1]）",
    ),
    entry="transpose_linear", params=["x", "weight", "bias", "group_list"],
)

MASKED_SOFTMAX = OperatorSpec(
    name="MaskedSoftmax", category=Category.FUSED_COMPOSITE, level=Level.L2,
    signature="masked_softmax(Tensor x, Tensor mask, int dim) -> Tensor y",
    inputs=[_io("x", FP), _io("mask", ["int8", "uint8"] + FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("dim", "int64", None, required=True)],
    shape_support=ShapeSupport(min_rank=1, max_rank=3, notes="x、mask、y 的 shape 一致；掩码为 0 处不参与归一化"),
    entry="masked_softmax", params=["x", "mask", "dim"],
)

MASKED_GELU = OperatorSpec(
    name="MaskedGelu", category=Category.FUSED_COMPOSITE, level=Level.L2,
    signature="masked_gelu(Tensor x, Tensor mask, str approximate=\"none\") -> Tensor y",
    inputs=[_io("x", FP), _io("mask", ["int8", "uint8"] + FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("approximate", "string", "none")],
    shape_support=ShapeSupport(min_rank=1, max_rank=3, notes="x、mask、y 的 shape 一致；掩码为 0 处输出置零"),
    entry="masked_gelu", params=["x", "mask", "approximate"],
)

GATHER_EXP = OperatorSpec(
    name="GatherExp", category=Category.FUSED_COMPOSITE, level=Level.L2,
    signature=("gather_exp(Tensor x, Tensor index, int dim=0, float base=-1.0, float scale=1.0, float shift=0.0) -> "
               "Tensor y"),
    # x 只收浮点：对整数做 exp 再转回整数会饱和截断（exp(99) 远超 int32），
    # 判定语义病态——整数用例的 Exp 段没有可判定的正确答案
    inputs=[_io("x", FP), _io("index", ["int8", "int32", "int64"])],
    outputs=[_io("y", FP)],
    attrs=[_attr("dim", "int64", 0), _attr("base", "float", -1.0),
           _attr("scale", "float", 1.0), _attr("shift", "float", 0.0)],
    shape_support=ShapeSupport(min_rank=1, max_rank=3, notes="x 与 index 维度数相同；输出 shape 与 index 一致"),
    entry="gather_exp", params=["x", "index", "dim", "base", "scale", "shift"],
)

# ---- Tier-1：来源池直收（字段忠实于池 proto.yaml，差异逐条注明） ----

# 池：tasks/level3/add_rms_norm_dynamic_quant，公式 quantize(rmsnorm(x1+x2)*gamma)。
# 收录主干 add + rms_norm；dynamic_quant 段（y, xOut, scaleOut 三输出中的量化
# 两路）随量化批次另行收录。池 dtype fp16/bf16，此处加 fp32（精度判定最稳的
# 基准档，与既有 FP 口径一致）。
ADD_RMS_NORM = OperatorSpec(
    name="AddRmsNorm", category=Category.FUSED_COMPOSITE, level=Level.L3,
    signature="add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-6) -> Tensor y",
    inputs=[_io("x1", FP), _io("x2", FP), _io("gamma", FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("epsilon", "float", 1e-6)],
    shape_support=ShapeSupport(min_rank=1, max_rank=3, notes="x1、x2 同形；gamma 与最后一维一致"),
    source_name="AddRmsNormDynamicQuant", source_dir="level3/add_rms_norm_dynamic_quant",
    entry="add_rms_norm", params=["x1", "x2", "gamma", "epsilon"],
)

# 池：tasks/level2/apply_rotary_pos_emb，公式 y = x*cos + rotate_half(x)*sin，
# cos/sin 是**输入**（池 proto 明确其为输入张量，非算子内查表）。
# 池的 layout 属性只影响输入排布约定，不影响末维旋转的数学，故不收录；
# rotary_mode 保留（half=连续半分式，interleaved=交错式）。
APPLY_ROTARY_POS_EMB = OperatorSpec(
    name="ApplyRotaryPosEmb", category=Category.FUSED_COMPOSITE, level=Level.L2,
    signature="apply_rotary_pos_emb(Tensor x, Tensor cos, Tensor sin, str rotary_mode=\"half\") -> Tensor y",
    inputs=[_io("x", FP), _io("cos", FP), _io("sin", FP)],
    outputs=[_io("y", FP)],
    attrs=[_attr("rotary_mode", "string", "half")],
    shape_support=ShapeSupport(
        min_rank=2, max_rank=3,
        notes="x 最后一维 head_dim 必须为偶数；cos/sin 与 x 同形或可广播至 x",
    ),
    source_name="ApplyRotaryPosEmb", source_dir="level2/apply_rotary_pos_emb",
    entry="apply_rotary_pos_emb", params=["x", "cos", "sin", "rotary_mode"],
)

# 池：tasks/level3/moe_gating_top_k_softmax，公式
#   softmaxOut = softmax(x, -1); yOut, expertIdx = topK(softmaxOut, k); rowIdx = 行号重排。
# 池的可选输入 finished（布尔掩码）不收录（proto 标 optional）；三输出中
# 当前判定契约只比对第一个张量（同 Cummin 的已披露缺口）。
# dtype 不含 fp16：softmax 概率在 fp16 下易量化出并列值，topk 选中下标在
# 实现与参考间会合法地分歧——fp32/bf16 下并列概率可忽略。
# 池 category 为 LayoutTransform，本注册表按语义归入 FusedComposite（差异已记录）。
MOE_GATING_TOP_K_SOFTMAX = OperatorSpec(
    name="MoeGatingTopKSoftmax", category=Category.FUSED_COMPOSITE, level=Level.L3,
    signature="moe_gating_top_k_softmax(Tensor x, int k) -> (Tensor values, Tensor expert_indices, Tensor row_indices)",
    inputs=[_io("x", ["float32", "bfloat16"])],
    outputs=[_io("values", ["float32", "bfloat16"]),
             _io("expert_indices", ["int64"]),
             _io("row_indices", ["int64"])],
    attrs=[_attr("k", "int64", None, required=True)],
    shape_support=ShapeSupport(min_rank=2, max_rank=3, notes="x:(..., E) E 为专家数；0 < k <= E 且 k <= 1024"),
    source_name="MoeGatingTopKSoftmax", source_dir="level3/moe_gating_top_k_softmax",
    entry="moe_gating_top_k_softmax", params=["x", "k"],
)

# 池：tasks/level2/unsorted_segment_sum，公式 y[i] = sum(data[j] where ids[j]==i)。
# 池自带 f32 覆盖档 0.001（NPU 算子精度与 CPU 有差异）——本体系容差走
# tolerance.yaml 的算子级覆盖机制，不在注册表放宽。
UNSORTED_SEGMENT_SUM = OperatorSpec(
    name="UnsortedSegmentSum", category=Category.SCATTER_UPDATE, level=Level.L2,
    signature="unsorted_segment_sum(Tensor data, Tensor segment_ids, int num_segments) -> Tensor y",
    inputs=[_io("data", ["float16", "bfloat16", "float32", "int32", "int64"]),
            _io("segment_ids", ["int32", "int64"])],
    outputs=[_io("y", ["float16", "bfloat16", "float32", "int32", "int64"])],
    attrs=[_attr("num_segments", "int64", None, required=True)],
    shape_support=ShapeSupport(min_rank=1, max_rank=2, notes="ids 长度 = data 第 0 维；y:[num_segments, ...]"),
    source_name="UnsortedSegmentSum", source_dir="level2/unsorted_segment_sum",
    entry="unsorted_segment_sum", params=["data", "segment_ids", "num_segments"],
)


ALL_OPERATORS: list[OperatorSpec] = [
    # 第 1 批
    EXP, GELU, CUMMIN, MAXIMUM, MASKED_SCALE, SOFTMAX, RMS_NORM, GATHER, SCATTER,
    # 第 2 批
    GROUPED_MATMUL, TRANSPOSE, ARG_MAX, CROSS_ENTROPY_LOSS, SWI_GLU,
    # 第 3 批：融合算子
    LINEAR_GELU, RMS_NORM_LINEAR, GROUPED_MATMUL_SWI_GLU, TRANSPOSE_LINEAR,
    MASKED_SOFTMAX, MASKED_GELU, GATHER_EXP,
    ADD_RMS_NORM, APPLY_ROTARY_POS_EMB, MOE_GATING_TOP_K_SOFTMAX,
    UNSORTED_SEGMENT_SUM,
    # ── Ascend 官方教程算子（triton-ascend-ops 样例仓）── 仅进 v0.6 语料，不进 v0.5 定稿集
    *ASCEND_OPS,
]

ALL_OPERATORS.extend(additional_specs(ALL_OPERATORS))

# 导入即校验参数序：让「规格写错」在进程启动时就暴露，
# 而不是等某份提交照着它写实现、跑不通才发现。
for _spec in ALL_OPERATORS:
    _spec.validate_param_order()


# ---------------- 组合表（算子组合维度的规格侧声明） ----------------
#
# 每个融合算子声明其组成**段落**：段落是语义段（Tier-2 引用已注册原语，
# Tier-1 引用池任务定义的语义段）。标签器（coverage.label_case）从这张表
# 纯函数地推导融合原子——不看任何实现，不依赖提交方行为。


class _Stage:
    """组合表的一个段落：语义段名 + 语义族（标签推导的输入）。"""

    __slots__ = ("name", "category")

    def __init__(self, name: str, category: Category):
        self.name = name
        self.category = category


COMPOSITIONS: dict[str, list[_Stage]] = {
    # ---- Tier-2：段落 = 已注册原语 ----
    "LinearGelu": [_Stage("GroupedMatmul", Category.CONTRACTION),
                   _Stage("Gelu", Category.ELEMENTWISE)],
    "RmsNormLinear": [_Stage("RmsNorm", Category.NORMALIZATION),
                      _Stage("GroupedMatmul", Category.CONTRACTION)],
    "GroupedMatmulSwiGlu": [_Stage("GroupedMatmul", Category.CONTRACTION),
                            _Stage("SwiGlu", Category.FUSED_COMPOSITE)],
    "TransposeLinear": [_Stage("Transpose", Category.LAYOUT_TRANSFORM),
                        _Stage("GroupedMatmul", Category.CONTRACTION)],
    "MaskedSoftmax": [_Stage("MaskedScale", Category.MASK_PREDICATE),
                      _Stage("Softmax", Category.NORMALIZATION)],
    "MaskedGelu": [_Stage("MaskedScale", Category.MASK_PREDICATE),
                   _Stage("Gelu", Category.ELEMENTWISE)],
    "GatherExp": [_Stage("Gather", Category.INDEX_GATHER),
                  _Stage("Exp", Category.ELEMENTWISE)],
    # ---- Tier-1：段落 = 池任务定义的语义段 ----
    "AddRmsNorm": [_Stage("Add", Category.ELEMENTWISE),
                   _Stage("RmsNorm", Category.NORMALIZATION)],
    "ApplyRotaryPosEmb": [_Stage("SinCos", Category.ELEMENTWISE),
                          _Stage("RotateHalf", Category.ELEMENTWISE)],
    "MoeGatingTopKSoftmax": [_Stage("Softmax", Category.NORMALIZATION),
                             _Stage("TopK", Category.SORT_SELECT)],
    "UnsortedSegmentSum": [_Stage("Scatter", Category.SCATTER_UPDATE),
                           _Stage("SegmentSum", Category.REDUCTION)],
}


def composition_of(operator: str) -> list[_Stage] | None:
    """融合算子的组成段落；非融合算子返回 None。"""
    return COMPOSITIONS.get(operator)


def fusion_operators() -> list[str]:
    """全部融合算子名（组合表的键，保持注册顺序）。"""
    return [op.name for op in ALL_OPERATORS if op.name in COMPOSITIONS]


def by_name(name: str) -> OperatorSpec | None:
    for op in ALL_OPERATORS:
        if op.name == name:
            return op
    return None


def by_category(category: str) -> list[OperatorSpec]:
    return [op for op in ALL_OPERATORS if op.category.value == category]


def by_level(level: str) -> list[OperatorSpec]:
    return [op for op in ALL_OPERATORS if op.level.value == level]


def names() -> list[str]:
    return [op.name for op in ALL_OPERATORS]
