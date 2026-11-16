# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""triton-ascend-ops 官方教程算子 → 本评测 OperatorSpec 映射。

来源：reference/triton-ascend-ops（Ascend 官方样例仓，15 个示例）。
本模块映射 11 个已接入项目。其余四项尚未接入，不等于没有数学语义：
ub_overflow 是 KV 槽位分配，tiling 是 Gather，load_order 有状态更新，
load_care_padding 是访存 API 的文档说明。逐项处置见官方来源核查文档。

设计约束（与 triton_eval 执行器契约一致，见 runner.py / harness/npu_exec.py /
inputs.py）：
  * params：张量输入在前，属性走 **kwargs（validate_param_order 硬约束）。
  * 输出缓冲（out / Mean / Rstd / conv_state 等）一律在 entry / reference
    内部分配，绝不进入 make_inputs 的「输入」列表。
  * 多输出：reference / entry 返回 tuple，as_outputs 逐张量 allclose。
  * 来源教程保留；示范适配实现使用独立提交身份。

示范内核需在真实 NPU 上编译执行；部分内核依赖 CANN 扩展。
v0.7 的 torch 主参考与独立 CPU 第二参考均可脱离 Triton 执行。

分级（参考 CANN Bench 的 L1-L4，按「内核复杂度 × 输入约束 × 多输出」判定）：
  L1：单 kernel、单/双输入、无约束            → VectorAdd
  L2：含 NPU 专属语义或布局变换、2 输入       → DiscreteMemoryAccess /
       InsertSlice / ExtractSlice / VectorCmp
  L3：MoE 重排 / 注意力 / 卷积状态更新、多输入多输出 → GatherScatter 系列 /
       CausalConv1d / DecodeGroupedAttention / AssignReqToTokenPool
"""
from __future__ import annotations

from .operatorspec import AttrSpec, IOSpec, OperatorSpec, ShapeSupport
from .taxonomy import Category, Level

ASCEND_OPS: list[OperatorSpec] = [
    # ───────────────────────── 基础优化（basic） ─────────────────────────
    OperatorSpec(
        name="VectorAdd", category=Category.ELEMENTWISE, level=Level.L1,
        signature="vector_add(Tensor x, Tensor y) -> Tensor z",
        inputs=[IOSpec(name="x", dtype=["float32", "int32", "int64"]),
                IOSpec(name="y", dtype=["float32", "int32", "int64"])],
        outputs=[IOSpec(name="z", dtype=["float32", "int32", "int64"])],
        shape_support=ShapeSupport(min_rank=1, max_rank=1),
        source_name="vector_add", source_dir="tutorial/basic/001-vector_add",
        entry="vector_add", params=["x", "y"],
    ),
    OperatorSpec(
        name="VectorCmp", category=Category.NORMALIZATION, level=Level.L2,
        signature="vector_cmp(Tensor x) -> (Tensor out, Tensor mean, Tensor rstd)",
        inputs=[IOSpec(name="x", dtype=["float32"])],
        outputs=[IOSpec(name="out", dtype=["float32"]),
                 IOSpec(name="mean", dtype=["float32"]),
                 IOSpec(name="rstd", dtype=["float32"])],
        attrs=[AttrSpec(name="eps", type="float", default=1e-6)],
        shape_support=ShapeSupport(min_rank=2, max_rank=2,
                                   notes="(rows, cols)；cols 需为 2 的幂（kernel BLOCK_N）"),
        source_name="vector_cmp", source_dir="tutorial/basic/002-vector_cmp",
        entry="vector_cmp", params=["x"],
    ),
    OperatorSpec(
        name="DiscreteMemoryAccess", category=Category.INDEX_GATHER, level=Level.L2,
        signature="discrete_pick(Tensor x, Tensor indices) -> Tensor y",
        inputs=[IOSpec(name="x", dtype=["float32", "int32"]),
                IOSpec(name="indices", dtype=["int32"])],
        outputs=[IOSpec(name="y", dtype=["float32", "int32"])],
        attrs=[AttrSpec(name="m", type="int", required=True),
               AttrSpec(name="n", type="int", required=True)],
        shape_support=ShapeSupport(min_rank=1, max_rank=1),
        source_name="discrete_memory_access",
        source_dir="tutorial/basic/004-discrete_memory_access",
        # 索引参数统一叫 indices：与 InsertSlice / ExtractSlice / GatherScatter
        # 一致，也与提交包 kernel.py 和语料 ref.py 的签名一致。此前这里写成
        # idx，导致真机加载时报 SUB-LOD-007「期望 ['x','idx']，实际
        # ['x','indices']」，提交校验不通过、整轮 0 用例。
        entry="discrete_pick", params=["x", "indices"],
    ),

    # ───────────────────────── 扩展语义（op_extension） ─────────────────────────
    OperatorSpec(
        name="InsertSlice", category=Category.LAYOUT_TRANSFORM, level=Level.L2,
        signature="token_rearrange(Tensor x, Tensor indices) -> Tensor out",
        inputs=[IOSpec(name="x", dtype=["float32"]),
                IOSpec(name="indices", dtype=["int32"])],
        outputs=[IOSpec(name="out", dtype=["float32"])],
        shape_support=ShapeSupport(min_rank=2, max_rank=2,
                                   notes="(S, D)；indices 为 0..S-1 的置换"),
        source_name="insert_slice", source_dir="tutorial/op_extension/001-insert_slice",
        entry="token_rearrange", params=["x", "indices"],
    ),
    OperatorSpec(
        name="ExtractSlice", category=Category.LAYOUT_TRANSFORM, level=Level.L2,
        signature="token_reverse(Tensor x, Tensor indices) -> Tensor out",
        inputs=[IOSpec(name="x", dtype=["float32"]),
                IOSpec(name="indices", dtype=["int32"])],
        outputs=[IOSpec(name="out", dtype=["float32"])],
        shape_support=ShapeSupport(min_rank=2, max_rank=2,
                                   notes="(S, D)；indices 为 0..S-1 的置换"),
        source_name="extract_slice", source_dir="tutorial/op_extension/002-extract_slice",
        entry="token_reverse", params=["x", "indices"],
    ),

    # ───────────────────────── 完整算子优化（best_practice） ─────────────────────────
    OperatorSpec(
        name="AssignReqToTokenPool", category=Category.SCATTER_UPDATE, level=Level.L3,
        signature=("assign_req_to_token_pool(Tensor req_pool_indices, Tensor req_to_token, "
                   "Tensor start_offset, Tensor end_offset, Tensor out_cache_loc) "
                   "-> Tensor req_to_token"),
        inputs=[IOSpec(name="req_pool_indices", dtype=["int32"]),
                IOSpec(name="req_to_token", dtype=["int32"]),
                IOSpec(name="start_offset", dtype=["int32"]),
                IOSpec(name="end_offset", dtype=["int32"]),
                IOSpec(name="out_cache_loc", dtype=["int32"])],
        outputs=[IOSpec(name="req_to_token", dtype=["int32"])],
        attrs=[AttrSpec(name="pool_len", type="int", required=True),
               AttrSpec(name="bs_upper", type="int", required=True)],
        shape_support=ShapeSupport(min_rank=1, max_rank=1),
        source_name="assign_req_to_token_pool",
        source_dir="tutorial/best_practice/001-assign_req_to_token_pool",
        entry="assign_req_to_token_pool", params=["req_pool_indices", "req_to_token",
                                                  "start_offset", "end_offset", "out_cache_loc"],
    ),
    OperatorSpec(
        name="DecodeGroupedAttention", category=Category.CONTRACTION, level=Level.L3,
        signature=("decode_grouped_attention(Tensor q, Tensor k_buffer, Tensor v_buffer, "
                   "Tensor kv_indptr, Tensor kv_indices, Tensor num_kv_splits) "
                   "-> (Tensor att_out, Tensor att_lse)"),
        inputs=[IOSpec(name="q", dtype=["bfloat16"]),
                IOSpec(name="k_buffer", dtype=["bfloat16"]),
                IOSpec(name="v_buffer", dtype=["bfloat16"]),
                IOSpec(name="kv_indptr", dtype=["int32"]),
                IOSpec(name="kv_indices", dtype=["int32"]),
                IOSpec(name="num_kv_splits", dtype=["int32"])],
        outputs=[IOSpec(name="att_out", dtype=["float32"]),
                 IOSpec(name="att_lse", dtype=["float32"])],
        attrs=[AttrSpec(name="max_kv_splits", type="int", default=4),
               AttrSpec(name="sm_scale", type="float", default=0.0)],
        shape_support=ShapeSupport(min_rank=3, max_rank=3,
                                   notes="q:(B,Hq,D) k/v_buffer:(B*S,Hkv,Dv)"),
        source_name="decode_grouped_attention",
        source_dir="tutorial/best_practice/002-decode_grouped_attention",
        entry="decode_grouped_attention", params=["q", "k_buffer", "v_buffer",
                                                  "kv_indptr", "kv_indices", "num_kv_splits"],
    ),
    OperatorSpec(
        name="CausalConv1d", category=Category.FUSED_COMPOSITE, level=Level.L3,
        signature=("causal_conv1d_update(Tensor x, Tensor conv_state, Tensor weight, "
                   "Tensor bias, Tensor conv_state_indices) -> (Tensor out, Tensor conv_state)"),
        inputs=[IOSpec(name="x", dtype=["bfloat16"]),
                IOSpec(name="conv_state", dtype=["bfloat16"]),
                IOSpec(name="weight", dtype=["bfloat16"]),
                IOSpec(name="bias", dtype=["bfloat16"]),
                IOSpec(name="conv_state_indices", dtype=["int32"])],
        outputs=[IOSpec(name="out", dtype=["bfloat16"]),
                 IOSpec(name="conv_state", dtype=["bfloat16"])],
        attrs=[AttrSpec(name="activation", type="String", default="silu")],
        shape_support=ShapeSupport(min_rank=2, max_rank=3,
                                   notes="x:(B,D) 或 (B,D,seqlen)；conv_state:(N,D,state_len)"),
        source_name="causal_conv1d", source_dir="tutorial/best_practice/003-fused-cat-slice-conv1d",
        entry="causal_conv1d_update", params=["x", "conv_state", "weight", "bias",
                                              "conv_state_indices"],
    ),
    OperatorSpec(
        name="GatherScatter", category=Category.INDEX_GATHER, level=Level.L3,
        signature="gather_scatter(Tensor x, Tensor indices, Tensor weights) -> Tensor out",
        inputs=[IOSpec(name="x", dtype=["float16"]),
                IOSpec(name="indices", dtype=["int32"]),
                IOSpec(name="weights", dtype=["float16"])],
        outputs=[IOSpec(name="out", dtype=["float16"])],
        attrs=[AttrSpec(name="top_k", type="int", required=True),
               AttrSpec(name="num_experts", type="int", default=8)],
        shape_support=ShapeSupport(min_rank=2, max_rank=2, notes="x:(sl, hs)"),
        source_name="gather_scatter", source_dir="tutorial/best_practice/004-gather_scatter",
        entry="gather_scatter", params=["x", "indices", "weights"],
    ),
    OperatorSpec(
        name="BinnedGatherScatter", category=Category.INDEX_GATHER, level=Level.L3,
        signature=("binned_gather_scatter(Tensor x, Tensor indices, Tensor weights, "
                   "Tensor bins, Tensor bin_ids) -> Tensor out"),
        inputs=[IOSpec(name="x", dtype=["float16"]),
                IOSpec(name="indices", dtype=["int32"]),
                IOSpec(name="weights", dtype=["float16"]),
                IOSpec(name="bins", dtype=["int32"]),
                IOSpec(name="bin_ids", dtype=["int32"])],
        outputs=[IOSpec(name="out", dtype=["float16"])],
        attrs=[AttrSpec(name="expert_capacity", type="int", required=True),
               AttrSpec(name="top_k", type="int", required=True),
               AttrSpec(name="num_experts", type="int", default=8)],
        shape_support=ShapeSupport(min_rank=2, max_rank=2, notes="x:(sl, hs)"),
        source_name="binned_gather_scatter",
        source_dir="tutorial/best_practice/005-binned_gather_scatter",
        entry="binned_gather_scatter", params=["x", "indices", "weights", "bins", "bin_ids"],
    ),
    OperatorSpec(
        name="PaddedGatherScatter", category=Category.INDEX_GATHER, level=Level.L3,
        signature=("padded_gather_scatter(Tensor x, Tensor indices, Tensor weights, "
                   "Tensor bin_ids, Tensor bins, Tensor padded_bins) -> Tensor out"),
        inputs=[IOSpec(name="x", dtype=["float16"]),
                IOSpec(name="indices", dtype=["int32"]),
                IOSpec(name="weights", dtype=["float16"]),
                IOSpec(name="bin_ids", dtype=["int32"]),
                IOSpec(name="bins", dtype=["int32"]),
                IOSpec(name="padded_bins", dtype=["int32"])],
        outputs=[IOSpec(name="out", dtype=["float16"])],
        attrs=[AttrSpec(name="top_k", type="int", required=True),
               AttrSpec(name="num_experts", type="int", default=8)],
        shape_support=ShapeSupport(min_rank=2, max_rank=2, notes="x:(sl, hs)"),
        source_name="padded_gather_scatter",
        source_dir="tutorial/best_practice/006-padded_gather_scatter",
        entry="padded_gather_scatter",
        params=["x", "indices", "weights", "bin_ids", "bins", "padded_bins"],
    ),
]


def extend_registry() -> None:
    """把 11 个 ascend 算子并入主注册表（oracle 解析依赖 by_name）。"""
    from .operator_registry import ALL_OPERATORS
    known = {op.name for op in ALL_OPERATORS}
    for op in ASCEND_OPS:
        if op.name not in known:
            ALL_OPERATORS.append(op)
