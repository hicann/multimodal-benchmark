# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""昇腾示范算子的可执行输入关系。"""

from __future__ import annotations


def validate_inputs(operator, inputs, attrs):
    from .oracle import active_corpus

    if (active_corpus() / "coverage-plan.json").is_file():
        import json
        from .pattern_generation import PATTERN_VERSIONS

        plan = json.loads((active_corpus() / "coverage-plan.json").read_text())
        if plan.get("version") == "v0.19-workload-completion-candidates":
            from triton_eval.workload_generation import TARGETS
            from triton_eval.workload_contracts import validate_inputs as validate_workload

            if operator in TARGETS:
                validate_workload(operator, inputs, attrs)
                return
        if plan.get("version") in PATTERN_VERSIONS and operator in ("Cummin", "GroupedMatmul"):
            from .pattern_contracts import validate_inputs as validate_pattern

            validate_pattern(operator, inputs, attrs)
            return
        from .coverage_generation import EXTENSIONS

        if operator in EXTENSIONS:
            from .coverage_reference import validate_inputs as validate_extension

            validate_extension(operator, inputs, attrs)
        else:
            from .official_reference import validate_inputs as validate_official

            validate_official(operator, inputs, attrs)
        return

    if (active_corpus() / "official-catalog.json").is_file():
        from .official_catalog import BY_NAME
        from .official_reference import validate_inputs as validate_official

        if operator in BY_NAME:
            validate_official(operator, inputs, attrs)
            return

    _validate_ascend_inputs(operator, inputs, attrs)


def _validate_ascend_inputs(operator, inputs, attrs):
    import torch

    from .operator_registry import by_name
    from .operator_registry_ascend_ops import ASCEND_OPS

    if operator not in {spec.name for spec in ASCEND_OPS}:
        return
    spec = by_name(operator)
    if len(inputs) != len(spec.inputs):
        raise ValueError("输入元数不符合规格")
    for tensor, declared in zip(inputs, spec.inputs):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("输入必须为张量")
        if str(tensor.dtype).replace("torch.", "") not in declared.dtype:
            raise ValueError(f"输入 {declared.name} dtype 不符合规格")
        if tensor.device != inputs[0].device or not tensor.is_contiguous():
            raise ValueError("当前契约要求所有输入同设备且连续")
    if operator in (
        "VectorAdd",
        "VectorCmp",
        "InsertSlice",
        "ExtractSlice",
        "DiscreteMemoryAccess",
    ):
        _validate_vector_inputs(operator, inputs, attrs)
        return
    validators = {
        "AssignReqToTokenPool": _validate_pool_inputs,
        "CausalConv1d": _validate_convolution_inputs,
        "DecodeGroupedAttention": _validate_attention_inputs,
    }
    validators.get(operator, _validate_routing_inputs)(operator, inputs, attrs)


def _validate_vector_inputs(operator, inputs, attrs):
    import torch

    first = inputs[0]
    if operator == "VectorAdd":
        _require(
            first.shape == inputs[1].shape and first.dtype == inputs[1].dtype,
            "加法输入形状和 dtype 必须相同",
        )
        return
    if operator == "VectorCmp":
        _require(
            first.ndim == 2 and float(attrs.get("eps", 1e-6)) > 0, "归一化需要二维输入和正 eps"
        )
        return
    if operator in ("InsertSlice", "ExtractSlice"):
        indices = inputs[1]
        _require(first.ndim == 2 and indices.shape == (first.shape[0],), "置换尺寸错误")
        _require(
            torch.equal(
                torch.sort(indices.long()).values, torch.arange(first.shape[0], device=first.device)
            ),
            "indices 必须为行置换",
        )
        return
    if operator == "DiscreteMemoryAccess":
        _require(first.ndim == 1 and first.numel() == int(attrs["m"]), "m 与输入不一致")
        _require(inputs[1].numel() == int(attrs["n"]), "n 与索引不一致")
        _range(inputs[1], first.numel())


def _validate_pool_inputs(operator, inputs, attrs):
    import torch

    rows, pool, start, end, locations = inputs
    batch = int(attrs["bs_upper"])
    _require(rows.shape == start.shape == end.shape == (batch,), "请求向量尺寸错误")
    _require(pool.ndim == 2 and pool.shape[1] == int(attrs["pool_len"]), "池行长错误")
    _range(rows, pool.shape[0])
    _require(torch.unique(rows).numel() == batch, "请求行必须唯一，禁止未定义写竞争")
    _require(bool(((start >= 0) & (end >= start) & (end <= pool.shape[1])).all()), "写区间越界")
    _require(locations.numel() == int((end - start).sum()), "写入长度错误")


def _validate_convolution_inputs(operator, inputs, attrs):
    import torch

    values, state, weight, bias, indices = inputs
    batch, channels = values.shape[:2]
    _require(values.ndim in (2, 3) and state.ndim == 3, "卷积输入维数错误")
    _require(
        state.shape[1] == channels
        and weight.ndim == 2
        and weight.shape[0] == channels
        and state.shape[-1] >= weight.shape[-1] - 1,
        "卷积通道或历史窗口错误",
    )
    _require(bias.shape == (channels,) and indices.shape == (batch,), "卷积 bias/状态行尺寸错误")
    _range(indices, state.shape[0])
    _require(torch.unique(indices).numel() == batch, "状态行必须唯一")
    _require(
        str(attrs.get("activation", "silu")).lower() in ("silu", "swish", "none", "null"),
        "非法激活函数",
    )


def _validate_attention_inputs(operator, inputs, attrs):
    queries, keys, values, indptr, indices, counts = inputs
    batch, heads, dimension = queries.shape
    _require(
        keys.ndim == values.ndim == 3
        and keys.shape[:2] == values.shape[:2]
        and keys.shape[-1] == dimension
        and heads % keys.shape[1] == 0,
        "KV/head 关系错误",
    )
    _require(indptr.shape == (batch + 1,) and counts.shape == (batch,), "请求边界尺寸错误")
    _require(
        int(indptr[0]) == 0
        and int(indptr[-1]) == indices.numel()
        and bool((indptr[1:] >= indptr[:-1]).all()),
        "KV 请求边界错误",
    )
    _range(indices, keys.shape[0])
    _require(bool(((counts > 0) & (counts <= int(attrs["max_kv_splits"]))).all()), "split 数越界")


def _validate_routing_inputs(operator, inputs, attrs):
    import torch

    first = inputs[0]
    indices, weights = inputs[1:3]
    top_k = int(attrs["top_k"])
    _require(first.ndim == 2 and top_k > 0, "重排输入必须二维且 top_k 为正")
    _require(indices.numel() == weights.numel() == first.shape[0] * top_k, "路由长度错误")
    _require(
        torch.equal(
            torch.sort(indices.long()).values, torch.arange(indices.numel(), device=first.device)
        ),
        "路由索引必须为置换",
    )
    if operator != "GatherScatter":
        bins = inputs[3] if operator == "BinnedGatherScatter" else inputs[4]
        bin_ids = inputs[4] if operator == "BinnedGatherScatter" else inputs[3]
        _require(
            bins.ndim == 1
            and bins.numel() > 0
            and int(bins[-1]) == indices.numel()
            and bool((bins[1:] >= bins[:-1]).all()),
            "专家边界错误",
        )
        _require(bin_ids.shape == indices.shape, "专家编号长度错误")
        lengths = torch.diff(torch.cat((bins.new_zeros(1), bins))).long()
        expected = torch.repeat_interleave(torch.arange(bins.numel(), device=bins.device), lengths)
        _require(torch.equal(bin_ids.long(), expected), "专家编号与边界不一致")
        if operator == "BinnedGatherScatter":
            _require(int(attrs["expert_capacity"]) > 0, "专家容量必须为正")
        else:
            padded = inputs[5]
            padded_lengths = torch.diff(torch.cat((padded.new_zeros(1), padded))).long()
            _require(
                padded.shape == bins.shape
                and bool((padded_lengths >= lengths).all())
                and bool((padded_lengths % 128 == 0).all()),
                "padding 边界错误",
            )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _range(indices, size):
    _require(bool(((indices >= 0) & (indices < size)).all()), "索引越界")
