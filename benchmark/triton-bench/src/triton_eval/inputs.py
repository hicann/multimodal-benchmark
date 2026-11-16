# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""输入构造与返回值归一化。

单独成模块（而不是留在 L1 的 npu_exec 里）的原因：
    这些函数只依赖 torch，不需要 NPU。而参考实现的自检脚本
    `scripts/verify_refs.py` 要在**纯 CPU** 上把它们跑一遍；
    npu_exec 因为 `import torch_npu` 在无昇腾的机器上无从导入，
    于是参考实现只能等到昇腾机上才第一次被执行——问题发现得太晚。

torch 一律在函数内导入：L0 的测试与 CLI 子命令必须能在没有 torch 的
机器上运行（这是 results.jsonl 契约带来的能力）。
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from .models import CaseSpec


def reference_sample(seed: int, count: int, dtype):
    """Sample legacy indexing inputs using their original private CPU generator."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)
    if dtype in integer_dtypes:
        return torch.randint(1, 100, (count,), generator=generator, dtype=torch.int64).to(dtype)
    return torch.randn(count, generator=generator).to(dtype)


def permutation_inputs(seed, shape, device):
    import torch

    torch.manual_seed(seed)
    rows, columns = shape
    target = device if isinstance(device, torch.device) else torch.device(device)
    values = torch.randn((rows, columns), device=target, dtype=torch.float32)
    indices = torch.randperm(rows, device=target).to(torch.int32)
    return [values, indices]


def vector_add_inputs(seed, dtype, shape, device):
    import torch

    torch.manual_seed(seed)
    count = shape[0]
    target = device if isinstance(device, torch.device) else torch.device(device)
    dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    if not dtype.is_floating_point:
        first = torch.randint(-5, 5, (count,), device=target).to(dtype)
        second = torch.randint(-5, 5, (count,), device=target).to(dtype)
    else:
        first = torch.randn((count,), device=target).to(dtype)
        second = torch.randn((count,), device=target).to(dtype)
    return [first, second]


def vector_cmp_inputs(seed, shape, device):
    import torch

    torch.manual_seed(seed)
    rows, columns = shape
    target = device if isinstance(device, torch.device) else torch.device(device)
    return [torch.randn((rows, columns), device=target, dtype=torch.float32)]


def token_pool_inputs(seed, device, attrs, *, unique_requests):
    import torch

    torch.manual_seed(seed)
    batch = int(attrs["bs_upper"])
    pool_length = int(attrs["pool_len"])
    maximum_batch = 512
    target = device if isinstance(device, torch.device) else torch.device(device)
    pool = torch.zeros((maximum_batch, pool_length), device=target, dtype=torch.int32)
    if unique_requests:
        requests = torch.randperm(maximum_batch, device=target)[:batch].to(torch.int32)
    else:
        requests = torch.randint(0, maximum_batch, (batch,), device=target, dtype=torch.int32)
    starts = torch.randint(0, pool_length - 2, (batch,), device=target, dtype=torch.int32)
    ends = starts + torch.randint(1, 3, (batch,), device=target, dtype=torch.int32)
    length = int((ends - starts).sum())
    locations = torch.randint(0, 8 * pool_length, (length,), device=target, dtype=torch.int32)
    return [requests, pool, starts, ends, locations]


class ExpertRoutingInputs(NamedTuple):
    values: Any
    experts: Any
    bins: Any
    indices: Any
    expert_count: int


def expert_routing_inputs(seed, shape, device, attrs, *, legacy_shape=False):
    import torch

    torch.manual_seed(seed)
    if legacy_shape:
        rows, columns, num_experts = shape
    else:
        rows, columns = shape
        num_experts = int(attrs["num_experts"])
    top_k = int(attrs["top_k"])
    target = device if isinstance(device, torch.device) else torch.device(device)
    values = torch.randn((rows, columns), device=target).to(torch.float16)
    experts = torch.randint(0, num_experts, (rows * top_k,), device=target).to(torch.int32)
    bins, indices = torch.sort(experts)
    return ExpertRoutingInputs(values, experts, bins, indices, num_experts)


def _dtype_of(name: str):
    """用例的 dtype 字符串 → torch.dtype。"""
    import torch

    mapping = {
        "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
        "bfloat16": torch.bfloat16, "int8": torch.int8, "int16": torch.int16,
        "int32": torch.int32, "int64": torch.int64, "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if name.lower() not in mapping:
        raise KeyError(f"不支持的 dtype：{name}")
    return mapping[name.lower()]


def _as_tensor(obj: Any):
    """把实现或参考的返回值统一为**单个**张量（取第一个元素）。

    仅用于只需主输出的场合（如形状检查、诊断输出）。
    判定一律走 as_outputs——见下方说明。
    """
    return as_outputs(obj)[0]


def as_outputs(obj: Any) -> list:
    """把实现或参考的返回值统一为张量列表。

    为什么必须保留全部输出：
        曾经只取第一个元素，于是多输出算子的**第二个输出完全不参与判定**。
        实测（ADV-F1）：cummin 的 indices 全填垃圾，5 个用例全部 PASS——
        「多输出」在冻结集里是个假象：名义上覆盖了（Cummin 有 values/indices
        两个输出），实际上零校验。

        这类假象比「没覆盖」更危险：它会让人以为测过了。
    """
    import torch

    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)) and obj:
        outs = [o for o in obj if isinstance(o, torch.Tensor)]
        if outs:
            return outs
    raise TypeError(f"实现/参考的返回值不是张量：{type(obj).__name__}")


def tensor_fp(t) -> tuple:
    """张量指纹：(dtype, shape, 数值摘要)。用于发现输入被就地修改。

    只取数值摘要而非副本：输入可能很大，全量拷贝会拖慢评测。
    放在 L0（而非 npu_exec）的原因：这条契约的测试不需要 NPU，
    留在 npu_exec 里会因为 `import torch_npu` 只能在昇腾机上验证。
    """
    import torch

    try:
        d = t.detach().to("cpu").to(torch.float32)
        return (str(t.dtype), tuple(t.shape), float(d.sum()), float(d.abs().sum()))
    except Exception:  # noqa: BLE001
        return (str(t.dtype), tuple(t.shape), 0.0, 0.0)


def _default_inputs(case: CaseSpec, dtype, dev: str) -> list:
    """单输入默认构造：由 seed 驱动，保证可复现。

    整型输入用 randint 而非 randn→int（后者会截断出大量 0，失去区分度）。
    """
    n = 1
    for d in case.shape:
        n *= d
    return [reference_sample(case.seed, n, dtype).reshape(case.shape).to(dev)]


# extreme 只对宽动态范围的浮点开放：fp16 max=65504，
# ±20σ 的中间乘积（如 grouped_matmul 的 x·w）会溢出成 Inf，
# 那是数据表示的天花板而不是被测实现的缺陷
_EXTREME_SAFE_DTYPES = ("float32", "bfloat16")


def _apply_value_range(t, case: CaseSpec):
    """对主输入施加数值分布（zeros / mixed / extreme）。

    只动主输入的第一个张量：其余输入（index / mask / target / weight ...）
    的取值有语义约束（索引必须落在界内、掩码必须是 0/1），
    事后缩放会破坏不变式。主输入没有这类约束。
    """
    import torch

    kind = (case.value_range or {}).get("kind", "normal")
    if kind == "zeros":
        return t.zero_()
    if kind == "mixed":
        flat = t.reshape(-1)
        n = flat.numel()
        half = max(n // 2, 1)
        with torch.no_grad():
            flat[half:] = 0
        return t
    if kind == "extreme":
        if str(t.dtype).replace("torch.", "") not in _EXTREME_SAFE_DTYPES:
            return t  # 安全阀：不该出现在生成层，防御性兜底
        return t.mul_(20.0)
    return t


def _make_strided(t):
    """逻辑 shape 不变、存储跨步的构造：翻倍分配后隔位取。

    实现必须处理 stride（或显式 .contiguous() 聚合）才能正确读数——
    这正是布局维度的考点。
    """
    import torch

    if t.dim() == 0 or t.numel() < 2:
        return t
    flat = t.reshape(-1)
    doubled = torch.zeros(flat.numel() * 2, dtype=t.dtype, device=t.device)
    doubled[::2].copy_(flat)
    return doubled[::2].reshape(t.shape)


def make_inputs_for(
    case: CaseSpec,
    dtype,
    dev: str,
    attrs: dict[str, Any] | None = None,
    make_inputs: Callable[..., Any] | None = None,
) -> list:
    """按用例规格构造输入。

    为什么允许算子自定义（make_inputs）：
        一个 shape 只能描述**一个**张量。maximum / gather / scatter /
        grouped_matmul / rms_norm / masked_scale / cross_entropy_loss
        都是多输入算子，第二个输入的形状与取值范围（如 gather 的 index
        必须落在 [0, size(dim))、grouped_matmul 的 weight 组数由
        group_list 决定）无法从主输入的 shape 推出来。
        输入构造因此下沉到算子的参考实现 ref.py，缺失时回落到单张量默认实现。

    默认在统一出口对第一个张量施加数值分布。自定义构造器可显式声明
    value_range 参数，接管合法域内的分布构造（例如 Where 的主数据为第二
    个张量，Log 的输入必须为正）；此时不再重复施加通用变换。
    布局构造始终在数值构造之后，并保持全部逻辑数值。
    """
    owns_value_range = False
    if make_inputs is None:
        inputs = _default_inputs(case, dtype, dev)
    else:
        import inspect

        owns_value_range = "value_range" in inspect.signature(make_inputs).parameters
        extra = {"value_range": dict(case.value_range or {})} if owns_value_range else {}
        inputs = list(
            make_inputs(
                seed=case.seed,
                dtype=dtype,
                shape=list(case.shape),
                device=dev,
                attrs=dict(attrs or {}),
                **extra,
            )
        )
    if inputs and isinstance(inputs[0], object) and hasattr(inputs[0], "shape"):
        # 顺序必须先数值后跨步：reshape 在非连续张量上会退化为拷贝，
        # mixed 的就地写会写丢；先施加数值分布（此时仍连续）再构造跨步
        if not owns_value_range:
            _apply_value_range(inputs[0], case)
        if not case.contiguous:
            # Official Where declares the data tensor explicitly through its factory.
            main_index = getattr(make_inputs, "main_input_index", 0)
            inputs[main_index] = _make_strided(inputs[main_index])
    return inputs


def validate_elementwise_layout(inputs, first, elementwise):
    """Validate layout metadata without sharing any oracle computation."""
    for tensor in inputs:
        if tensor.is_contiguous():
            continue
        if not elementwise or tensor is not first:
            raise ValueError("only the elementwise main input may be strided")
        expected_stride = 2
        for size, stride in zip(reversed(tensor.shape), reversed(tensor.stride())):
            if size > 1 and stride != expected_stride:
                raise ValueError("main input must have uniform flat stride=2")
            expected_stride *= size
