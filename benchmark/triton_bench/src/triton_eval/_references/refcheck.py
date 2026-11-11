# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现的自检（L0；有 torch 时可执行，不需要 NPU）。

存在原因：
    参考实现是判定的锚。它一旦有问题（缺文件、参数错位、输入越界、
    结果不确定），失败会被记成 ORACLE_FAIL——那是**评测方**的资产缺口，
    却长着一张「被测方没跑通」的脸。

    这类问题必须在冻结之前暴露，而不是等到昇腾机上跑完一轮才发现。
    实测教训：cross_entropy_loss 的两个整型用例在昇腾上两次执行结果不一致，
    却已经进了冻结集——因为冻结时从没跑过自检。

单独成模块的原因：
    自检既要被 `scripts/verify_refs.py`（人工核查）用，
    也要被 `scripts/build_corpus.py`（冻结门禁）用。
    两份实现迟早会不一致，而门禁失效比没有门禁更危险。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, NamedTuple

from .models import CaseSpec


def load_reference_segment(source_file: str, entry: str):
    """Load a primitive from the caller's corpus version without a module cache."""
    from .corpus_assets import resolve_reference_path

    source_path = Path(source_file).resolve()
    path = source_path.parents[1] / entry / "ref.py"
    if source_path.parents[1].name == "operators":
        path = resolve_reference_path(source_path.parents[2], f"operators/{entry}/ref.py")
    specification = importlib.util.spec_from_file_location(f"te_seg_{entry}", path)
    if specification is None or specification.loader is None:
        raise ValueError(f"Reference segment has no loader: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def scatter_reference(data, index, updates, dim=0, reduce=None):
    """Shared primary reference; independent CPU oracles do not call this helper."""
    out = data.clone()
    if reduce is None:
        out.scatter_(dim, index, updates)
    elif reduce == "add":
        out.scatter_add_(dim, index, updates)
    elif reduce in ("mul", "multiply"):
        out.scatter_mul_(dim, index, updates)
    else:
        raise ValueError(f"不支持的 reduce：{reduce}（合法取值：None / add / mul）")
    return out


class _ReferenceDependencies(NamedTuple):
    torch: Any
    as_outputs: Any
    dtype_of: Any
    make_inputs_for: Any
    operator_make_inputs: Any
    reference_for: Any
    by_name: Any


def _load_deps():
    """torch 是可选的：无 torch 的机器（纯 L0 开发机）应能导入本模块。"""
    import torch

    from .inputs import as_outputs, _dtype_of, make_inputs_for
    from .oracle import make_inputs_for as operator_make_inputs
    from .oracle import reference_for
    from .operator_registry import by_name

    return _ReferenceDependencies(
        torch=torch, as_outputs=as_outputs, dtype_of=_dtype_of, make_inputs_for=make_inputs_for,
        operator_make_inputs=operator_make_inputs, reference_for=reference_for, by_name=by_name)


def _mismatch_detail(actual, expected) -> str:
    """Bounded, exact value diagnostics for a failed independent reference gate."""
    import torch

    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    detail = (f"，shape={tuple(actual.shape)}/{tuple(expected.shape)}"
              f"，dtype={actual.dtype}/{expected.dtype}")
    if actual.shape != expected.shape:
        return detail
    unequal = actual != expected
    count = int(unequal.sum())
    detail += f"，mismatch_count={count}/{actual.numel()}"
    if count:
        index = tuple(int(value) for value in unequal.nonzero()[0].tolist())
        detail += f"，first_index={index}，actual={actual[index].item()}，expected={expected[index].item()}"
        difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
        detail += f"，max_abs={float(difference.max()):.3e}"
    return detail


def check_case(case: CaseSpec, device: str = "cpu") -> str | None:
    """自检一个用例的参考路径，返回失败原因；通过返回 None。"""
    (torch, as_outputs, _dtype_of, make_inputs_for, operator_make_inputs,
     reference_for, by_name) = _load_deps()

    spec = by_name(case.operator)
    if spec is None:
        return f"算子未注册：{case.operator}"
    reference = reference_for(case.operator)
    if reference is None:
        return "缺少 ref.py 或其中没有 reference"

    attrs = spec.resolve_attrs(case.attrs)
    kwargs = {k: v for k, v in attrs.items() if v is not None}
    try:
        inputs = make_inputs_for(case, _dtype_of(case.dtype), device, attrs,
                                 operator_make_inputs(case.operator))
        from .ascend_contracts import validate_inputs

        validate_inputs(case.operator, inputs, attrs)
    except Exception as exc:  # noqa: BLE001
        return f"构造输入失败：{type(exc).__name__}: {exc}"

    try:
        first_inputs = [tensor.clone() for tensor in inputs]
        from .reference_execution import reference_execution
        with reference_execution(case.operator, first_inputs):
            out = as_outputs(reference(*first_inputs, **kwargs))
        if any(not torch.equal(before, after)
               for before, after in zip(inputs, first_inputs)):
            return "参考实现修改了输入"
        out = [tensor.clone() for tensor in out]
    except Exception as exc:  # noqa: BLE001
        return f"参考实现失败：{type(exc).__name__}: {exc}"

    error = _check_reference_outputs(out, spec, inputs, torch)
    if error:
        return error

    # 确定性：同一个锚点原地再算一次必须给出同一个结果。
    # 不确定的参考会让同一用例在不同轮次得到不同判定，
    # 那样「静默错误」与「测量噪声」就无法区分。
    try:
        second_inputs = [tensor.clone() for tensor in inputs]
        with reference_execution(case.operator, second_inputs):
            again = as_outputs(reference(*second_inputs, **kwargs))
        if any(not torch.equal(before, after)
               for before, after in zip(inputs, second_inputs)):
            return "参考实现复算修改了输入"
    except Exception as exc:  # noqa: BLE001
        return f"参考实现复算失败：{type(exc).__name__}: {exc}"
    if len(out) != len(again):
        return "参考实现不确定：两次输出元数不同"
    error = _check_reference_repeat(out, again, torch)
    if error:
        return error
    return _check_independent_reference(case, inputs, kwargs, out, torch, as_outputs)


def check_all(
    cases: list[CaseSpec], device: str = "cpu",
    corpus_dir: Path | str | None = None,
) -> dict[str, str]:
    """自检全部用例，返回 {case_code: 失败原因}。

    corpus_dir：这些用例所属的语料目录。参考实现随语料版本冻结，
    必须从同一目录加载——不传则维持 oracle 的当前配置（仅适合
    调用方已 configure_corpus 的场景）。
    """
    if corpus_dir is not None:
        from .oracle import configure_corpus

        configure_corpus(corpus_dir)
    failures: dict[str, str] = {}
    for case in cases:
        why = check_case(case, device)
        if why:
            failures[case.code] = why
    return failures


def torch_available() -> bool:
    from importlib import util

    return util.find_spec("torch") is not None


def _unused(_: Any) -> None:  # pragma: no cover
    return None


def _check_reference_outputs(out, spec, inputs, torch):
    if len(out) != len(spec.outputs):
        return f"参考输出元数不符：规格 {len(spec.outputs)}，实际 {len(out)}"
    for index, (tensor, declared) in enumerate(zip(out, spec.outputs)):
        if str(tensor.dtype).replace("torch.", "") not in declared.dtype:
            return f"参考输出 {index} dtype 不符合规格：{tensor.dtype}"
        if tensor.device != inputs[0].device:
            return f"参考输出 {index} device 不符合输入设备"
        if tensor.numel() == 0:
            return f"参考输出 {index} 为空张量"
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            return f"参考输出 {index} 含 NaN / Inf——锚点本身不可信"

    return None



def _check_reference_repeat(out, again, torch):
    for index, (first, second) in enumerate(zip(out, again)):
        if first.dtype == second.dtype and first.device == second.device \
                and torch.equal(first, second):
            continue
        # 附带不一致的量级（诊断用）：平台级不确定性（如 torch_npu 的
        # cross_entropy 归约）是否在容差量级内，决定登记口径——只报
        # "不一致"则无从判断。2026-10-14 真机 5e 首跑发现 CE 3 例时加。
        detail = ""
        try:
            if first.shape == second.shape:
                a, b = first.to(torch.float32), second.to(torch.float32)
                diff = (a - b).abs()
                denom = a.abs().clamp_min(1e-30)
                detail = (f"，max_abs={float(diff.max()):.3e}"
                          f"，max_rel={float((diff / denom).max()):.3e}")
            else:
                detail = "，两次结果形状不同，无法计算量级"
        except Exception:  # noqa: BLE001 - 量级是附加诊断，算不出不影响判定
            detail = "，量级计算失败"
        return f"参考实现不确定：输出 {index} 两次执行结果不一致{detail}"


def _check_independent_reference(case, inputs, kwargs, out, torch, as_outputs):
    from .comparison import output_matches
    from .oracle import cross_reference_for
    from .specload import load_tolerance

    secondary = cross_reference_for(case.operator)
    if secondary is None:
        return None
    try:
        independent_inputs = [tensor.detach().to("cpu").clone() for tensor in inputs]
        before = [tensor.clone() for tensor in independent_inputs]
        independent = as_outputs(secondary(*independent_inputs, **kwargs))
        if any(not torch.equal(first, second)
               for first, second in zip(before, independent_inputs)):
            return "第二参考修改了输入"
        if len(out) != len(independent):
            return "第二参考输出元数不一致"
        table = load_tolerance()
        for index, (first, second) in enumerate(zip(out, independent)):
            dtype = str(first.dtype).replace("torch.", "")
            tolerance = table.for_dtype(dtype)
            if first.shape != second.shape or first.dtype != second.dtype \
                    or not output_matches(first, second, tolerance):
                return f"第二参考输出 {index} 不一致{_mismatch_detail(first, second)}"
    except Exception as exc:
        return f"第二参考失败：{type(exc).__name__}: {exc}"
    return None
