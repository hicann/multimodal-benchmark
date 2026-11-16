# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""样例提交的自检（L0，不需要 torch / NPU）。

守护什么：
    `examples/demo-submission` 是给提交方的模板，也是端到端验证的输入。
    它一旦过不了自己的校验器，整套演示就是假的——而且失败会发生在
    昇腾机上，来回一轮就是几十分钟。

    本测试只做**不需要执行**的检查：静态反作弊扫描与入口签名。
    这两类恰好是最容易写错的（实测教训：常量误判、参数名不符），
    而在本地即可完整验证。运行正确性由 NPU 侧的 test_npu 负责。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

from triton_eval.anticheat import scan_source
from triton_eval.operator_registry import by_name
from triton_eval.submission import SubmissionManifest, load_manifest

pytestmark = pytest.mark.torch

DEMO = fixture_path(Path(__file__).resolve().parents[2] / "examples" / "demo-submission")


def _manifest() -> SubmissionManifest:
    return load_manifest(DEMO)


def _func_params(source: str, name: str) -> list[str]:
    """用 AST 取入口函数的参数名。不 import 模块——那会拉进 triton。"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    raise AssertionError(f"文件中未找到函数 {name}")


def test_manifest_declares_registered_operators() -> None:
    for entry in _manifest().operators:
        assert by_name(entry.name) is not None, f"清单声明了未登记的算子：{entry.name}"
        assert (DEMO / entry.path).is_file(), f"实现文件不存在：{entry.path}"


def test_entry_signature_matches_operator_spec() -> None:
    """参数名与个数必须和规格一致——调用约定是位置传张量、关键字传属性。"""
    for entry in _manifest().operators:
        spec = by_name(entry.name)
        source = (DEMO / entry.path).read_text(encoding="utf-8")
        assert _func_params(source, entry.entry) == spec.params, (
            f"{entry.name} 的入口签名与规格不符：期望 {spec.params}，"
            f"实际 {_func_params(source, entry.entry)}"
        )


def test_kernels_pass_static_anti_cheat_scan() -> None:
    """模板自己必须过得了反作弊——否则是在教提交方怎么写违规实现。"""
    for entry in _manifest().operators:
        spec = by_name(entry.name)
        source = (DEMO / entry.path).read_text(encoding="utf-8")
        findings = scan_source(
            source, category=spec.category.value, entry=entry.entry
        )
        assert not findings, (
            f"{entry.name} 未通过静态扫描：" +
            "; ".join(f"[{f.rule_id}] {f.message}（第 {f.lineno} 行）" for f in findings)
        )


@pytest.mark.parametrize("banned", ["torch.softmax", "torch.argmax", "torch.gather",
                                    "torch.matmul", " torch.", "F."])
def test_no_framework_compute_calls_outside_kernels(banned: str) -> None:
    """核外不得出现框架计算调用。

    粗筛而非穷举：真正精确的判定由 scan_source 负责（见上一个测试），
    这里只是把最典型的代算写法显式钉住，防止以后有人「顺手」加回来。
    """
    if banned.strip() == "torch.":
        # 允许 torch.empty / empty_like 这类分配：分配不是计算
        pytest.skip("torch. 前缀本身不违规，由 scan_source 精确判定")
    for entry in _manifest().operators:
        source = (DEMO / entry.path).read_text(encoding="utf-8")
        assert banned not in source, f"{entry.name} 出现疑似代算调用：{banned}"


def test_transpose_launch_matches_kernel_signature() -> None:
    """kernel 签名与启动调用的参数个数必须一致（本机无 triton，AST 静态守卫）。

    实测教训：rank-8 扩展改了 kernel 签名（d0..d7 / s0..s7）却忘了改
    启动调用，8 个维度参数只传 4 个——本机没有 triton 编译不到这一步，
    错误直到昇腾机上才以 RUNTIME_FAIL 的形式暴露。
    """

    src = (DEMO / "operators" / "transpose" / "kernel.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    kernel_params: int | None = None
    launched: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Attribute) and d.attr == "jit"
            for d in node.decorator_list
        ):
            kernel_params = len(node.args.args)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            # 形如 _permute_kernel[(grid,)](x, out, n, ...)
            inner = getattr(node.func.value, "id", None)
            if inner and inner.endswith("_kernel"):
                launched = len(node.args) + len(node.keywords)
    assert kernel_params, "没找到 @triton.jit kernel"
    assert launched is not None, "没找到 kernel 启动调用"
    assert launched == kernel_params, (
        f"启动传 {launched} 个参数，kernel 声明 {kernel_params} 个——"
        "本机静态检查就该拦住，不要等昇腾机 RUNTIME_FAIL"
    )
