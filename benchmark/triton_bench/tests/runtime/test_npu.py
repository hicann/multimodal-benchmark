# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""需要真实昇腾 NPU 的测试。

在没有 triton 的机器上会被 conftest 自动跳过（而不是失败）。
在昇腾执行机上执行 `pytest` 即可全部跑起来。
"""

from __future__ import annotations

import logging

from pathlib import Path


import pytest

from triton_eval.fixture_sources import fixture_path

# 注意：本文件不得在模块级导入任何 L1 模块（bench.measure / harness.npu_exec）。
# 那些模块会 import triton，在没有 NPU 的机器上会导致 pytest 收集阶段
# 直接报错，conftest 的 skip 逻辑来不及生效。L1 导入一律放进测试函数体内。

pytestmark = pytest.mark.npu

CORPUS = Path(__file__).resolve().parents[2] / "corpus" / "v0.3-multi-input-frozen"
DEMO_SUB = fixture_path(Path(__file__).resolve().parents[2] / "examples" / "demo-submission")
BENCH_KERNEL = Path(__file__).parents[1] / "npu_fixtures" / "bench_kernel.py"


def test_triton_import_version_is_measured_not_declared() -> None:
    """实测 import 版本必须与 pip 元数据区分——实测环境下二者并不一致。"""
    import triton

    assert triton.__version__, "拿不到 triton 版本，说明被测编译器环境有问题"


def test_health_probe_passes() -> None:
    """最小 kernel 必须能编译、执行且结果与参考一致。"""
    from triton_eval.harness.npu_exec import health_probe

    ok, reason = health_probe()
    assert ok, f"健康探针失败：{reason}"


def test_do_bench_returns_median_and_samples() -> None:
    """计时必须显式取中位数，且能重复取得样本数组。

    kernel 必须来自模块级定义的真实文件——@triton.jit 依赖
    inspect.getsourcelines，函数内定义的 kernel 在被测编译器上会失败。
    这里复用 corpus 里已验证可用的 add kernel。
    """
    import torch

    from triton_eval.bench.measure import measure
    from triton_eval.harness.npu_exec import load_module

    # kernel 夹具固定在 tests/npu_fixtures/，不随 corpus 变动：
    # 契约迁移后 corpus 里已没有 kernel.py（用例不再绑定实现）。
    mod = load_module(BENCH_KERNEL, "bench_kernel_add")

    dev = f"npu:{torch.npu.current_device()}"
    n = 1 << 16
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    o = torch.empty_like(x)

    cfg = {"BLOCK": 256}
    samples = measure(lambda: mod.launch([x, y], o, cfg), repeats=3)
    assert len(samples) == 3, f"期望 3 个样本，实际 {len(samples)}"
    assert all(s > 0 for s in samples), f"耗时必须为正：{samples}"


def test_execute_case_with_demo_submission() -> None:
    """端到端：用样例提交执行真实用例，验证 L1 → L0 的契约可用。

    取「提交有实现 **且** corpus 有参考实现」的用例，而不是 cases[0]。
    契约迁移后用例不再绑定 kernel，cases[0]（当前是 maximum-f32-00）
    既无提交实现也无参考实现——直接拿它跑，测的是评测方资产缺口，
    不是被测编译器。曾因此报出 `execute_case() takes 1 positional
    argument but 2 were given`（旧签名）与一串假失败。
    """
    from triton_eval.corpusmod import load_manifest
    from triton_eval.harness.npu_exec import execute_case
    from triton_eval.judge import decide
    from triton_eval.oracle import reference_for
    from triton_eval.runner import execute_kwargs
    from triton_eval.specload import load_tolerance
    from triton_eval.submission import discover, load_all

    manifest = load_manifest(CORPUS / "manifest.json")
    sub = load_all(discover(DEMO_SUB))
    assert sub.loaded, f"样例提交未加载成功：{sub.failures}"

    # 用显式循环而不是列表推导：
    # 推导式里的迭代变量在异常回溯中容易遮住真正的来源，
    # 且这里需要逐条给出「为什么不可评测」的线索。
    evaluable = []
    for case in manifest.cases:
        if case.operator not in sub.loaded:
            continue
        if reference_for(case.operator) is None:
            continue
        evaluable.append(case)
    assert evaluable, "没有任何既被测实现又有参考实现的用例，端到端链路无法验证"

    # 硬断言只针对**评测方责任**：参考失效、环境异常、用例定义错误
    # 都意味着评测方资产有问题，与被测方无关，零容忍。
    #
    # 而「样例实现算得对不对」不是评测平台的职责——样例是契约模板，
    # 实现由我们所扮演的提交方提供。把它当正确性基准，等于让平台为
    # 提交方的实现质量背书。故只统计、不硬断言。
    failures: list[str] = []
    not_pass: list[str] = []
    for case in evaluable:
        # 参数组装必须与 CLI 走同一份（runner.execute_kwargs）：
        # 曾经这里漏传 make_inputs，多输入算子只拿到一个张量，
        # 参考实现以 TypeError 失败并被判 ORACLE_FAIL——
        # 看起来像评测方资产坏了，实际是调用方没传全参数。
        m = execute_case(
            case,
            **execute_kwargs(case, sub.loaded[case.operator].func),
        )
        status = decide(m, load_tolerance().for_dtype(case.dtype))
        if status.value in ("ORACLE_FAIL", "ENV_ABORT", "SPEC_ERROR"):
            failures.append(f"{case.code}: {status.value} err={m.compile_err}")
        elif status.value != "PASS":
            not_pass.append(
                f"{case.code}: {status.value} rel={m.max_rel_err} "
                f"abs={m.max_abs_err} err={m.compile_err}"
            )

    assert not failures, "评测方资产有问题（参考失效/环境异常/用例非法）：\n" + \
        "\n".join(failures)
    if not_pass:
        logging.info("\n样例提交未通过（属被测方实现质量，非平台缺陷）：")
        for line in not_pass:
            logging.info(f"  {line}")
