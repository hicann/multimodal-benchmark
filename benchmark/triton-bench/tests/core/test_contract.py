# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""契约守卫测试。

守护的是几条跨模块的约定，它们一旦被破坏，整套可迁移性就没了。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_NPU = PROJECT_ROOT / "tests/runtime/test_npu.py"
CONFTEST = PROJECT_ROOT / "tests/conftest.py"


def _triton_present() -> bool:
    return importlib.util.find_spec("triton") is not None


def test_results_jsonl_is_the_source_of_truth() -> None:
    """results.jsonl 必须能被独立读取并重建记录。"""
    from triton_eval.corpusmod import read_result_lines
    from triton_eval.models import CaseResultRecord

    rec = CaseResultRecord(run_id=1, case_code="c", operator="add", status="PASS")
    p = Path("/tmp/_te_contract.jsonl")
    p.write_text(rec.model_dump_json() + "\n", encoding="utf-8")
    loaded = read_result_lines(p)
    assert len(loaded) == 1 and loaded[0].case_code == "c"


def test_npu_tests_are_skipped_not_failed_without_hardware() -> None:
    """没有 NPU 时，npu 标记的用例必须被跳过，而不是失败。

    否则同一份测试套件无法在两种环境下共用。
    """
    if _triton_present():
        pytest.skip("当前环境有 triton，跳过行为不适用")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST_NPU), "-q"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=False,
    )
    assert proc.returncode == 0, f"本地执行 npu 用例不应失败：\n{proc.stdout}"
    assert "skipped" in proc.stdout


def test_conftest_exposes_triton_probe() -> None:
    spec = importlib.util.spec_from_file_location("_te_conftest", CONFTEST)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.triton_available() is _triton_present()


def test_l1_modules_import_torch_only_at_module_level() -> None:
    """L1 模块的源码里应当出现 torch / triton 的导入，
    L0 模块则不应出现——这是分层是否被破坏的静态检查。
    """
    from triton_eval import scorer, store, judge

    for mod in (scorer, store, judge):
        src = Path(mod.__file__).read_text(encoding="utf-8")  # type: ignore[arg-type]
        assert "import triton" not in src, f"{mod.__name__} 不应 import triton"
        assert "import torch" not in src, f"{mod.__name__} 不应 import torch"


def _imports_triton(path: Path) -> bool:
    """用 AST 判断模块是否真的 import 了 triton（不误判注释与文档字符串）。"""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "triton" or alias.name.startswith("triton.") for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "triton" or mod.startswith("triton."):
                return True
    return False


def test_triton_imports_confined_to_l1_modules() -> None:
    """triton 只允许出现在 L1 模块里。

    允许的导入点：
        harness/npu_exec.py  —— 编译与执行
        bench/measure.py     —— do_bench 计时
        envprobe/collect.py  —— 读取实测 import 版本（包在 try 里，可降级）
    任何 L0 模块 import triton 都意味着分层被破坏。
    """
    allowed = {"npu_exec.py", "measure.py", "collect.py"}
    hits = sorted(
        p.name for p in (PROJECT_ROOT / "src").rglob("*.py") if _imports_triton(p)
    )
    assert set(hits) <= allowed, f"出现非预期的 triton 导入点：{hits}"


def test_npu_exec_imports_torch_npu_at_module_level() -> None:
    """harness/npu_exec.py 必须在模块级 import torch_npu。

    它是**副作用导入**：注册 `torch.npu` 属性。漏掉会导致
        AttributeError: module 'torch' has no attribute 'npu'
    且症状具有迷惑性——CLI 能跑（因为 cli.py 的 L1 链路上间接导入了），
    但直接调用 execute_case 的测试和脚本全崩。
    曾漏在 build_corpus.py --screen 与多个测试文件中。
    """
    import ast

    path = PROJECT_ROOT / "src" / "triton_eval" / "harness" / "npu_exec.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_level.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_level.add(node.module or "")
    assert "torch_npu" in module_level, (
        "harness/npu_exec.py 缺少模块级 import torch_npu，"
        "会导致 torch.npu 未注册"
    )


def test_npu_test_modules_do_not_import_l1_at_module_level() -> None:
    """tests/ 下带 npu 标记的文件，不得在模块级导入 L1 模块。

    曾经的 bug：test_npu.py 在模块级 `from triton_eval.bench.measure
    import measure`，而 measure 会 import triton。在没有 NPU 的机器上，
    pytest **收集阶段**就崩，conftest 的 skip 逻辑来不及生效，
    整个测试套件无法运行。L1 导入必须放进测试函数体内。
    """
    import ast

    for path in Path(__file__).parent.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # 只看模块级语句
            if not (isinstance(node, ast.ImportFrom)):
                continue
            mod = node.module or ""
            if mod.startswith("triton_eval.bench.measure") or \
                    mod.startswith("triton_eval.harness.npu_exec"):
                pytest.fail(
                    f"{path.name} 在模块级导入了 L1 模块 {mod}。"
                    "这会 import triton，导致无 NPU 环境下 pytest 收集阶段崩溃。"
                    "请把该导入移进测试函数体内。"
                )


def test_l0_modules_never_import_triton() -> None:
    from triton_eval import corpusmod, judge, models, report, scorer, specload, store
    from triton_eval.bench import stats
    from triton_eval.envprobe import parse

    for mod in (corpusmod, judge, models, report, scorer, specload, store, stats, parse):
        path = Path(mod.__file__)  # type: ignore[arg-type]
        assert not _imports_triton(path), f"L0 模块 {mod.__name__} 不应 import triton"


def test_grouped_framework_keeps_public_module_imports_and_asset_paths():
    import triton_eval

    names = (
        'backend_capture', 'backend_evidence', 'compilation_evidence', 'coverage_evidence',
        'coverage_gate', 'coverage_generation', 'coverage_overlay', 'coverage_reference',
        'implementation_evidence', 'pattern_contracts', 'pattern_evidence', 'pattern_generation',
        'pattern_release', 'reviewed_kernels', 'official_reference', 'official_reference_cpu',
        'refcheck', 'reference_execution', 'oracle', 'official_catalog', 'official_generation',
        'ascend_contracts', 'ascend_lowering', 'independent_submissions', 'submission_http',
        'submission_preflight', 'submission_service', 'submission_simulation',
    )
    for name in names:
        module = importlib.import_module('triton_eval.' + name)
        expected = triton_eval.framework_path('src/triton_eval/' + name + '.py')
        assert Path(module.__file__) == expected
        assert module.__name__ == 'triton_eval.' + name
        assert module.__package__ == 'triton_eval'
        assert triton_eval.PROJECT_ROOT == PROJECT_ROOT
