# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""提交契约测试（L0，无 NPU 依赖）。

每条规则都要有**合法样例**与**非法样例**——这是 cann-bench 的做法，
也是「鲁棒」能被引证的前提。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from triton_eval.submission import (
    MANIFEST_NAME,
    discover,
    load_all,
    load_manifest,
    load_operator,
    verify_unchanged,
)

MANIFEST_TMPL = """
submit_id: {sid}
operators:
  - name: Exp
    entry: exp
    path: operators/exp/kernel.py
"""

KERNEL_SRC = '''
def exp(x, base=-1.0, scale=1.0, shift=0.0):
    return x
'''


@pytest.fixture()
def good_submission(tmp_path: Path) -> Path:
    root = tmp_path / "sub-good"
    (root / "operators" / "exp").mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(
        MANIFEST_TMPL.format(sid="sub-good"), encoding="utf-8"
    )
    (root / "operators" / "exp" / "kernel.py").write_text(KERNEL_SRC, encoding="utf-8")
    return root


# ---------- 任务 4.1 ----------


def test_valid_manifest_parsed(good_submission: Path) -> None:
    m = load_manifest(good_submission)
    assert m.submit_id == "sub-good"
    assert len(m.operators) == 1
    assert m.operators[0].entry == "exp"


def test_missing_manifest_rejected_with_filename(tmp_path: Path) -> None:
    """缺清单必须报出文件名，不能只说「找不到」。"""
    with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
        load_manifest(tmp_path)


def test_discover_nonexistent_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "nope")


def test_unknown_field_in_manifest_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    root = tmp_path / "sub"
    root.mkdir()
    (root / MANIFEST_NAME).write_text(
        "submit_id: s\nunexpected_field: 1\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError):
        load_manifest(root)


# ---------- 任务 4.2：零构建加载 ----------


def test_load_without_build_step(good_submission: Path) -> None:
    """不得要求任何打包或编译产物。"""
    sub = discover(good_submission)
    # 目录中没有 setup.py / build.sh / dist/
    assert not (good_submission / "setup.py").exists()
    assert not (good_submission / "build.sh").exists()
    assert not (good_submission / "dist").exists()

    loaded = load_all(sub)
    assert "Exp" in loaded.loaded
    assert loaded.failures == {}


# ---------- 任务 4.3：签名校验 ----------


def test_entry_function_absent_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sub"
    (root / "operators" / "exp").mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(
        MANIFEST_TMPL.format(sid="sub"), encoding="utf-8"
    )
    (root / "operators" / "exp" / "kernel.py").write_text(
        "def not_exp(x):\n    return x\n", encoding="utf-8"
    )
    sub = discover(root)
    with pytest.raises(AttributeError, match="缺少入口函数"):
        load_operator(sub, sub.manifest.operators[0])


def test_implementation_file_missing_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sub"
    root.mkdir()
    (root / MANIFEST_NAME).write_text(
        MANIFEST_TMPL.format(sid="sub"), encoding="utf-8"
    )
    sub = discover(root)
    with pytest.raises(FileNotFoundError, match="实现文件不存在"):
        load_operator(sub, sub.manifest.operators[0])


def test_non_callable_entry_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sub"
    (root / "operators" / "exp").mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(
        MANIFEST_TMPL.format(sid="sub"), encoding="utf-8"
    )
    (root / "operators" / "exp" / "kernel.py").write_text(
        "exp = 42\n", encoding="utf-8"
    )
    sub = discover(root)
    with pytest.raises(TypeError, match="不可调用"):
        load_operator(sub, sub.manifest.operators[0])


def test_signature_exposed_for_validation(good_submission: Path) -> None:
    sub = load_all(discover(good_submission))
    op = sub.loaded["Exp"]
    assert op.signature() == ["x", "base", "scale", "shift"]


def test_parameter_count_mismatch_detected(good_submission: Path) -> None:
    """签名不符必须能同时给出期望与实际。"""
    from triton_eval.operator_registry import by_name

    spec = by_name("Exp")
    assert spec is not None
    sub = load_all(discover(good_submission))
    actual = sub.loaded["Exp"].signature()
    assert len(actual) == len(spec.params)


# ---------- 任务 4.4：失败隔离 ----------


def test_partial_failure_isolated(tmp_path: Path) -> None:
    """单算子失败不得影响其余算子。"""
    root = tmp_path / "sub-mixed"
    (root / "operators" / "exp").mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(
        """
submit_id: sub-mixed
operators:
  - name: Exp
    entry: exp
    path: operators/exp/kernel.py
  - name: Gelu
    entry: gelu
    path: operators/gelu/kernel.py
""",
        encoding="utf-8",
    )
    # Exp 正常
    (root / "operators" / "exp" / "kernel.py").write_text(
        "def exp(x, base=-1.0, scale=1.0, shift=0.0):\n    return x\n", encoding="utf-8"
    )
    # Gelu 缺入口函数
    (root / "operators" / "gelu").mkdir(parents=True)
    (root / "operators" / "gelu" / "kernel.py").write_text(
        "def wrong(x):\n    return x\n", encoding="utf-8"
    )

    sub = load_all(discover(root))
    assert "Exp" in sub.loaded, "正常的算子必须仍被加载"
    assert "Gelu" in sub.failures, "失败的算子必须被记录"
    assert len(sub.loaded) == 1
    assert len(sub.failures) == 1


def test_failures_never_raise(tmp_path: Path) -> None:
    """load_all 不得因单算子失败而抛异常。"""
    root = tmp_path / "sub"
    root.mkdir()
    (root / MANIFEST_NAME).write_text(
        """
submit_id: sub
operators:
  - name: A
    entry: a
    path: operators/a/kernel.py
""",
        encoding="utf-8",
    )
    sub = discover(root)
    load_all(sub)  # 不抛异常
    assert "A" in sub.failures


# ---------- 任务 4.5：不可变 ----------


def test_snapshot_detects_modification(good_submission: Path) -> None:
    sub = discover(good_submission)
    before = sub.snapshot()
    (good_submission / "operators" / "exp" / "kernel.py").write_text(
        "def exp(x, base=1.0, scale=1.0, shift=0.0):\n    return x\n", encoding="utf-8"
    )
    after = sub.snapshot()
    changed = verify_unchanged(before, after)
    assert len(changed) == 1


def test_snapshot_no_change_when_untouched(good_submission: Path) -> None:
    sub = discover(good_submission)
    before = sub.snapshot()
    assert verify_unchanged(before, sub.snapshot()) == []


# ---------- 任务 3.3 相关：kernel 必须是真实文件 ----------


def test_kernel_must_be_real_file_not_inline(tmp_path: Path) -> None:
    """@triton.jit 依赖 inspect.getsourcelines，实现必须来自真实文件。"""
    from triton_eval.submission import load_module

    p = tmp_path / "k.py"
    p.write_text("VALUE = 1\n", encoding="utf-8")
    mod = load_module(p, "inline_test")
    assert mod.VALUE == 1
