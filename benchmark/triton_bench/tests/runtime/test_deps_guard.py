# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""依赖清单守卫测试。

这些测试守护的是实测得出的硬约束，不是风格偏好：

  triton 与 triton-ascend 安装目录同名。若评测工具链依赖中出现社区 triton，
  pip 会覆盖被测编译器的目录并静默破坏被测对象（实测症状：
  `ImportError: cannot import name 'getenv' from 'triton._C.libtriton'`）。

因此「评测工具链依赖里不许有 triton」必须由机器校验，不能靠注释或人工检查。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQ_EVAL = PROJECT_ROOT / "env" / "requirements-eval.txt"
REQ_COMPILER = PROJECT_ROOT / "env" / "requirements-compiler.txt"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def requirement_names(path: Path) -> list[str]:
    """按 pip 的规则解析出依赖名：跳过注释行、空行与命令行选项。"""
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # 取版本说明符、extras、环境变量标记之前的部分作为包名
        name = re.split(r"[<>=!~\[; ]", line, 1)[0].strip()
        if name:
            names.append(name)
    return names


def test_requirements_files_exist() -> None:
    assert REQ_EVAL.is_file(), f"缺少 {REQ_EVAL}"
    assert REQ_COMPILER.is_file(), f"缺少 {REQ_COMPILER}"


def test_eval_requirements_do_not_include_triton() -> None:
    """评测工具链依赖中不得出现 triton（会覆盖被测编译器目录）。"""
    names = [n.lower() for n in requirement_names(REQ_EVAL)]
    assert "triton" not in names, (
        "env/requirements-eval.txt 出现 triton。它会覆盖 triton-ascend 的同名安装目录，"
        "静默破坏被测编译器。评测工具链不得依赖 triton。"
    )


def test_pyproject_dependencies_do_not_include_triton() -> None:
    """pyproject 的运行时依赖同样不得出现 triton。"""
    text = PYPROJECT.read_text(encoding="utf-8")
    deps_block = text.split("[project]", 1)[-1].split("[project.optional-dependencies]", 1)[0]
    declared = re.findall(r'"([^"]+)"', deps_block)
    names = [re.split(r"[<>=!~\[; ]", d, 1)[0].strip().lower() for d in declared]
    assert "triton" not in names, (
        "pyproject.toml 的 dependencies 中出现 triton，会覆盖被测编译器目录。"
    )


def test_compiler_requirements_pin_triton_ascend() -> None:
    """被测编译器依赖必须显式锁定 triton-ascend 版本。"""
    names = [n.lower() for n in requirement_names(REQ_COMPILER)]
    assert "triton-ascend" in names, "被测编译器依赖中缺少 triton-ascend"
    pinned = [
        line
        for line in REQ_COMPILER.read_text(encoding="utf-8").splitlines()
        if line.strip().lower().startswith("triton-ascend") and "==" in line
    ]
    assert pinned, "triton-ascend 必须使用 == 锁定版本，否则环境不可复现"


def test_eval_requirements_installable_names_are_known() -> None:
    """评测工具链依赖应保持极小集合，且都是跨平台可用的包。"""
    allowed = {"pydantic", "pyyaml", "jinja2", "pytest", "numpy"}
    names = {n.lower() for n in requirement_names(REQ_EVAL)}
    unexpected = names - allowed
    assert not unexpected, f"评测工具链出现非预期依赖：{sorted(unexpected)}"


@pytest.mark.npu
def test_compiler_is_usable_in_current_env() -> None:
    """需要昇腾主机：被测编译器在当前环境下必须可用且版本可测。

    关于「是否必须在 venv 目录内」的设计取舍（用户已决策：承认现状）：
        三个昇腾环境都由平台预装 triton-ascend 到系统 Python，
        venv 以 --system-site-packages 继承。强行在 venv 内重装会
        引入 CANN 配套风险，收益（路径内聚）小于风险。
        因此本测试只校验「可加载 + 版本可测」，不校验加载路径。
        可追溯性由 env_fp 中的 venv_path + venv_lock_hash 保证。
    """
    import triton

    assert triton.__version__, "拿不到 triton 版本，说明被测编译器环境有问题"
    # 版本必须是实测值：triton-ascend 会接管 triton 目录，
    # pip 元数据与实际值在实测环境下并不一致（3.5.0 vs 3.2.0）
    assert isinstance(triton.__version__, str)


@pytest.mark.npu
def test_triton_path_is_recorded_for_provenance() -> None:
    """被测编译器加载路径必须可被记录，用于环境指纹溯源。"""
    import triton

    assert triton.__file__, "无法定位 triton 加载路径，环境指纹将缺少溯源信息"
