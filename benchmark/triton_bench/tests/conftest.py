# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""pytest 配置。

核心作用：让 `npu` 标记的测试在没有昇腾 NPU 的机器上自动跳过，而不是失败。

这样同一份测试套件在两种环境下都能直接跑：
    本地 macOS      —— npu 用例跳过，其余全部执行
    昇腾执行机      —— 全部执行
不需要调用方记得加 `-m "not npu"`。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


sys.path.extend(str(Path(__file__).parent / group)
                for group in ("core", "coverage", "runtime", "submissions"))


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def triton_available() -> bool:
    """需要 triton 的用例：例如要真正 import 一份 kernel 实现。

    与 npu 标记分开：这些用例不需要昇腾硬件，只需要被测编译器在位。
    """
    return importlib.util.find_spec("triton") is not None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_npu = pytest.mark.skip(reason="需要真实昇腾 NPU（当前环境无 triton）")
    skip_torch = pytest.mark.skip(reason="需要 torch（当前环境未安装）")
    skip_triton = pytest.mark.skip(reason="需要 triton（当前环境未安装被测编译器）")
    for item in items:
        if item.get_closest_marker("npu") and not triton_available():
            item.add_marker(skip_npu)
        if item.get_closest_marker("triton") and not triton_available():
            item.add_marker(skip_triton)
        # 只需要 torch、不需要 NPU 的用例（如参考实现自检）单独标记：
        # 它们必须在有 torch 的机器上跑起来，否则参考实现只能等到
        # 昇腾机上才第一次被执行。
        if item.get_closest_marker("torch") and not _torch_available():
            item.add_marker(skip_torch)
        # 静态阶段不需要 triton，但加载阶段会真正 import 实现文件；
        # 同一个文件里两类用例并存，故按用例单独打标记。
