# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""安全加载 triton-ascend-ops 教程模块（hyphen 文件名无法 import，见 submission 侧副本）。

教程文件混合了两种 utils 引用：
  * best_practice/* 用 `from utils import ...`（顶层绝对导入）
  * basic/* 用 `from ..utils import ...`（相对导入，期望父包有 utils）

但仓库里 `utils.py` 仅位于 `best_practice/`。本加载器统一处理：
  * 把 utils.py（best_practice 那份，含 is_npu / next_power_of_2 等）注入为
    顶层 `utils`（满足绝对导入）与 `tutpkg.utils`（满足 `from ..utils` 相对导入）；
  * 用唯一模块名加载教程，且不触发 `if __name__ == "__main__": run()`。
内核函数体原样加载，**绝不修改**。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

# Ascend 扩展算子（get_element / extract_slice / insert_slice ...）只存在于
# triton.language.extra.cann.extension，**不会**自动挂到 tl 上（实测 tl 里
# 仅有 inline_asm_elementwise）。best_practice/006 写的是
# `extension.get_element(...)` 所以没事；而 op_extension/001、002 用的是旧写法
# `tl.get_element(...)` / `tl.extract_slice(...)`，于是报
# "module 'triton.language' has no attribute 'get_element'"
# （InsertSlice / ExtractSlice 共 40 例 RUNTIME_FAIL）。
# 教程内核函数体按约定原样加载、绝不修改，因此在加载前把扩展 API 补挂到 tl，
# 且不覆盖 tl 已有属性。
try:
    import triton.language as _tl
    import triton.language.extra.cann.extension as _ext

    for _n in dir(_ext):
        if not _n.startswith("_") and not hasattr(_tl, _n):
            setattr(_tl, _n, getattr(_ext, _n))
except Exception:  # noqa: BLE001  社区版 triton 没有该模块
    pass

_TUT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reference",
                 "triton-ascend-ops", "tutorial"))


def _resolve_utils(tutdir: str) -> str | None:
    cand = os.path.join(tutdir, "utils.py")
    if os.path.exists(cand):
        return cand
    # 仓库中 utils.py 仅位于 best_practice/；basic/op_extension 回退复用它
    cand = os.path.join(os.path.dirname(tutdir), "best_practice", "utils.py")
    return cand if os.path.exists(cand) else None


def load_tutorial(subpath: str):
    base = os.path.normpath(os.path.join(_TUT_ROOT, subpath))
    tutdir = os.path.dirname(base)
    if tutdir not in sys.path:
        sys.path.insert(0, tutdir)

    # 注入 utils：既是顶层 `utils`（best_practice 的 `from utils import`），
    # 也是 `tutpkg.utils`（basic 的 `from ..utils import`）。
    utils_path = _resolve_utils(tutdir)
    if utils_path:
        uspec = importlib.util.spec_from_file_location("tutorial_utils", utils_path)
        umod = importlib.util.module_from_spec(uspec)
        uspec.loader.exec_module(umod)
        sys.modules.setdefault("utils", umod)
        pkg = sys.modules.setdefault("tutpkg", types.ModuleType("tutpkg"))
        sys.modules["tutpkg.utils"] = umod

    modname = "tut_" + os.path.basename(subpath).replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(modname, base)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "tutpkg." + os.path.basename(tutdir)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod
