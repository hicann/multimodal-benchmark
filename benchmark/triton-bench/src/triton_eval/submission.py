# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""用户提交的目录契约与发现机制（L0，无 NPU 依赖）。

目录式提交，零构建：

    submissions/<submit-id>/
    ├── manifest.yaml
    └── operators/<op>/kernel.py

为什么不用 wheel（参考仓库的做法）：
    triton kernel 是纯 Python 源文件，无编译环节。为其引入
    「建虚拟包 → 打包 → 安装 → 导入」是为零收益增加失败点。

manifest 示例：

    submit_id: demo-001
    operators:
      - name: Exp
        entry: exp
        path: operators/exp/kernel.py
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel, ConfigDict, Field

from .safeexec import guard_call

MANIFEST_NAME = "manifest.yaml"
OPERATORS_DIRNAME = "operators"


class OperatorEntry(BaseModel):
    """清单中的一条算子声明。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="算子名，须与注册表一致")
    entry: str = Field(description="入口函数名")
    path: str = Field(description="实现文件相对路径")


class SubmissionManifest(BaseModel):
    """提交清单。"""

    model_config = ConfigDict(extra="forbid")

    submit_id: str
    operators: list[OperatorEntry] = Field(default_factory=list)


class LoadedOperator:
    """一个已加载的算子实现。"""

    def __init__(self, name: str, entry: str, func: object, module: ModuleType,
                 path: Path) -> None:
        self.name = name
        self.entry = entry
        self.func = func
        self.module = module
        self.path = path

    def __repr__(self) -> str:
        return f"<LoadedOperator {self.name}.{self.entry}{self.signature()}>"

    def signature(self) -> list[str]:
        try:
            return list(inspect.signature(self.func).parameters)
        except (TypeError, ValueError):
            return []


class Submission:
    """一次提交。"""

    def __init__(self, root: Path, manifest: SubmissionManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.loaded: dict[str, LoadedOperator] = {}
        # 算子名 -> 失败原因；与已加载的相互独立
        self.failures: dict[str, str] = {}

    @property
    def submit_id(self) -> str:
        return self.manifest.submit_id

    def operator_names(self) -> list[str]:
        return [o.name for o in self.manifest.operators]

    def sources(self) -> list[Path]:
        return [self.root / o.path for o in self.manifest.operators]

    def snapshot(self) -> dict[Path, bytes]:
        """记录提交目录当前内容的摘要，用于验证评测期间不可变。

        排除 `__pycache__`（import 的解释器副产物，不是提交方写的内容）：
        否则第一次评测后每个 kernel 目录都会多出 .pyc，
        快照指纹跨轮漂移、不可变核查永远误报。
        """
        out: dict[Path, bytes] = {}
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            out[p] = p.read_bytes()
        return out


def load_manifest(root: Path) -> SubmissionManifest:
    """读取并解析提交清单。缺文件必须明确报出文件名。"""
    import yaml

    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"提交缺少清单文件：{MANIFEST_NAME}（位于 {root}）")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SubmissionManifest.model_validate(raw)


def discover(root: Path) -> Submission:
    """发现并返回一个提交对象。此时不做加载，只解析清单。"""
    from .fixture_sources import fixture_path

    root = fixture_path(Path(root))
    if not root.is_dir():
        raise FileNotFoundError(f"提交目录不存在：{root}")
    return Submission(root, load_manifest(root))


def load_module(path: Path, name: str) -> ModuleType:
    """从真实文件路径加载模块。

    @triton.jit 依赖 inspect.getsourcelines 反查源码，
    因此实现必须来自真实文件，不能是内联字符串或数据库字段。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {path} 加载模块")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_operator(sub: Submission, entry: OperatorEntry) -> LoadedOperator:
    """加载单个算子实现。任何失败都抛异常，由调用方归入该算子的失败。"""
    path = sub.root / entry.path
    if not path.is_file():
        raise FileNotFoundError(f"算子 {entry.name} 的实现文件不存在：{entry.path}")

    mod_name = f"te_sub_{sub.submit_id}_{entry.name}".replace("-", "_")
    mod = load_module(path, mod_name)

    func = getattr(mod, entry.entry, None)
    if func is None:
        raise AttributeError(
            f"算子 {entry.name} 缺少入口函数 {entry.entry}；"
            f"文件 {entry.path} 中未找到该名称"
        )
    if not callable(func):
        raise TypeError(f"算子 {entry.name} 的 {entry.entry} 不可调用")
    return LoadedOperator(entry.name, entry.entry, func, mod, path)


def load_all(sub: Submission) -> Submission:
    """加载提交中的全部算子。

    单算子失败必须隔离：不得影响其余算子的加载与评测。
    """
    for entry in sub.manifest.operators:
        try:
            # 必须走 guard_call：普通 except Exception 拦不住 SystemExit，
            # 而提交代码的 import 期可以随便调用 sys.exit()，
            # 那会让整个评测进程静默消失（实测：无输出、退出码 0）。
            op = guard_call(load_operator, sub, entry)
            sub.loaded[entry.name] = op
        except Exception as exc:  # noqa: BLE001
            sub.failures[entry.name] = f"{type(exc).__name__}: {exc}"
    return sub


def verify_unchanged(before: dict[Path, bytes], after: dict[Path, bytes]) -> list[str]:
    """比对两次快照，返回发生变化的文件列表。"""
    changed: list[str] = []
    for p in sorted(set(before) | set(after)):
        if before.get(p) != after.get(p):
            changed.append(str(p))
    return changed
