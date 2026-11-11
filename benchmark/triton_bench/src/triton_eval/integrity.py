# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""运行时完整性校验（L0 定义 / L1 使用）。

两道防线，都来自被攻击的实测 + cann-bench 的工程结论：

1. **ApiGuard——关键 API 身份快照 → 校验 → 恢复。**
   静态扫描只看文本，`setattr(torch, "softmax", 假实现)` 换个写法
   （别名导入 / 链式属性 / 方法形态）就绕过（ADV-D1~D7 攻防结论）。
   运行期身份比对不看写法，只看结果：对象还是不是当初那个对象。
   快照必须在**任何提交代码 import 之前**完成，否则快照到的就是
   被篡改后的状态（快照点错=防线整体失效）。

2. **module_identity——子进程实际加载的模块 vs 父进程环境指纹。**
   「构建期环境与运行期环境错配」是 cann-bench 的真实事故：另一套
   CANN 被 build.sh 重新 source 后，kernel 与运行环境错配，只在
   profiler 下 SIGSEGV、极难归因。这里核对 torch / torch_npu / triton
   的实测版本：子进程加载的必须与指纹声称的一致。

白名单原则：宁可漏报，不可误报。名单里出现一条会被正常评测触发的
假阳性，整个校验就会失去公信力——出现误报时的正确动作是摘掉该条目
并记录原因，而不是放松整道防线。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

# ---------------- 关键 API 名单 ----------------
# (模块名, 属性路径)；属性路径可用点号深入类属性（torch.Tensor.exp）。
# 模块名必须是 sys.modules 里真实存在的模块——torch.Tensor 是类不是模块，
# 要写成 ("torch", "Tensor.exp")。写错的名字会被静默跳过（为了兼容可选
# API），所以 tests/runtime/test_integrity.py 钉住了关键条目必须被守护到。
# 维护守则：
#   - 只收「被替换就能改变判定或计时结果」的对象；
#   - 不收「评测自身会在运行中合法替换」的对象。
CRITICAL_APIS: list[tuple[str, str]] = [
    # 计时与同步：篡改这些可以凭空造出速度（ADV-D6 的攻击面）
    ("time", "perf_counter"), ("time", "perf_counter_ns"),
    ("time", "monotonic"), ("time", "time"),
    # NPU 同步与事件：do_bench 与评测都依赖
    ("torch.npu", "synchronize"), ("torch.npu", "Event"),
    # 主 oracle 参考实现用到的核心计算接口——
    # 篡改它们等于「期望值由提交方定义」（ADV-D7 实测穿透）
    ("torch", "softmax"), ("torch", "exp"), ("torch", "sigmoid"),
    ("torch", "matmul"), ("torch", "bmm"), ("torch", "mm"),
    ("torch", "cummin"), ("torch", "cumsum"), ("torch", "gather"),
    ("torch", "scatter"), ("torch", "amax"), ("torch", "sum"),
    ("torch", "mean"), ("torch", "max"), ("torch", "min"),
    ("torch", "argmax"), ("torch", "argmin"), ("torch", "erf"),
    ("torch", "transpose"), ("torch", "empty_like"), ("torch", "zeros_like"),
    ("torch", "equal"), ("torch", "isnan"), ("torch", "isinf"),
    ("torch", "isposinf"), ("torch", "isneginf"), ("torch", "isfinite"),
    ("torch", "allclose"),
    ("torch.nn.functional", "conv1d"), ("torch.nn.functional", "silu"),
    ("torch.nn.functional", "softmax"), ("torch.nn.functional", "gelu"),
    ("torch.nn.functional", "cross_entropy"),
    ("torch.nn.functional", "layer_norm"),
    # 比对链路上的张量方法（替换会伪造比较结果）。
    # 注意 torch.Tensor 不是模块而是属性，走 (模块, 点号属性路径) 形态
    ("torch", "Tensor.exp"), ("torch", "Tensor.softmax"),
    ("torch", "Tensor.mul"), ("torch", "Tensor.add"),
    ("torch", "Tensor.matmul"), ("torch", "Tensor.to"),
    ("torch", "Tensor.abs"), ("torch", "Tensor.max"),
    ("torch", "Tensor.item"), ("torch", "Tensor.clone"),
    ("torch", "Tensor.detach"),
    # 被测编译器入口与官方计时器
    ("triton", "jit"), ("triton.testing", "do_bench"),
]


def _resolve(module: Any, attr_path: str) -> Any:
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass(frozen=True)
class _Entry:
    module_name: str
    attr_path: str
    module: Any
    original: Any


class ApiGuard:
    """关键 API 的身份快照。

    verify() 返回被篡改的接口名列表（空 = 干净）；restore() 把被篡改的
    接口换回原始对象并返回已恢复的名字。校验与恢复分离：先校验再由
    调用方决定处置，恢复动作本身可被测试。
    """

    def __init__(self, entries: list[_Entry]) -> None:
        self._entries = entries

    @property
    def covered(self) -> list[str]:
        """实际纳入守护的接口名（受环境可用性影响，仅供诊断）。"""
        return [f"{e.module_name}.{e.attr_path}" for e in self._entries]

    @classmethod
    def snapshot(cls) -> "ApiGuard":
        """对名单逐一取身份快照。模块未加载则先加载（仍先于提交代码）。

        名单里的属性在当前环境不存在（如老版本 torch 没有 rms_norm）
        就跳过：守一个不存在的接口没有意义。
        """
        entries: list[_Entry] = []
        for module_name, attr_path in CRITICAL_APIS:
            mod = sys.modules.get(module_name)
            if mod is None:
                try:
                    mod = importlib.import_module(module_name)
                except Exception:  # noqa: BLE001
                    continue
            try:
                original = _resolve(mod, attr_path)
            except AttributeError:
                continue
            entries.append(_Entry(module_name, attr_path, mod, original))
        return cls(entries)

    def verify(self) -> list[str]:
        """返回被篡改的 `模块.属性` 列表；被删除也视为篡改。"""
        tampered: list[str] = []
        for e in self._entries:
            try:
                current = _resolve(e.module, e.attr_path)
            except AttributeError:
                tampered.append(f"{e.module_name}.{e.attr_path}（被删除）")
                continue
            if current is not e.original:
                tampered.append(f"{e.module_name}.{e.attr_path}")
        return tampered

    def restore(self) -> list[str]:
        """把被篡改的接口换回快照时的原始对象，返回已恢复的名字。"""
        restored: list[str] = []
        for e in self._entries:
            try:
                current = _resolve(e.module, e.attr_path)
            except AttributeError:
                current = None
            if current is e.original:
                continue
            try:
                obj = e.module
                parts = e.attr_path.split(".")
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], e.original)
                restored.append(f"{e.module_name}.{e.attr_path}")
            except Exception:  # noqa: BLE001
                # 恢复失败不抛：宁可有记录地继续，不让防线自己变成故障源
                restored.append(f"{e.module_name}.{e.attr_path}（恢复失败）")
        return restored


# ---------------- 环境一致性 ----------------

# 模块名 → 环境指纹里对应的版本字段
IDENTITY_TO_ENV_KEY: dict[str, str] = {
    "torch": "torch_version",
    "torch_npu": "torch_npu_version",
    "triton": "triton_import_version",
}


def module_identity() -> dict[str, dict[str, str]]:
    """当前进程实际加载的关键模块身份（版本 + 文件路径）。"""
    out: dict[str, dict[str, str]] = {}
    for name in IDENTITY_TO_ENV_KEY:
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = importlib.import_module(name)
            except Exception:  # noqa: BLE001
                out[name] = {"version": "", "file": ""}
                continue
        out[name] = {
            "version": getattr(mod, "__version__", "") or "",
            "file": getattr(mod, "__file__", "") or "",
        }
    return out


def check_identity(expected_env: dict[str, Any]) -> list[str]:
    """子进程实际加载的模块版本 vs 父进程环境指纹。

    只在「两边都有值」时比较：空值意味着该侧没采集到（如本机没有
    torch_npu），比较它会制造假警报。
    """
    identity = module_identity()
    mismatches: list[str] = []
    for mod_name, env_key in IDENTITY_TO_ENV_KEY.items():
        expected = str((expected_env or {}).get(env_key, "") or "")
        actual = identity.get(mod_name, {}).get("version", "")
        if expected and actual and expected != actual:
            mismatches.append(
                f"{mod_name}：环境指纹为 {expected}，本进程实际加载 {actual}"
            )
    return mismatches
