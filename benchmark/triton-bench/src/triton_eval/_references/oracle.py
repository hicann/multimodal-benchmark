# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现的加载（评测方资产，L0）。

参考实现必须与提交实现**来源分离**——否则就成了自己验证自己。
它放在冻结 corpus 的 `operators/<entry>/ref.py`（主 oracle，同设备执行）
与 `ref_cpu.py`（第二参考，CPU 路径，用于交叉校验）里，
由评测方提供，绝不能从提交目录加载。

单独成模块而不是 CLI 私有函数的原因：
    测试与其它调用方同样需要「按算子名取参考实现」，
    藏在 cli 里会导致测试重写一遍加载逻辑，两份实现迟早不一致。

ref.py 的可选契约：
    reference(*inputs, **attrs) -> Tensor
    make_inputs(*, seed, dtype, shape, device, attrs) -> list[Tensor]  可选

`make_inputs` 只对多输入算子需要：一个 shape 只能描述一个张量，
第二个输入的范围（如 gather 的 index 必须落在 [0, size(dim))）
无法从主输入 shape 推出。缺失时执行器回落到单张量默认构造。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from types import ModuleType
from typing import Any, Callable
from . import PROJECT_ROOT

DEFAULT_CORPUS = PROJECT_ROOT / "corpus" / "v0.3-multi-input-frozen"

# 当前生效的语料目录。参考实现是**随语料版本冻结的资产**（新版本会有
# 新算子的 ref.py），必须与评测所用的语料一致。
# 历史缺陷（2026-10-13 修复）：ref_path 此前固定指向 DEFAULT_CORPUS，
# 运行 v0.4 用例时参考静默从 v0.3 加载——两版参考内容恰好相同所以无害，
# 但 v0.5 起新算子的参考只存在于 v0.5 目录，会全体报"缺少 ref.py"。
# 评测主流程（evalchild）在验完清单后必须调用 configure_corpus。
_active_corpus: Path = DEFAULT_CORPUS

# 参考实现加载失败的真实原因（模块名 → 异常摘要）。
# 之所以要存：_load_module 对外只返回 None，而子进程的 stderr 被 supervisor
# 接管、不会进主运行日志，导致「缺少参考实现」永远查不出真因。
_LOAD_ERRORS: dict[str, str] = {}


def configure_corpus(corpus_dir: Path | str) -> None:
    """设置当前生效的语料目录，并清空参考实现的加载缓存。

    必须在加载任何参考实现之前调用（evalchild 在验完 manifest 后调用；
    构建脚本在跑参考自检前调用）。幂等：重复设置同一目录无副作用。
    """
    global _active_corpus
    _active_corpus = Path(corpus_dir)
    # 参考按 (语料目录, 算子) 解析——切目录后旧缓存全部作废
    _ref_module.cache_clear()
    reference_for.cache_clear()
    make_inputs_for.cache_clear()
    cross_reference_for.cache_clear()


def active_corpus() -> Path:
    """当前生效的语料目录（供测试与诊断断言用）。"""
    return _active_corpus


def _entry(operator: str) -> str | None:
    from .operator_registry import by_name

    spec = by_name(operator)
    return spec.entry if spec else None


def ref_path(operator: str) -> Path | None:
    """主参考实现的文件路径；算子未注册时返回 None。

    目录命名约定在不同 registry 之间**并不一致**，不能只认 entry：
      * 主 registry（v0.3~v0.5）：目录名 = entry（如 `exp`）；
      * ascend_ops（v0.6）：目录名 = source_name（如 `discrete_memory_access`），
        而 entry 是 kernel 函数名（`discrete_pick`）。
    只用 entry 硬拼会让 v0.6 的 4 个算子指向不存在的目录，文件查不到 →
    _load_module 静默返回 None → 被报成「缺少参考实现（评测方资产缺失）」，
    看上去像语料资产缺失，实则是路径拼错（实测 80 例因此误判）。
    这里取真实存在的那个目录。
    """
    return _reference_file(operator, "ref.py")


def cross_ref_path(operator: str) -> Path | None:
    """第二参考（CPU 路径）的文件路径；算子未注册时返回 None。"""
    return _reference_file(operator, "ref_cpu.py")


def _reference_file(operator: str, filename: str) -> Path | None:
    from .corpus_assets import resolve_reference_path
    from .operator_registry import by_name

    spec = by_name(operator)
    if spec is None:
        return None
    for name in (spec.entry, getattr(spec, "source_name", None)):
        if name:
            path = resolve_reference_path(_active_corpus, f"operators/{name}/{filename}")
            if path.is_file():
                return path
    return _active_corpus / "operators" / spec.entry / filename


def _load_module(path: Path | None, mod_name: str) -> ModuleType | None:
    """按路径加载模块。任何失败都返回 None，绝不抛出。

    参考实现缺失是「评测方资产不全」，不是被测方的问题，
    不能让一轮评测因为一个算子缺参考而整体崩掉。
    """
    if path is None or not path.is_file():
        return None
    from .submission import load_module

    try:
        return load_module(path, mod_name)
    except Exception as exc:  # noqa: BLE001
        # 必须留下真实原因：调用方只看得到 `return None`，对外统一报
        # 「缺少参考实现（评测方资产缺失）」。实测这一句掩盖了「教程模块 import
        # 失败」和「ref.py 引用了教程中不存在的函数」两类完全不同的问题。
        # 注意：打 stderr 没用——子进程的输出被 supervisor 接管，不会进主日志；
        # 必须存下来让调用方写进 compile_err，才能随 results.jsonl 带回来。
        _LOAD_ERRORS[mod_name] = f"{type(exc).__name__}: {exc}"
        return None


def oracle_load_error(operator: str) -> str | None:
    """返回该算子参考实现加载失败的原因（供诊断，正常时为 None）。"""
    return _LOAD_ERRORS.get(f"te_ref_{operator}")


def _attr(mod: ModuleType | None, name: str) -> Callable[..., Any] | None:
    if mod is None:
        return None
    func = getattr(mod, name, None)
    return func if callable(func) else None


@lru_cache(maxsize=64)
def _ref_module(operator: str) -> ModuleType | None:
    return _load_module(ref_path(operator), f"te_ref_{operator}")


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """给参考实现套一层隔离壳。

    为什么需要：
        参考实现是判定的锚。它一旦被替换，判定权就归了提交方。
        实测（ADV-D7）：提交代码在 import 期执行
            `torch.softmax = lambda *a, **k: torch.zeros_like(a[0])`
        于是 `reference_for("Softmax")` 的输出和从 2.0 变成 0.0——
        被测方只要跟着返回全零就能拿满分。静态扫描看不见这个
        （它没有调用任何被禁接口），只有运行期才暴露。

        根治要进程隔离（参考实现在独立进程/解释器中执行）；
        在单进程架构下，这里做**可验证的降级**：
        每次取参考时校验关键符号是否被换过，换过就判锚点失效，
        而不是拿一个被投毒的参考去判定他人。
    """
    import functools

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        tampered = _detect_module_tamper()
        if tampered:
            raise RuntimeError(
                f"参考实现所依赖的符号在运行期被替换（{tampered}）；"
                "判定锚点不可信，本用例判为参考失效"
            )
        return fn(*args, **kwargs)

    return wrapped


_TAMPER_PROBE: tuple[str, ...] = ("softmax", "exp", "matmul", "gather",
                                  "scatter", "argmax", "cummin", "maximum")


def _detect_module_tamper() -> str | None:
    """探测 torch 顶层的关键计算符号是否被替换。

    判据是「原始类型/模块归属」，不是身份相等——
    重绑定成同名的 lambda 会丢失 __module__，
    这是最廉价也最普适的一层探测。
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None

    for name in _TAMPER_PROBE:
        obj = getattr(torch, name, None)
        if obj is None or not callable(obj):
            continue
        module = getattr(obj, "__module__", "") or ""
        qualname = getattr(obj, "__qualname__", "") or ""
        # 正规 torch 函数的 __module__ 属于 torch.* 家族
        if module and not module.startswith("torch"):
            return f"torch.{name} 被替换为一个来自 {module} 的对象"
        # lambda / 局部函数：__module__ 可能仍是 torch，但 qualname 会露馅
        if "<lambda>" in qualname or "<locals>" in qualname:
            return f"torch.{name} 被替换为一个局部定义的对象"
    return None


@lru_cache(maxsize=64)
def reference_for(operator: str) -> Callable[..., Any] | None:
    """取算子的主参考实现。"""
    fn = _attr(_ref_module(operator), "reference")
    return _guard(fn) if fn is not None else None


@lru_cache(maxsize=64)
def make_inputs_for(operator: str) -> Callable[..., Any] | None:
    """取算子的输入构造函数（可选，多输入算子才需要）。"""
    return _attr(_ref_module(operator), "make_inputs")


@lru_cache(maxsize=64)
def cross_reference_for(operator: str) -> Callable[..., Any] | None:
    """取算子的第二参考实现（CPU 路径）。"""
    return _attr(_load_module(cross_ref_path(operator), f"te_refcpu_{operator}"),
                 "reference")
