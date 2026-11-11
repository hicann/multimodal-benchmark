# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""在不可信代码边界上的异常兜底。

为什么需要单独一个模块：
    提交方代码是不可信的，而 `SystemExit` / `KeyboardInterrupt` 继承自
    **BaseException** 而不是 Exception。所有 `except Exception` 的兜底都拦不住
    它们——于是一份在 import 期调用 `sys.exit()` / `exit()` / `raise SystemExit`
    的提交，会让整个评测进程**静默消失**：无告警、无报告、退出码 0。

    实测（ADV-E3）：`validate` 与 `run` 均无输出且 rc=0，
    CI 会把「评测被攻击掉了」当成评测成功。这是提交闭环最致命的一类缺陷：
    不是判错，是根本没有判定。

    这类漏洞的特点是无法靠"记得写 except BaseException"避免——
    边界散落在加载、校验、执行、CLI 顶层四处，漏一处就前功尽弃。
    故收敛到一个函数，所有边界统一调用。

用法：
    from .safeexec import guard_call

    try:
        guard_call(lambda: load_module(...))
    except Exception as exc:      # BaseException 已被转成 SubmissionEscape
        ...
"""

from __future__ import annotations

from typing import Any, Callable


class SubmissionEscape(Exception):
    """提交方代码试图跳出评测控制流。

    把不属于普通异常或操作者中断的 BaseException 转成普通 Exception，
    使其能被既有的 `except Exception` 兜底捕获并记录为违规——
    否则它们会穿透所有兜底，直接终止进程。
    """

    def __init__(self, origin: BaseException) -> None:
        self.origin = origin
        super().__init__(
            f"提交方代码试图终止进程（{type(origin).__name__}：{origin}）；"
            "这是不可信代码的越界行为，已作为违规处理"
        )


def guard_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """调用不可信代码，把越界的 BaseException 转为 SubmissionEscape。

    KeyboardInterrupt 不转换而是原样抛出：那是**操作者**的中断意图，
    不是提交方行为，吞掉会让 Ctrl-C 失效。
    """
    try:
        return func(*args, **kwargs)
    except BaseException as exc:
        if isinstance(exc, (Exception, KeyboardInterrupt)):
            raise
        raise SubmissionEscape(exc) from exc


def is_escape(exc: BaseException) -> bool:
    """判定一个异常是否为「试图跳出控制流」的越界行为。"""
    return isinstance(exc, SubmissionEscape) or (
        isinstance(exc, BaseException) and not isinstance(exc, (Exception, KeyboardInterrupt))
    )
