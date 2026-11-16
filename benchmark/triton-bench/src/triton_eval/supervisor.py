# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""评测子进程监督者（L0，不依赖 torch / triton / NPU）。

为什么必须有这一层（实测教训 + cann-bench 的工程结论）：

1. **SIGALRM 不是硬超时。** 它注入的是异常，被测代码一句
   `except BaseException: 赖着` 就能吞掉（ADV-F4 实测：吞掉后多赖 15s
   照常判 PASS）。进程级 terminate → 宽限 → kill 才是硬超时。
2. **不可信代码不该与评测主进程同室。** 用户实现与 oracle、数据库、
   评测状态同进程时，一个 setattr 就能劫持判定权（ADV-D7）。
   子进程化之后崩溃不再波及主进程，状态也不跨提交泄漏。
3. **崩溃要能归因，结果要能抢救。** 子进程被信号杀死时
   （kernel SIGSEGV、OOM kill），死因（信号名 + stderr 尾行）必须
   留痕，已产出的事件照常落库——部分结果恢复。

事件通道为什么走文件（run_dir/events.jsonl）而不是子进程 stdout：
    子进程的 stdout/stderr 是诊断信息（torch_npu 警告、kernel 打印、
    提交方代码的 print）。若事件协议复用 stdout，提交方一行 print
    就能伪造事件行。协议走独立文件 + 子进程对 fd 的重定向进日志，
    stdout 上的任何内容都只当诊断对待。
"""

from __future__ import annotations

import json
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# 每个用例的事件预算 = 该用例的 timeout_s + 余量。
# 余量必须盖住冷编译（实测昇腾冷编译约 22s）、探针、参考实现与
# 性能计时（repeats 轮）——把它压小会把慢用例误杀成超时。
IDLE_MARGIN_S = 180.0
# SIGTERM 后的宽限：给子进程把手上已有的事件刷盘、走异常兜底的机会
GRACE_S = 10.0
POLL_S = 0.1


@dataclass
class ChildOutcome:
    """子进程的终局画像。"""

    exit_code: int | None = None
    signal_name: str | None = None    # returncode < 0 时的信号名（SIGSEGV 等）
    killed_by_us: bool = False         # 是否由监督者执行了 terminate/kill
    stderr_tail: list[str] = field(default_factory=list)
    unparsed_lines: list[str] = field(default_factory=list)


def _reader(pipe, sink: Callable[[str], None], sentinel: Callable[[], None]) -> None:
    try:
        for line in pipe:
            sink(line)
    finally:
        sentinel()


def run_child(
    cmd: list[str],
    *,
    events_path: Path,
    case_timeouts: list[int],
    on_event: Callable[[dict], None],
    stdout_file: Path,
    stderr_file: Path,
    idle_margin_s: float = IDLE_MARGIN_S,
    grace_s: float = GRACE_S,
) -> ChildOutcome:
    """启动并监督评测子进程直到退出。

    事件从 events_path 尾部读取（子进程逐行 append+flush）；
    stdout/stderr 由本函数持续抽走落盘（不读会撑爆管道缓冲区，
    子进程会阻塞在写上）。
    """
    for p in (stdout_file, stderr_file, events_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    outcome = ChildOutcome()

    def _on_stdout(line: str) -> None:
        if line.strip():
            with open(stdout_file, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(line)

    def _on_stderr(line: str) -> None:
        outcome.stderr_tail.append(line.rstrip("\n"))
        del outcome.stderr_tail[:-50]  # 只留尾行：归因要的是线索，不是全量
        with open(stderr_file, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line)

    # 只负责持续抽干管道（不读会撑爆缓冲区，子进程会阻塞在写上），
    # 退出探测走 proc.poll()，不依赖线程间通信
    t_out, t_err = _start_pipe_readers(proc, _on_stdout, _on_stderr)

    def _budget(case_idx: int) -> float:
        t = case_timeouts[case_idx] if case_idx < len(case_timeouts) else 60
        return float(t) + idle_margin_s

    def _escalate(sig: int) -> None:
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass  # 已死：reap 即可

    try:
        _monitor_events(proc, events_path, outcome, _budget, _escalate, on_event, grace_s)
    finally:
        _reap_child(proc, outcome, _escalate, t_out, t_err)

    return outcome


def _reap_child(proc, outcome, escalate, t_out, t_err):
    try:
        proc.wait(timeout=GRACE_S + 30)
    except subprocess.TimeoutExpired:
        escalate(signal.SIGKILL)
        proc.wait(timeout=10)
    outcome.exit_code = proc.returncode
    if proc.returncode is not None and proc.returncode < 0:
        try:
            outcome.signal_name = signal.Signals(-proc.returncode).name
        except ValueError:
            outcome.signal_name = f"SIG{-proc.returncode}"
    t_out.join(timeout=5)
    t_err.join(timeout=5)


def _start_pipe_readers(proc, _on_stdout, _on_stderr):
    t_out = threading.Thread(
        target=_reader, args=(proc.stdout, _on_stdout, lambda: None), daemon=True
    )
    t_err = threading.Thread(
        target=_reader, args=(proc.stderr, _on_stderr, lambda: None), daemon=True
    )
    t_out.start()
    t_err.start()
    return t_out, t_err


@dataclass
class _MonitorState:
    case_idx: int = 0
    killed: bool = False
    kill_rounds: int = 0
    deadline: float = 0.0


def _poll_event_file(events_fh, proc, state, outcome, budget, escalate, grace_s):
    line = events_fh.readline()
    if line:
        return line, "process"
    now = time.monotonic()
    if now >= state.deadline:
        if not state.killed:
            state.killed = True
            outcome.killed_by_us = True
            escalate(signal.SIGTERM)
            state.deadline = now + grace_s
        else:
            state.kill_rounds += 1
            if state.kill_rounds > 2:
                return line, "break"
            escalate(signal.SIGKILL)
            state.deadline = now + grace_s
            return line, "continue"
    if proc.poll() is not None:
        time.sleep(POLL_S)
        line = events_fh.readline()
        return line, "process" if line else "break"
    time.sleep(POLL_S)
    return line, "continue"


def _consume_event_line(line, state, outcome, budget, on_event):
    stripped = line.strip()
    if not stripped:
        return
    try:
        evt = json.loads(stripped)
    except json.JSONDecodeError:
        outcome.unparsed_lines.append(stripped[:200])
        state.deadline = time.monotonic() + budget(state.case_idx)
        return
    if not (isinstance(evt, dict) and "event" in evt):
        outcome.unparsed_lines.append(stripped[:200])
        state.deadline = time.monotonic() + budget(state.case_idx)
        return
    if evt.get("event") == "case":
        state.case_idx += 1
    state.deadline = time.monotonic() + budget(state.case_idx)
    try:
        on_event(evt)
    except Exception:  # noqa: BLE001
        outcome.unparsed_lines.append(f"<on_event failed for {evt.get('event')}>")


def _monitor_events(proc, events_path, outcome, budget, escalate, on_event, grace_s):
    with open(events_path, "r", encoding="utf-8", errors="replace") as events_fh:
        state = _MonitorState(deadline=time.monotonic() + budget(0))
        while True:
            line, action = _poll_event_file(events_fh, proc, state, outcome, budget, escalate, grace_s)
            if action == "break":
                break
            if action == "continue":
                continue
            _consume_event_line(line, state, outcome, budget, on_event)
