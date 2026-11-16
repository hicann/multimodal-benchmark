# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""监督者的测试（L0，用假子进程——不需要 torch / triton / NPU）。

守护什么：
    监督者的三件硬事：事件流转发、硬超时（SIGTERM→宽限→SIGKILL）、
    崩溃画像（退出码 / 信号名 / stderr 尾行）。
    用假子进程（stub 脚本）把它们钉在本地可测——真实子进程
    （evalchild）需要昇腾环境，靠昇腾机验证。

    真实性与测试性的接缝：TRITON_EVAL_CHILD_CMD 环境变量允许
    替换子进程命令，与 cli._child_cmd 的行为一致。
"""

from __future__ import annotations

import json
import signal
import sys
import textwrap
from pathlib import Path

from triton_eval.supervisor import run_child

# 假子进程的骨架：把「事件序列」参数化，其余行为（挂死 / 吞信号 /
# 打印垃圾 / 崩溃）按需叠加
STUB = '''
import json, sys, time

events_path = sys.argv[1]
mode = sys.argv[2]

def emit(kind, **payload):
    with open(events_path, "a") as fh:
        fh.write(json.dumps({"event": kind, **payload}) + "\\n")
        fh.flush()

if mode == "garbage":
    with open(events_path, "a") as fh:
        fh.write("这不是 JSON\\n")
        fh.flush()
emit("meta", validation_ok=True, operators=["Exp"])
emit("probe", ok=True, reason="")
emit("case", record={"run_id": 1, "case_code": "exp-f32-00", "operator": "Exp",
                     "status": "PASS"})
if mode == "hang":
    # 吞掉 SIGTERM（不可信代码可以做到），然后赖着不走——
    # 这正是监督者必须 SIGKILL 兜底的原因（ADV-F4 的进程级版本）
    import signal as s
    s.signal(s.SIGTERM, s.SIG_IGN)
    time.sleep(600)
if mode == "crash":
    sys.stderr.write("stub is crashing\\n")
    sys.exit(3)
emit("case", record={"run_id": 1, "case_code": "exp-f16-01", "operator": "Exp",
                     "status": "PASS"})
emit("done", reason="ok")
'''


def _write_stub(tmp_path: Path) -> Path:
    p = tmp_path / "stub_child.py"
    p.write_text(textwrap.dedent(STUB), encoding="utf-8")
    return p


def _run(tmp_path: Path, mode: str, *, case_timeouts=None,
         idle_margin_s=2.0, grace_s=1.0):
    stub = _write_stub(tmp_path)
    events = tmp_path / "events.jsonl"
    events.touch()
    seen: list[dict] = []
    outcome = run_child(
        [sys.executable, str(stub), str(events), mode],
        events_path=events,
        case_timeouts=case_timeouts or [60],
        on_event=seen.append,
        stdout_file=tmp_path / "child-stdout.log",
        stderr_file=tmp_path / "child-stderr.log",
        idle_margin_s=idle_margin_s,
        grace_s=grace_s,
    )
    return outcome, seen


def test_happy_path_streams_all_events(tmp_path: Path) -> None:
    outcome, seen = _run(tmp_path, "normal")
    kinds = [e["event"] for e in seen]
    assert kinds == ["meta", "probe", "case", "case", "done"]
    assert seen[2]["record"]["case_code"] == "exp-f32-00"
    assert outcome.exit_code == 0
    assert not outcome.killed_by_us
    assert outcome.signal_name is None


def test_unparsed_lines_do_not_break_protocol(tmp_path: Path) -> None:
    outcome, seen = _run(tmp_path, "garbage")
    kinds = [e["event"] for e in seen]
    assert kinds == ["meta", "probe", "case", "case", "done"], "垃圾行必须被跳过而不是断流"
    assert outcome.unparsed_lines and "这不是 JSON" in outcome.unparsed_lines[0]


def test_hanging_child_is_killed(tmp_path: Path) -> None:
    """吞掉 SIGTERM 还赖着的子进程，必须被 SIGKILL 终结。"""
    outcome, seen = _run(tmp_path, "hang", case_timeouts=[1],
                         idle_margin_s=1.0, grace_s=1.0)
    kinds = [e["event"] for e in seen]
    assert "meta" in kinds, "挂死前的事件必须已被收下（部分结果恢复）"
    assert "done" not in kinds
    assert outcome.killed_by_us
    assert outcome.signal_name == "SIGKILL"
    assert outcome.exit_code == -signal.SIGKILL


def test_crashing_child_is_diagnosed(tmp_path: Path) -> None:
    outcome, seen = _run(tmp_path, "crash")
    kinds = [e["event"] for e in seen]
    assert kinds == ["meta", "probe", "case"], "崩溃前的事件保留"
    assert outcome.exit_code == 3
    assert not outcome.killed_by_us
    assert any("stub is crashing" in line for line in outcome.stderr_tail), \
        "stderr 尾行是崩溃归因的线索"


def test_child_command_override_via_env(tmp_path: Path, monkeypatch) -> None:
    """TRITON_EVAL_CHILD_CMD 接缝可用（测试/调试时换假子进程）。"""
    from triton_eval.cli import _child_cmd

    monkeypatch.setenv("TRITON_EVAL_CHILD_CMD", "/bin/echo hi")
    assert _child_cmd() == ["/bin/echo", "hi"]
    monkeypatch.delenv("TRITON_EVAL_CHILD_CMD")
    cmd = _child_cmd()
    assert cmd[-1] == "triton_eval.evalchild"
    assert cmd[0] == sys.executable


def test_evalchild_module_imports_without_heavy_deps() -> None:
    """子进程入口必须保持「零重依赖可导入」——它起不来时唯一线索
    是父进程的报错，import 错误会掩盖真正的故障。
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "triton_eval.evalchild", "--help"],
        capture_output=True, text=True, timeout=60,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
             "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "run-dir" in proc.stdout
