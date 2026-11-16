# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""shell 脚本的字节级回归钉（L0，无需 NPU）。

背景（2026-10-12 真机排障时实测穿帮）：bash 在部分上下文里，
"$var" 后紧跟多字节字符（全角括号等）会解析错乱——实测
echo "结束（rc=$rc）" 输出成 "rc=" 加残缺括号：$rc 展开为空、
括号首字节被吞（同一写法在顶层偶尔正常，函数内必错，靠行为区分
不可靠）。修复口径：变量接多字节字符时一律写 ${var}。
本测试按字节扫描全部 shell 脚本，命中即失败，防止地雷回埋。
"""

from __future__ import annotations

import re
from pathlib import Path

# $name 后紧跟多字节字符首字节（>=0x80）——全角括号、中文等
_VAR_BEFORE_MULTIBYTE = re.compile(rb"\$[A-Za-z_][A-Za-z0-9_]*[\x80-\xff]")

_ROOT = Path(__file__).resolve().parents[2]
SHELL_SOURCES = sorted(_ROOT.glob("scripts/*.sh")) + sorted(_ROOT.glob("env/*.sh"))


def test_no_variable_directly_before_multibyte_char() -> None:
    offenders: list[str] = []
    for fn in SHELL_SOURCES:
        for lineno, line in enumerate(fn.read_bytes().splitlines(), 1):
            for m in _VAR_BEFORE_MULTIBYTE.finditer(line):
                snippet = line[max(0, m.start() - 8):m.end() + 10]
                offenders.append(f"{fn.name}:{lineno}: …{snippet.decode('utf-8', 'replace')}…")
    assert not offenders, (
        "shell 脚本里 $var 后紧跟多字节字符——bash 解析缺陷会吞掉变量展开"
        "并损坏后续字符，接全角字符的变量必须写 ${var}：\n" + "\n".join(offenders)
    )
