# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""非法提交样例集的回归测试（静态/加载阶段不需要 NPU）。

为什么要有一整个目录的样例，而不是只在单测里构造源码字符串：
    规则表里写着、单测里测着，不代表校验链路里真的生效。
    实测教训：detect_address_keyed_cache 定义了、单测也测了，
    但 scan_source 从没调用它——SUB-STR-003 是一条死规则，
    直到为每个规则建立真实目录后才暴露。

因此这里的断言全部走 `validate_submission`（真实入口），
而不是直接调 scan_source——测的是链路，不是零件。

每一条：目录必须被拒，且报出**对应**的规则编号。
「被拒」本身不够：报错编号会让提交方按错误的方向改。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

from triton_eval.validator import validate_submission

pytestmark = pytest.mark.torch

INVALID = fixture_path(Path(__file__).resolve().parents[2] / "examples" / "invalid-submissions")
DEMO = fixture_path(Path(__file__).resolve().parents[2] / "examples" / "demo-submission")

# 目录 → 期望命中的规则编号
CASES: dict[str, str] = {
    "SUB-STR-001": "SUB-STR-001",  # 内置接口代算
    "SUB-STR-002": "SUB-STR-002",  # 主机侧往返
    "SUB-STR-003": "SUB-STR-003",  # 地址键缓存
    "SUB-STR-004": "SUB-STR-004",  # 布局搬运越权
    "SUB-STR-005": "SUB-STR-005",  # 干预计时
    "SUB-STR-006": "SUB-STR-006",  # 源码不可解析
    "SUB-LOD-001": "SUB-LOD-001",  # 清单缺失
    "SUB-LOD-002": "SUB-LOD-002",  # 未登记算子
    "SUB-LOD-003": "SUB-LOD-003",  # 实现文件不存在
    "SUB-LOD-004": "SUB-LOD-004",  # 入口缺失
    "SUB-LOD-005": "SUB-LOD-005",  # 入口不可调用
    "SUB-LOD-006": "SUB-LOD-006",  # 参数个数不符
    "SUB-LOD-007": "SUB-LOD-007",  # 参数名不符
    "SUB-LOD-008": "SUB-LOD-008",  # 路径逃出提交目录
    "SUB-LOD-009": "SUB-LOD-009",  # 仅限位置参数
    "SUB-STR-007": "SUB-STR-007",  # 自行搜索启动配置
}


@pytest.mark.parametrize("dirname,rule_id", sorted(CASES.items()))
def test_invalid_submission_rejected_with_expected_rule(dirname: str, rule_id: str) -> None:
    sub_dir = INVALID / dirname
    assert sub_dir.is_dir(), f"缺少样例目录：{sub_dir}"

    result = validate_submission(sub_dir)
    assert not result.ok, f"{dirname} 必须被拒绝"
    ids = [e.rule_id for e in result.errors]
    assert rule_id in ids, (
        f"{dirname} 应命中 {rule_id}，实际报出 {ids}。"
        "报错编号会让提交方按错误的方向改。"
    )


@pytest.mark.parametrize("dirname,rule_id", sorted(CASES.items()))
def test_rejection_is_deterministic(dirname: str, rule_id: str) -> None:
    """同一份提交连续两次校验必须完全一致——否则提交方无从整改。"""
    a = validate_submission(INVALID / dirname)
    b = validate_submission(INVALID / dirname)
    assert [e.rule_id for e in a.errors] == [e.rule_id for e in b.errors]
    assert a.ok == b.ok


def test_rules_are_all_covered_by_a_sample() -> None:
    """规则表里每条静态/加载规则都必须有对应样例，不允许有规则无反例。"""
    from triton_eval.validator import RULES

    # 从规则表直接推导，而不是遍历一张需要手工同步的清单：
    # 新增规则时若忘了补样例，这里必须红。
    # 曾经手工维护 CASES 与规则的对应关系，两条新规则就漏了同步。
    covered = set(CASES.values())
    structural = [rule for rule in RULES if rule.startswith(("SUB-STR-", "SUB-LOD-"))]
    missing = [rule for rule in structural if rule not in covered]
    assert not missing, f"以下规则缺少非法样例：{sorted(missing)}"

    # 反向：清单里的目录必须都是规则表里存在的编号
    unknown = sorted(set(CASES.values()) - set(RULES))
    assert not unknown, f"以下样例对应了不存在的规则：{unknown}"


@pytest.mark.triton
@pytest.mark.parametrize("dirname,rule_id", sorted(CASES.items()))
def test_sample_hits_only_its_own_rule(dirname: str, rule_id: str) -> None:
    """样例必须**只**命中它对应的那条规则。

    实测教训：SUB-LOD-004~007 的样例里写了 torch.exp，
    于是在静态阶段就被「代算」拦下并短路，加载阶段的规则永远测不到，
    报出的编号还是 SUB-STR-001——提交方会照着错误的方向改。
    样例之间互相抢编号，等于这几条规则没有反例。
    """
    result = validate_submission(INVALID / dirname)
    unexpected = sorted({e.rule_id for e in result.errors} - {rule_id})
    assert not unexpected, (
        f"{dirname} 应只命中 {rule_id}，却额外报出 {unexpected}。"
        "样例自身带了别的违规会掩盖目标规则。"
    )


def test_sample_directory_exists() -> None:
    """目录必须真的在仓库里。

    SUB-LOD-001 演示「清单缺失」，本身是个空目录——
    git 不跟踪空目录，于是远端根本没有这个样例。
    用 .gitkeep 一类的占位文件解决，此测试守住这件事。
    """
    for dirname in CASES:
        assert (INVALID / dirname).is_dir(), f"缺少样例目录：{dirname}"


@pytest.mark.triton
def test_legal_demo_submission_still_passes() -> None:
    """阳性对照：合法提交不得被误杀。

    只有「非法被拒」与「合法放行」同时成立，才说明判定既不过松也不过严。
    """
    result = validate_submission(DEMO)
    assert result.ok, f"合法提交被误杀：{result.render()}"


@pytest.mark.triton
def test_validate_command_reports_and_exits_nonzero(capsys) -> None:
    """提交方要能在不跑评测的前提下自查——validate 子命令就是为此存在。"""
    from triton_eval import cli

    rc = cli.main(["validate", "--submission", str(INVALID / "SUB-STR-001")])
    assert rc == 1, "非法提交必须返回非 0，便于脚本判断"
    out = capsys.readouterr().out
    assert "SUB-STR-001" in out

    rc = cli.main(["validate", "--submission", str(DEMO)])
    assert rc == 0, "合法提交应返回 0"
    assert "校验通过" in capsys.readouterr().out
