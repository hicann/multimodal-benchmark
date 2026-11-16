# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""生成能力点清单的「共同确认」交付物（docs/feature-atoms-checklist.md）。

任务书 2.8 的前提是「双方共同确认的 Triton 核心语言特性清单」。本脚本把
机器清单（spec/feature_atoms.yaml）连同题库覆盖证据（corpus manifest 的
用例标签）渲染成一份可直接拿去定稿的文档：每个能力点一行——白话描述、
哪些算子在盯它、多少道题在考它、确认栏。

文档与数据的一致性由 tests/core/test_docs.py 对账（重新渲染必须与提交的文档
逐字一致）：改了清单或题库而忘了重跑本脚本，测试会红。

用法：
    python scripts/build_capability_checklist.py             # 默认 v0.5 题库 → docs/
    python scripts/build_capability_checklist.py <corpus 目录> <输出路径>
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
import logging

import json


import yaml
from triton_eval.console import emit

ROOT = Path(__file__).resolve().parents[1]
ATOMS_PATH = ROOT / "spec" / "feature_atoms.yaml"
DEFAULT_CORPUS = ROOT / "corpus" / "v0.5-cov20-frozen"  # 证据源随现行语料升级
DEFAULT_OUT = ROOT / "docs" / "feature-atoms-checklist.md"

# 分类名只存在于 yaml 注释里（数据字段只有编号），在此固化一份供文档展示；
# 成员的归属以 yaml 的 category 字段为准。
CATEGORY_NAMES = {
    "A01": "编程模型（grid 调度）",
    "A04": "线性代数",
    "A05": "内存 / 指针",
    "A06": "索引与算子属性",
    "A07": "数学函数",
    "A08": "归约",
    "A09": "归一化",
    "A10": "原子操作",
    "A12": "迭代器",
    "A13": "数值稳定性",
    "A14": "精度",
    "A15": "形态与规模",
    "A16": "算子组合（融合）",
    "A17": "优化手法（性能负载型）",
}


def _cell(text: object) -> str:
    """表格单元格：竖线转义，避免描述文本撑破表格。"""
    return str(text).replace("|", "\\|")


def render(atoms_path: Path = ATOMS_PATH, corpus_dir: Path = DEFAULT_CORPUS) -> str:
    fa = yaml.safe_load(atoms_path.read_text(encoding="utf-8"))
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))

    if manifest.get("feature_atoms_version") != fa.get("version"):
        raise ValueError(
            f"题库标签清单版本（{manifest.get('feature_atoms_version')}）与"
            f"spec/feature_atoms.yaml（{fa.get('version')}）不一致——"
            "请先对齐再生成确认版文档。"
        )

    # 覆盖证据：每个能力点被哪些算子、多少道题盯着
    ops_by_atom: dict[str, set[str]] = defaultdict(set)
    count_by_atom: Counter[str] = Counter()
    for case in manifest["cases"]:
        for atom in case.get("feature_atoms") or []:
            ops_by_atom[atom].add(case["operator"])
            count_by_atom[atom] += 1

    active = fa.get("atoms", [])
    reserved = fa.get("reserved", [])
    covered = sum(1 for a in active if count_by_atom[a["id"]] > 0)

    return _render_checklist(fa, manifest, corpus_dir, active, reserved, covered, ops_by_atom, count_by_atom)


def _render_checklist(fa, manifest, corpus_dir, active, reserved, covered, ops_by_atom, count_by_atom):
    lines: list[str] = []
    lines.append(f"# 语言能力点清单 —— 双方共同确认版（依据 {fa.get('version')}，待定稿）")
    lines.append("")
    lines.append(f"> 本文档由 `scripts/build_capability_checklist.py` 从机器清单"
                 f"（`spec/feature_atoms.yaml`）与题库覆盖证据"
                 f"（`corpus/{corpus_dir.name}`，冻结于 {manifest.get('frozen_at')}）"
                 "自动生成；改清单或题库后重跑脚本即可，对账测试保证文档不脱节。")
    lines.append("")
    lines.append("**这份文档怎么用**：任务书要求「覆盖**双方共同确认**的 Triton 核心语言"
                 "特性清单比例不低于 90%」。请逐条审阅：认可的条目在「确认」栏打勾；"
                 "有异议的条目在「意见」栏写明修改 / 删除 / 合并建议。全部过完后，"
                 "以确认版清单为分母重算覆盖率即可——标签是「算子规格 + 用例六维」的"
                 "纯函数，**无需重新生成任何用例**。")
    lines.append("")
    lines.append("**分母原则（诚实性原则）**：清单分「生效」（当前算子够得着，计入"
                 "覆盖率分母）与「预留」（当前够不着，不计入分母——把够不着的条目放进"
                 "分母，覆盖率就是自欺）两部分。算子扩容后，够得着的预留条目逐条迁入"
                 "生效。")
    lines.append("")
    lines.append(f"**当前状态**：生效 **{len(active)}** 条，题库全部有题盯"
                 f"（覆盖 {covered}/{len(active)}，我方口径）；预留 **{len(reserved)}** 条。")
    lines.append("")

    lines.append(f"## 一、生效条目（计入分母，{len(active)} 条）")
    lines.append("")
    _append_active_atoms(lines, active, ops_by_atom, count_by_atom)

    lines.append(f"## 二、预留条目（不计入分母，{len(reserved)} 条）")
    lines.append("")
    lines.append("| 能力点 | 为什么暂不进分母 | 确认 | 意见 |")
    lines.append("|---|---|---|---|")
    for atom in reserved:
        lines.append(f"| `{atom['id']}` | {_cell(atom.get('desc', ''))} | ☐ | |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*重新生成：`python scripts/build_capability_checklist.py`"
                 "（输出与数据逐字对账，见 tests/core/test_docs.py）*")
    lines.append("")
    return "\n".join(lines)


def _append_active_atoms(lines, active, ops_by_atom, count_by_atom):
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for atom in active:
        by_cat[atom["category"]].append(atom)
    for cat in sorted(by_cat):
        name = CATEGORY_NAMES.get(cat, "")
        title = f"{cat} {name}" if name else cat
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |")
        lines.append("|---|---|---|---|---|---|---|")
        for atom in by_cat[cat]:
            ops = "、".join(sorted(ops_by_atom.get(atom["id"], set()))) or "—"
            n = count_by_atom.get(atom["id"], 0)
            warn = "" if n else "（⚠ 题库暂无用例盯它）"
            lines.append(
                f"| `{atom['id']}` | {atom.get('layer', '—')} "
                f"| {_cell(atom.get('desc', ''))} | {ops} | {n}{warn} | ☐ | |"
            )
        lines.append("")


def main(argv: list[str]) -> int:
    corpus_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_CORPUS
    out_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUT
    try:
        text = render(corpus_dir=corpus_dir)
    except ValueError as exc:
        emit(str(exc), file=sys.stderr)
        return 1
    out_path.write_text(text, encoding="utf-8")
    emit(f"已生成 {out_path}（依据 {corpus_dir.name}）")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main(sys.argv))
