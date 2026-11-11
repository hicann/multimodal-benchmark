#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""测试集演化对照表生成器——两版清单之间自动 diff，供审计。

此前审计只能人工比对 manifest（v0.4→v0.5 的按算子再分配是手工数出来
的）。本脚本把它自动化：算子增删、逐算子用例数、用例级增/删/改
（同编号字段级差异）、规模与 dtype 再分配、覆盖原子得失、门禁剔除
记录、生成日志存续情况。

只输出**可从清单推导的事实**；版本内变更的背景与理由属人工叙述，
在方案文档的落地实况章节（如 docs/composition-and-pattern-dimensions.md
§6），本文档与其互相指引、互不冒充。

用法：
    python scripts/corpus_evolution.py                          # 全部相邻版本对，打印
    python scripts/corpus_evolution.py --all                    # 同上
    python scripts/corpus_evolution.py <旧目录> <新目录>         # 指定一对
    python scripts/corpus_evolution.py --all --out docs/corpus-evolution.md
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging

import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from triton_eval.buildlog import LOG_FILENAME, read_log  # noqa: E402
from triton_eval.corpusmod import load_manifest  # noqa: E402
from triton_eval.evolution import diff_manifests, render_markdown  # noqa: E402
from triton_eval.console import emit

DOC_HEADER = """# 测试集演化记录（自动生成，审计用）

> 由 `scripts/corpus_evolution.py` 从 `corpus/<版本>/manifest.json` 推导生成，
> 重新生成：`python scripts/corpus_evolution.py --all --out docs/corpus-evolution.md`。
> 对账测试钉住本文档与再生成结果一致，清单变了不重出文档即测试红。

本表只含清单可推导的事实。与它互补的三类留痕：

1. **门禁剔除记录**：`manifest.json` 的 `excluded` 字段，逐条含原因，随清单
   哈希冻结——事后抹掉或改写会破坏清单指纹，评测启动时即被检出；
2. **生成日志**：`corpus/<版本>/generation-log.yaml`，构建过程每步一行
   （2026-10-14 起启用；此前构建的终端输出未留存）；
3. **人工叙述**：版本内变更的背景与理由（为什么重冻结、为什么再分配）
   在方案文档的落地实况章节，如
   `docs/composition-and-pattern-dimensions.md` §6。
"""


def _version_key(name: str):
    """自然排序键：v0.10 排在 v0.9 之后。"""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _discover() -> list[Path]:
    dirs = sorted(
        (p.parent for p in (PROJECT_ROOT / "corpus").glob("v*/manifest.json")),
        key=lambda d: _version_key(d.name),
    )
    return dirs


def _log_summary(corpus_dir: Path) -> dict | None:
    path = corpus_dir / LOG_FILENAME
    if not path.is_file():
        return None
    events = read_log(path)
    return {"events": len(events), "first_ts": events[0]["ts"] if events else ""}


def render_all(pairs: list[tuple[Path, Path]] | None = None) -> str:
    """生成完整演化对照文档（对账测试与 main 共用同一出口）。"""
    if pairs is None:
        found = _discover()
        if len(found) < 2:
            raise ValueError("只找到一个冻结版本，没有可对照的迁移")
        pairs = list(zip(found[:-1], found[1:]))

    sections = []
    overview_rows = []
    for old_dir, new_dir in pairs:
        old, new = load_manifest(old_dir / "manifest.json"), \
            load_manifest(new_dir / "manifest.json")
        diff = diff_manifests(old, new)
        logs = {"old": _log_summary(old_dir), "new": _log_summary(new_dir)}
        sections.append(render_markdown(diff, gen_logs=logs))
        overview_rows.append(
            f"| {old.version} → {new.version} "
            f"| {diff['old']['cases']} → {diff['new']['cases']} "
            f"| +{len(diff['operators_added'])} / −{len(diff['operators_removed'])} "
            f"| +{len(diff['cases_added'])} / −{len(diff['cases_removed'])} / "
            f"改 {len(diff['cases_changed'])} |"
        )

    doc_parts = [DOC_HEADER, "## 迁移总览", "",
                 "| 迁移 | 用例数 | 算子 | 用例（增/删/改） |", "|---|---|---|---|"]
    doc_parts += overview_rows
    doc_parts += ["", "---", ""]
    return "\n".join(doc_parts) + "\n\n".join(sections)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="旧、新两个语料目录（缺省=全部相邻对）")
    ap.add_argument("--all", action="store_true", help="全部相邻版本对（与缺省同）")
    ap.add_argument("--out", default=None,
                    help="写入 markdown 文件（缺省只打印）")
    args = ap.parse_args()

    if args.all or not args.dirs:
        pairs = None
    else:
        if len(args.dirs) != 2:
            ap.error("给出目录时须恰好两个（旧、新）")
        pairs = [(Path(args.dirs[0]), Path(args.dirs[1]))]

    try:
        doc = render_all(pairs)
    except ValueError as exc:
        emit(str(exc), file=sys.stderr)
        return 1

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc, encoding="utf-8")
        emit(f"已写入 {out_path}")
    else:
        emit(doc)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
