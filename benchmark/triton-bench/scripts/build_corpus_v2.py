#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""覆盖引导生成 v2 —— 构建并冻结新测试集。

范式（参数化生成 + 覆盖引导选择的流水线）：

    候选池（casegen.derive_pool：形态×规模×dtype×属性×数值×布局，
            派生即校验合法性）
      → 打标（coverage.label_case：标签 = 算子规格 + 用例五维的纯函数）
      → 覆盖引导选择（coverage.guided_selection：
            原子播种 → 每算子每档规模/每 dtype 配额保底 → 多样性补齐）
      → 质量门禁（refcheck：参考实现能跑、可复现、输出有限——CPU 即可）
      → 冻结（指纹 + 标签随行，excluded 留痕）

用法：
    python scripts/build_corpus_v2.py                    # 每类 20 个
    python scripts/build_corpus_v2.py --per-category 40  # 规模旋钮
"""

from __future__ import annotations

import logging
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_eval.buildlog import builder_arguments, run_reference_gate, select_manifest_candidates
from triton_eval.buildlog import report_frozen_corpus
from triton_eval.casegen import manifest_candidates, manifest_operators  # noqa: E402
from triton_eval.coverage import coverage_stats  # noqa: E402
from triton_eval.corpusmod import build_manifest, write_manifest  # noqa: E402
from triton_eval.models import CaseSpec, OperatorSpec  # noqa: E402
from triton_eval.operator_registry import ALL_OPERATORS  # noqa: E402
from triton_eval.specload import load_feature_atoms, load_spec  # noqa: E402
from triton_eval.console import emit

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 从 v0.3 继承参考实现资产（ref.py 是评测方资产，随新版本整体迁移）
SRC_CORPUS = PROJECT_ROOT / "corpus" / "v0.3-multi-input-frozen"
CORPUS_DIR = PROJECT_ROOT / "corpus" / "v0.4-cov20-frozen"
VERSION = "v0.4-cov20-frozen"


def _operators() -> list[OperatorSpec]:
    return manifest_operators(ALL_OPERATORS)


def _candidates() -> list[CaseSpec]:
    return manifest_candidates(ALL_OPERATORS)


def main() -> int:
    args = _build_arguments()

    fa = load_feature_atoms()
    atom_universe = [a["id"] for a in fa["atoms"]]
    emit(f"FA 清单 {fa['version']}：active {len(atom_universe)} 个原子，"
                 f"reserved {len(fa.get('reserved', []))} 个（不计分母）")

    pool = _candidates()
    emit(f"候选池：{len(pool)} 个（14 算子 × 形态×规模×dtype×属性×数值×布局）")
    pool_stats = coverage_stats(pool, atom_universe)
    emit(f"池覆盖 {len(pool_stats['covered'])}/{len(atom_universe)}"
                 f"（{pool_stats['rate']:.0%}）——池都够不着的原子，选择阶段也无解")

    selected, report = select_manifest_candidates(
        pool, args.per_category, atom_universe)
    stats = report["coverage"]
    if stats["unknown_atoms"]:
        emit(f"  !! 池中出现清单外的原子（标签器与 yaml 漂移）："
                     f"{', '.join(stats['unknown_atoms'])}", file=sys.stderr)

    # ---- 质量门禁：参考实现自检（冻结门禁，不是善后）----
    excluded: list[dict[str, str]] = []
    cases = selected
    cases = _reference_gate(args, cases, excluded)

    # ---- 门禁后覆盖必须仍达标，否则说明门禁挖走了唯一的覆盖来源 ----
    final_stats = coverage_stats(cases, atom_universe)
    emit(f"门禁后：{len(cases)} 个用例，覆盖 "
                 f"{len(final_stats['covered'])}/{len(atom_universe)}"
                 f"（{final_stats['rate']:.1%}）")
    if final_stats["missing"]:
        emit(f"  门禁后仍未覆盖：{', '.join(final_stats['missing'])}", file=sys.stderr)

    # ---- 迁移评测方资产（ref.py），冻结清单 ----
    _migrate_references()

    manifest = build_manifest(
        CORPUS_DIR, VERSION, load_spec().spec_version, _operators(), cases,
        feature_atoms_version=str(fa["version"]),
        excluded=excluded,
    )
    out = CORPUS_DIR / "manifest.json"
    written = write_manifest(manifest, out)

    _report_frozen_corpus(cases, manifest, final_stats, fa, out, written)

    return 0




def _build_arguments():
    return builder_arguments()


def _reference_gate(args, cases, excluded):
    if not args.verify_refs:
        return cases
    return run_reference_gate(cases, args.device, CORPUS_DIR, excluded)[0]


def _migrate_references():
    from triton_eval.corpus_assets import copy_reference_tree, resolve_reference_path

    if not resolve_reference_path(CORPUS_DIR, "operators/exp/ref.py").is_file():
        copy_reference_tree(SRC_CORPUS, CORPUS_DIR)
        emit(f"已迁移 {len(list((CORPUS_DIR / 'operators').iterdir()))} 份 ref.py"
                     f"（来自 {SRC_CORPUS.name}）")


def _report_frozen_corpus(cases, manifest, final_stats, fa, out, written):
    report_frozen_corpus(
        cases, manifest, final_stats["rate"], fa["version"], out, written,
        version=VERSION, column_width=28)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
