#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""构建并冻结 v0.5 测试集（组合与手法维度扩展版）。

与 build_corpus_v2.py 的差异仅在目标版本与资产来源：
    - 语料：corpus/v0.4-cov20-frozen → corpus/v0.5-cov20-frozen
    - 算子：14 原语 → 25（+11 融合算子，注册表第 3 批）
    - 能力点清单：v1.0-corpus（55）→ v1.1-fusion（71，+融合/手法/并发原子）
    - 参考：原语 ref.py 已从 v0.4 迁移至 v0.5（构建时若缺会自动补迁）；
      11 个融合参考与 Scatter 的并发累加分支是 v0.5 的**版本内变更**，
      已就位于 v0.5 目录，不覆盖。

流水线与 v2 完全一致（参数化候选池 → 打标 → 覆盖引导选择 → 参考自检
门禁 → 冻结指纹），机制零改动。

生成日志（2026-10-14 起，任务书审计要求）：构建每一步追加进
corpus/<版本>/generation-log.yaml——候选池统计、近重复去重数量、
选择报告、门禁剔除逐条原因、终态指纹、清单是否重写。历史版本
（v0.3/v0.4/v0.5 首冻）的终端输出未留存，日志从本机制上线后的首次
构建（含重跑复验）开始积累。

用法：
    python scripts/build_corpus_v3.py                    # 每类 20 个
    python scripts/build_corpus_v3.py --per-category 40
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_eval.buildlog import builder_arguments, run_reference_gate, select_manifest_candidates
from triton_eval.buildlog import LOG_FILENAME, GenerationLog, report_frozen_corpus  # noqa: E402
from triton_eval.casegen import manifest_candidates, manifest_operators  # noqa: E402
from triton_eval.coverage import coverage_stats  # noqa: E402
from triton_eval.corpusmod import build_manifest, write_manifest  # noqa: E402
from triton_eval.models import CaseSpec, OperatorSpec  # noqa: E402
from triton_eval.operator_registry import ALL_OPERATORS  # noqa: E402
from triton_eval.specload import load_feature_atoms, load_spec  # noqa: E402
from triton_eval.console import emit

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# v0.5 的资产来源：原语参考从 v0.4 迁移；v0.5 自有的融合参考不覆盖
SRC_CORPUS = PROJECT_ROOT / "corpus" / "v0.4-cov20-frozen"
CORPUS_DIR = PROJECT_ROOT / "corpus" / "v0.5-cov20-frozen"
VERSION = "v0.5-cov20-frozen"


def _operators() -> list[OperatorSpec]:
    return manifest_operators(ALL_OPERATORS)


def _candidates() -> list[CaseSpec]:
    return manifest_candidates(ALL_OPERATORS)


def _git_commit() -> str | None:
    """构建时代码版本——语料由哪一次提交生成，审计可回溯。"""
    try:
        return subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - 无 git 环境则留空
        return None


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    args = _build_arguments()

    log = GenerationLog(CORPUS_DIR / LOG_FILENAME)
    log.step("build_start", target_version=VERSION,
             per_category=args.per_category, device=args.device,
             verify_refs=args.verify_refs,
             python=sys.version.split()[0], platform=platform.platform(),
             torch=_torch_version(), git_commit=_git_commit())

    fa = load_feature_atoms()
    atom_universe = [a["id"] for a in fa["atoms"]]
    emit(f"FA 清单 {fa['version']}：active {len(atom_universe)} 个原子，"
                 f"reserved {len(fa.get('reserved', []))} 个（不计分母）")
    log.step("feature_atoms", version=str(fa["version"]),
             active=len(atom_universe), reserved=len(fa.get("reserved", [])))

    candidates = _candidate_pool(atom_universe, log)

    selected = _select_candidate_cases(candidates, args, atom_universe, log)

    # ---- 质量门禁：参考实现自检（冻结门禁，不是善后）----
    excluded: list[dict[str, str]] = []
    cases = selected
    cases = _reference_gate(args, cases, excluded, log)

    # ---- 门禁后覆盖必须仍达标，否则说明门禁挖走了唯一的覆盖来源 ----
    final_stats = coverage_stats(cases, atom_universe)
    emit(f"门禁后：{len(cases)} 个用例，覆盖 "
                 f"{len(final_stats['covered'])}/{len(atom_universe)}"
                 f"（{final_stats['rate']:.1%}）")
    if final_stats["missing"]:
        emit(f"  门禁后仍未覆盖：{', '.join(final_stats['missing'])}",
                     file=sys.stderr)
    log.step("coverage_after_gate", cases=len(cases),
             covered=len(final_stats["covered"]), rate=final_stats["rate"],
             missing=final_stats["missing"])

    # ---- 参考资产迁移：原语 ref.py 从 v0.4 补迁（缺才拷，绝不覆盖 v0.5 自有参考）----
    _migrate_references(log)

    manifest = build_manifest(
        CORPUS_DIR, VERSION, load_spec().spec_version, _operators(), cases,
        feature_atoms_version=str(fa["version"]),
        excluded=excluded,
    )
    out = CORPUS_DIR / "manifest.json"
    written = write_manifest(manifest, out)

    by_op_size = _report_frozen_corpus(cases, manifest, final_stats, fa, out, written)

    log.step("build_end", version=VERSION, frozen_at=manifest.frozen_at,
             operators=len(manifest.operators), cases=len(manifest.cases),
             excluded_count=len(manifest.excluded),
             coverage_rate=final_stats["rate"],
             by_op_size=dict(sorted(by_op_size.items())),
             manifest_hash=manifest.manifest_hash,
             manifest_written=written)
    return 0




def _build_arguments():
    return builder_arguments()


def _candidate_pool(atom_universe, log):
    candidates = _candidates()
    emit(f"候选池：{len(candidates)} 个（{len(ALL_OPERATORS)} 算子 × "
                 f"形态×规模×dtype×属性×数值×布局）")
    pool_stats = coverage_stats(candidates, atom_universe)
    emit(f"池覆盖 {len(pool_stats['covered'])}/{len(atom_universe)}"
                 f"（{pool_stats['rate']:.0%}）——池都够不着的原子，选择阶段也无解")
    if pool_stats["unknown_atoms"]:
        emit(f"  !! 池中出现清单外的原子（标签器与 yaml 漂移）："
                     f"{', '.join(pool_stats['unknown_atoms'])}", file=sys.stderr)
    log.step("candidate_pool", candidates=len(candidates),
             operators=len(ALL_OPERATORS),
             pool_covered=len(pool_stats["covered"]),
             atom_total=len(atom_universe), pool_rate=pool_stats["rate"],
             pool_unknown_atoms=pool_stats["unknown_atoms"])
    return candidates


def _reference_gate(args, cases, excluded, log):
    gate_skipped = False
    if args.verify_refs:
        cases, gate_skipped = run_reference_gate(cases, args.device, CORPUS_DIR, excluded)
    log.step("gate_refcheck", device=args.device, skipped=gate_skipped,
             excluded_count=len(excluded), excluded=excluded)
    return cases


def _migrate_references(log):
    from triton_eval.corpus_assets import reference_sources, resolve_reference_path

    migrated = 0
    for relative, source in reference_sources(SRC_CORPUS).items():
        if Path(relative).name != "ref.py":
            continue
        target = CORPUS_DIR / relative
        if resolve_reference_path(CORPUS_DIR, relative).is_file():
            continue  # v0.5 已有（融合参考或版本内变更，如 scatter 并发分支）
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        migrated += 1
    if migrated:
        emit(f"从 {SRC_CORPUS.name} 补迁 {migrated} 份原语 ref.py")
    log.step("ref_migration", source=SRC_CORPUS.name, migrated=migrated)


def _report_frozen_corpus(cases, manifest, final_stats, fa, out, written):
    return report_frozen_corpus(
        cases, manifest, final_stats["rate"], fa["version"], out, written,
        version=VERSION, column_width=38)




def _select_candidate_cases(candidates, args, atom_universe, log):
    selected, report = select_manifest_candidates(
        candidates, args.per_category, atom_universe)
    stats = report["coverage"]
    if stats["unknown_atoms"]:
        emit(f"  !! 清单外的原子：{', '.join(stats['unknown_atoms'])}",
                     file=sys.stderr)
    # 近重复去重数：进选择器的候选 − 去重后的池（seed 不同不算差异，
    # 同签名只留最廉价的一个）。此前只发生在内存里，无任何留痕。
    log.step("selection", selected=report["selected"],
             pool_size=report["pool_size"],
             near_dup_dropped=len(candidates) - report["pool_size"],
             per_category_quota=args.per_category,
             covered=len(stats["covered"]), rate=stats["rate"],
             missing=stats["missing"], unknown_atoms=stats["unknown_atoms"])
    return selected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
