#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""测试集发布命令（3.4 版本发布机制）——打标即发布。

规则全文见 docs/versioning.md。两种用法：

    python scripts/release.py --corpus corpus/v0.5-cov20-frozen   # 正式发布（标签打在 HEAD）
    python scripts/release.py --backfill                         # 回溯补录历史版本（标签打在冻结提交）

两段式执行（打标前发布说明必须已提交，标签才指向包含自身说明的提交）：
    第一次运行：前置校验 → 生成/更新 docs/releases/<tag>.md → 提示提交；
    提交后重跑：校验 → 打 annotated 标签（正式=HEAD，补录=当前清单内容
    落地的提交）→ --push 时推送标签。

回溯补录只会处理**尚无标签**的语料——先正式发布当前版本，再跑
--backfill，其余历史版本自动补录。--push 之前所有标签先人工过目。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
import logging

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from triton_eval.buildlog import LOG_FILENAME  # noqa: E402
from triton_eval.corpusmod import load_manifest, verify_corpus_files  # noqa: E402
from triton_eval.release import (  # noqa: E402
    AXIS_ABSENT,
    errata_for,
    existing_tags,
    locate_freeze_commit,
    parse_errata,
    read_axes_at_commit,
    read_generation_env,
    read_toolchain_axes,
    render_release_notes,
    tag_name_for,
)
from triton_eval.console import emit

RELEASES_DIR = PROJECT_ROOT / "docs" / "releases"


def _git(*args: str) -> subprocess.CompletedProcess:
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("Git executable is unavailable")
    return subprocess.run(  # noqa: S603
        [str(Path(executable).resolve()), "-C", str(PROJECT_ROOT), *args],
        capture_output=True, text=True, timeout=120,
    )


def _tree_dirty() -> bool:
    return bool(_git("status", "--porcelain").stdout.strip())


def _verify_records_count() -> int:
    return sum(1 for p in (PROJECT_ROOT / "verify-records").glob("*/") if (p / "SUMMARY.md").is_file())


def _ensure_evolution_doc_fresh() -> None:
    """发布说明引用演化文档——必须与清单再生成一致，否则拒绝发布。"""
    script = PROJECT_ROOT / "scripts" / "corpus_evolution.py"
    spec = importlib.util.spec_from_file_location("corpus_evolution", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    committed = (PROJECT_ROOT / "docs" / "corpus-evolution.md").read_text(
        encoding="utf-8")
    if mod.render_all() != committed:
        raise ValueError(
            "✗ docs/corpus-evolution.md 与清单再生成结果不一致（发布说明会引用它）。\n"
            "  先执行：python scripts/corpus_evolution.py --all "
            "--out docs/corpus-evolution.md 并提交，再发布。")


def _write_notes(tag: str, notes: str) -> bool:
    """发布说明幂等写出：内容未变返回 False（已提交过的重跑场景）。"""
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    out = RELEASES_DIR / f"{tag}.md"
    if out.is_file() and out.read_text(encoding="utf-8") == notes:
        emit(f"  ✓ 发布说明已是最新：{out}")
        return False
    out.write_text(notes, encoding="utf-8")
    emit(f"  ✓ 已写入发布说明：{out}")
    return True


def _tag_message(notes_path: str, tag: str, manifest, axes: dict) -> str:
    return (
        f"{tag}: {manifest.version}\n"
        f"manifest {manifest.manifest_hash}\n"
        f"spec {axes['spec_version']} / scorer {axes['scorer_version']} / "
        f"tolerance {axes['tolerance_version']} / FA {axes['fa_version']}\n"
        f"用例 {len(manifest.cases)} / 算子 {len(manifest.operators)}\n"
        f"发布说明：{notes_path}（勘误见 docs/errata.md，规则见 docs/versioning.md）"
    )


def _create_tag(tag: str, commit: str, message: str) -> bool:
    r = _git("tag", "-a", tag, "-m", message, commit)
    if r.returncode != 0:
        emit(f"  ✗ 打标失败：{r.stderr.strip()}")
        return False
    emit(f"  ✓ 已打标 {tag} → {commit[:12]}")
    return True


def _push_tags(tags: list[str]) -> None:
    for tag in tags:
        r = _git("push", "origin", tag)
        emit(f"  {'✓' if r.returncode == 0 else '✗'} 推送 {tag}"
                     + ("" if r.returncode == 0 else f"：{r.stderr.strip()}"))


def _run_pytest() -> None:
    emit("  ── pytest 全量（发布前置校验）……")
    r = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT, timeout=1800,
    )
    if r.returncode != 0:
        raise ValueError("✗ pytest 未全绿，拒绝发布。")


def release_official(corpus_dir: Path, push: bool, skip_tests: bool) -> int:
    manifest = load_manifest(corpus_dir / "manifest.json")
    tag = tag_name_for(manifest.version)
    emit(f"═══ 正式发布 {manifest.version} → 标签 {tag}（打在 HEAD）═══")

    if tag in existing_tags(PROJECT_ROOT):
        emit(f"  ✗ 标签 {tag} 已存在")
        return 1

    emit("── 前置校验")
    verify_corpus_files(corpus_dir, manifest)
    emit("  ✓ 清单指纹可重算（manifest 哈希 + 逐用例内容哈希）")
    _ensure_evolution_doc_fresh()
    emit("  ✓ 演化对照文档与清单再生成一致")
    if not (corpus_dir / LOG_FILENAME).is_file():
        raise ValueError(f"✗ 缺少生成日志 {corpus_dir / LOG_FILENAME}")
    emit("  ✓ 生成日志存在")
    axes = read_toolchain_axes(PROJECT_ROOT)
    if AXIS_ABSENT in axes.values():
        raise ValueError(f"✗ 工具轴读取不完整：{axes}")
    if not skip_tests:
        _run_pytest()

    errata = errata_for(parse_errata(PROJECT_ROOT / "docs" / "errata.md"),
                        manifest.version)
    prev_summary = None
    siblings = sorted((PROJECT_ROOT / "corpus").glob("v*/manifest.json"))
    idx = [p.parent for p in siblings].index(corpus_dir)
    if idx > 0:
        from triton_eval.release import _diff_summary
        prev = load_manifest(siblings[idx - 1])
        prev_summary = _diff_summary(prev, manifest)
    freeze = locate_freeze_commit(PROJECT_ROOT, corpus_dir.name)

    notes = render_release_notes(
        manifest=manifest, axes=axes,
        gen_env=read_generation_env(corpus_dir),
        errata=errata, backfill=False,
        freeze_commit=freeze, prev_summary=prev_summary,
        verify_records=_verify_records_count(),
        corpus_dir_name=corpus_dir.name,
    )
    if _write_notes(tag, notes):
        emit("  → 发布说明有更新：请提交后重跑本命令完成打标")
        return 0

    if _tree_dirty():
        raise ValueError("✗ 工作区不干净（发布必须基于已提交状态）——先 commit 再重跑。")
    ok = _create_tag(tag, "HEAD",
                     _tag_message(f"docs/releases/{tag}.md", tag, manifest, axes))
    if ok and push:
        _push_tags([tag])
    return 0 if ok else 1


def release_backfill(push: bool) -> int:
    corpora = sorted(p.parent for p in (PROJECT_ROOT / "corpus").glob("v*/manifest.json"))
    tags = existing_tags(PROJECT_ROOT)
    errata_all = parse_errata(PROJECT_ROOT / "docs" / "errata.md")
    targets = []
    for d in corpora:
        m = load_manifest(d / "manifest.json")
        t = tag_name_for(m.version)
        if t in tags:
            emit(f"  ── 跳过 {d.name}（已有标签 {t}）")
            continue
        targets.append((d, m, t))
    if not targets:
        emit("所有语料版本均已打标，无补录对象。")
        return 0

    wrote = False
    for i, (d, m, t) in enumerate(targets):
        emit(f"═══ 补录 {m.version} → 标签 {t}（打在冻结提交）═══")
        freeze = locate_freeze_commit(PROJECT_ROOT, d.name)
        axes = read_axes_at_commit(PROJECT_ROOT, freeze)
        prev_summary = None
        if i > 0:
            prev_m = load_manifest(corpora[corpora.index(d) - 1] / "manifest.json")
            if prev_m.version != m.version:
                from triton_eval.release import _diff_summary
                prev_summary = _diff_summary(prev_m, m)
        notes = render_release_notes(
            manifest=m, axes=axes, gen_env=None,
            errata=errata_for(errata_all, m.version), backfill=True,
            freeze_commit=freeze, prev_summary=prev_summary,
            verify_records=_verify_records_count(),
            corpus_dir_name=d.name,
        )
        wrote |= _write_notes(t, notes)

    if wrote:
        emit("  → 补录说明有更新：请提交后重跑本命令完成打标")
        return 0
    if _tree_dirty():
        raise ValueError("✗ 工作区不干净——先 commit 再重跑。")

    created = []
    for d, m, t in targets:
        freeze = locate_freeze_commit(PROJECT_ROOT, d.name)
        axes = read_axes_at_commit(PROJECT_ROOT, freeze)
        if _create_tag(t, freeze,
                       "【补录】\n" + _tag_message(f"docs/releases/{t}.md",
                                               t, m, axes)):
            created.append(t)
    if push and created:
        _push_tags(created)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--corpus", metavar="DIR", help="正式发布该语料（标签打在 HEAD）")
    g.add_argument("--backfill", action="store_true",
                   help="回溯补录所有尚无标签的历史语料")
    ap.add_argument("--push", action="store_true", help="打标后推送标签")
    ap.add_argument("--skip-tests", action="store_true",
                    help="跳过 pytest 前置（不建议；发布应基于全绿状态）")
    args = ap.parse_args()

    try:
        if args.corpus:
            return release_official(PROJECT_ROOT / args.corpus, args.push, args.skip_tests)
        return release_backfill(args.push)
    except ValueError as exc:
        emit(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
