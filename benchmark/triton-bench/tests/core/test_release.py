# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""发布机制（release 模块）的单元测试——git 历史、勘误解析、发布说明渲染。"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import yaml

import pytest

from triton_eval.buildlog import LOG_FILENAME
from triton_eval.corpusmod import load_manifest
from triton_eval.release import (
    AXIS_ABSENT,
    _diff_summary,
    errata_for,
    locate_freeze_commit,
    parse_errata,
    read_axes_at_commit,
    read_generation_env,
    read_toolchain_axes,
    render_release_notes,
    tag_name_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ERRATA = PROJECT_ROOT / "docs" / "errata.md"




@pytest.fixture()
def release_history(tmp_path):
    """A real, isolated Git history; never reported as product/NPU evidence."""
    project = tmp_path / "history"
    project.mkdir()
    for name in ("v0.3-multi-input-frozen", "v0.4-cov20-frozen", "v0.5-cov20-frozen"):
        source = PROJECT_ROOT / "corpus" / name
        destination = project / "corpus" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copytree(source, destination)
    shutil.copytree(PROJECT_ROOT / "spec", project / "spec")
    executable = shutil.which('git')
    if executable is None:
        raise FileNotFoundError('Git executable is required for this regression')

    def git(*arguments):
        return subprocess.run([str(Path(executable).resolve()), "-c", "core.hooksPath=/dev/null",
                               "-c", "user.name=Regression Fixture",
                               "-c", "user.email=fixture@example.invalid", "-C", str(project), *arguments],
                              check=True, capture_output=True, text=True, timeout=60)

    git("init", "-q")
    for name, version in (("tolerance.yaml", "1.0"), ("feature_atoms.yaml", "1.0-corpus")):
        path = project / "spec" / name
        payload = yaml.safe_load(path.read_text())
        payload["version"] = version
        path.write_text(yaml.safe_dump(payload, allow_unicode=True))
    git("add", ".")
    git("commit", "-qm", "Fixture: freeze initial manifests and axes")
    shutil.copytree(PROJECT_ROOT / "spec", project / "spec", dirs_exist_ok=True)
    git("add", "spec")
    git("commit", "-qm", "Fixture: evolve current toolchain axes")
    return project


# ---------- 标签命名 ----------


def test_tag_name_strips_strategy_suffix() -> None:
    assert tag_name_for("v0.5-cov20-frozen") == "corpus-v0.5"
    assert tag_name_for("v0.4-cov20-frozen") == "corpus-v0.4"
    assert tag_name_for("v0.3-multi-input-frozen") == "corpus-v0.3"


def test_tag_name_rejects_unparseable_version() -> None:
    with pytest.raises(ValueError):
        tag_name_for("cov20-frozen")


# ---------- 五轴快照 ----------


def test_read_toolchain_axes_current_tree() -> None:
    """当前树四条工具轴必须齐全可读（发布快照的地基）。"""
    axes = read_toolchain_axes(PROJECT_ROOT)
    assert axes["spec_version"] == "0.1-mvp"
    assert axes["scorer_version"] == "0.2.4"
    assert axes["tolerance_version"] == "1.2"
    assert axes["fa_version"] == "1.1-fusion"
    assert AXIS_ABSENT not in axes.values()


def test_read_axes_at_v03_freeze_commit(release_history) -> None:
    """历史提交的五轴快照取自定位到的提交。

    v0.3 的当前清单内容落地在 v0.4 落地的同一提交（8df38f6：schema
    迁移重算指纹）——定位语义是"当前清单内容落地的提交"，不是首次
    冻结提交。该提交时点：tolerance 1.0 / FA 1.0-corpus。
    """
    commit = locate_freeze_commit(release_history, "v0.3-multi-input-frozen")
    axes = read_axes_at_commit(release_history, commit)
    assert axes["spec_version"] == "0.1-mvp"
    assert axes["tolerance_version"] == "1.0"
    assert axes["fa_version"] == "1.0-corpus"


# ---------- 回溯定位（对真实 git 历史跑） ----------


@pytest.mark.parametrize("name", [
    "v0.3-multi-input-frozen", "v0.4-cov20-frozen", "v0.5-cov20-frozen",
])
def test_locate_freeze_commit_content_matches(name: str, release_history) -> None:
    """定位到的提交上，manifest 与现存内容逐字一致（打标锚点的基本要求）。"""
    commit = locate_freeze_commit(release_history, name)
    executable = shutil.which('git')
    if executable is None:
        raise FileNotFoundError('Git executable is required for this regression')
    at = subprocess.run(  # noqa: S603
        [str(Path(executable).resolve()), "-C", str(release_history), "show",
         f"{commit}:corpus/{name}/manifest.json"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    assert at == (release_history / "corpus" / name / "manifest.json").read_text(
        encoding="utf-8")


# ---------- 勘误解析与过滤 ----------


def test_parse_errata_reads_all_entries() -> None:
    entries = parse_errata(ERRATA)
    assert len(entries) >= 8
    first = next(e for e in entries if e["id"] == "ERR-001")
    assert first["处置"] == "AMEND"
    assert first["versions"] == ["v0.3-multi-input-frozen", "v0.4-cov20-frozen"]
    ce = next(e for e in entries if e["id"] == "ERR-007")
    assert ce["处置"] == "ANNOTATE"
    assert ce["versions"] == ["v0.5-cov20-frozen"]


def test_errata_filters_by_corpus_version() -> None:
    entries = parse_errata(ERRATA)
    for_v05 = errata_for(entries, "v0.5-cov20-frozen")
    assert {e["id"] for e in for_v05} == {"ERR-005", "ERR-006", "ERR-007", "ERR-008"}
    for_v04 = errata_for(entries, "v0.4-cov20-frozen")
    assert {e["id"] for e in for_v04} == {"ERR-001", "ERR-002", "ERR-003", "ERR-004"}
    # 每条已登记勘误至少影响一个真实版本，且受影响版本全部来自冻结语料
    frozen = {"v0.3-multi-input-frozen", "v0.4-cov20-frozen", "v0.5-cov20-frozen"}
    for e in entries:
        assert e["versions"] and set(e["versions"]) <= frozen


def test_errata_dispositions_are_from_the_three_classes() -> None:
    for e in parse_errata(ERRATA):
        assert e["处置"] in {"AMEND", "RETRACT", "ANNOTATE"}, (
            f"{e['id']} 的处置分类不在三分类内")
    assert all(e["处置"] != "RETRACT" for e in parse_errata(ERRATA)), (
        "RETRACT 历史上未发生过——若出现需触发 MAJOR+1 并作废历史分数，"
        "登记时必须逐字说明")


# ---------- 生成环境与发布说明 ----------


def test_read_generation_env_present_for_v05() -> None:
    env = read_generation_env(
        PROJECT_ROOT / "corpus" / "v0.5-cov20-frozen")
    assert env is not None
    assert env["git_commit"] and env["python"] and env["torch"]
    assert env["events"] > 0


def test_render_official_notes_contains_key_sections(release_history) -> None:
    corpus_dir = release_history / "corpus" / "v0.5-cov20-frozen"
    manifest = load_manifest(corpus_dir / "manifest.json")
    axes = read_toolchain_axes(release_history)
    errata = errata_for(parse_errata(ERRATA), manifest.version)
    prev = load_manifest(release_history / "corpus" / "v0.4-cov20-frozen" / "manifest.json")
    notes = render_release_notes(
        manifest=manifest, axes=axes,
        gen_env=read_generation_env(corpus_dir),
        errata=errata, backfill=False,
        freeze_commit=locate_freeze_commit(release_history, corpus_dir.name),
        prev_summary=_diff_summary(prev, manifest),
        verify_records=15, corpus_dir_name=corpus_dir.name,
    )
    assert "# 测试集发布：corpus-v0.5" in notes
    for v in (manifest.version, *axes.values()):
        assert v in notes
    assert manifest.manifest_hash in notes
    assert "docs/corpus-evolution.md" in notes
    assert "ERR-006" in notes  # 受 v0.5 影响的勘误入选
    assert "ERR-001" not in notes  # v0.4 时代的勘误不混入
    assert "15 份" in notes
    # 快照文档不含当前时间——重渲染必须逐字一致（幂等基础）
    again = render_release_notes(
        manifest=manifest, axes=axes,
        gen_env=read_generation_env(corpus_dir),
        errata=errata, backfill=False,
        freeze_commit=locate_freeze_commit(release_history, corpus_dir.name),
        prev_summary=_diff_summary(prev, manifest),
        verify_records=15, corpus_dir_name=corpus_dir.name,
    )
    assert again == notes


def test_render_backfill_notes_marks_retroactive(release_history) -> None:
    corpus_dir = release_history / "corpus" / "v0.4-cov20-frozen"
    manifest = load_manifest(corpus_dir / "manifest.json")
    commit = locate_freeze_commit(release_history, corpus_dir.name)
    notes = render_release_notes(
        manifest=manifest,
        axes=read_axes_at_commit(release_history, commit),
        gen_env=None,
        errata=errata_for(parse_errata(ERRATA), manifest.version),
        backfill=True, freeze_commit=commit,
        prev_summary=None, verify_records=15,
        corpus_dir_name=corpus_dir.name,
    )
    assert "补录标签" in notes
    assert "ERR-001" in notes  # v0.4 时代的勘误入选
    assert "早于生成日志机制" in notes  # 该版无生成日志——如实标注
