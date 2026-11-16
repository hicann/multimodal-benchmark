# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""测试集发布（3.4 版本发布机制）——标签命名、五轴快照、勘误过滤、发布说明。

任务书要求"形成版本发布……的运行机制"。规则全文在 docs/versioning.md：
    发布 = 一个 annotated git 标签 + 一页发布说明（docs/releases/）。
    版本对象 = 测试集五轴组合（corpus / spec / scorer / tolerance / FA）。
    指纹回答"内容是否被改"，标签回答"这是哪一版"。

本模块是发布流程的可测内核；scripts/release.py 是命令行薄壳。
所有生成物**不含当前时间**——发布说明是快照文档，可由标签与文件状态
完全复现（frozen_at 取自清单，验证记录数取自发布时点的目录清点）。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import yaml

from .buildlog import LOG_FILENAME, read_log
from .corpusmod import CorpusManifest, load_manifest

AXIS_ABSENT = "（当时无此轴）"


# ---------- git 辅助（全部带超时：git 对断死的连接会无限挂起） ----------

def _git(repo: Path, *args: str) -> str | None:
    """返回 None 表示命令失败（如历史提交中文件不存在）。"""
    try:
        executable = shutil.which("git")
        if executable is None:
            raise FileNotFoundError("Git executable is unavailable")
        out = subprocess.run(  # noqa: S603
            [str(Path(executable).resolve()), "-C", str(repo), *args],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _repository_path(project: Path, relative: str) -> str:
    """Resolve a project-relative path for git-show in standalone/monorepo layouts."""
    prefix = _git(project, "rev-parse", "--show-prefix")
    return (prefix.strip() if prefix else "") + relative


def existing_tags(repo: Path) -> set[str]:
    out = _git(repo, "tag", "--list")
    return set(out.split()) if out else set()


def locate_freeze_commit(repo: Path, corpus_dir_name: str) -> str:
    """当前清单内容落地的提交（回溯打标的锚点）。

    从新到旧扫 touch 过 manifest.json 的提交，第一个内容与现存一致的
    即锚点——不一定是首次提交：v0.5 经历过 286→289 重冻结，锚点在
    重冻结那次。找不到（清单从未提交/内容漂移）则抛错。
    """
    rel = f"corpus/{corpus_dir_name}/manifest.json"
    current = (repo / rel).read_text(encoding="utf-8")
    log = _git(repo, "log", "--format=%H", "--", rel)
    if not log:
        raise RuntimeError(f"{rel} 没有任何提交记录，无法回溯打标")
    for commit in log.split():
        at = _git(repo, "show", f"{commit}:{_repository_path(repo, rel)}")
        if at is not None and at == current:
            return commit
    raise RuntimeError(
        f"{rel} 在全部历史提交中找不到与现存内容一致的版本——"
        "工作区清单可能未提交，先 commit 再回溯打标"
    )


# ---------- 五轴快照 ----------

def _axes_from_yaml_texts(
    spec_text: str | None, tolerance_text: str | None, fa_text: str | None,
) -> dict[str, str]:
    def _ver(text: str | None, key: str) -> str:
        if not text:
            return AXIS_ABSENT
        data = yaml.safe_load(text) or {}
        return str(data.get(key) or AXIS_ABSENT)

    return {
        "spec_version": _ver(spec_text, "spec_version"),
        "scorer_version": _ver(spec_text, "scorer_version"),
        "tolerance_version": _ver(tolerance_text, "version"),
        "fa_version": _ver(fa_text, "version"),
    }


def read_toolchain_axes(repo: Path) -> dict[str, str]:
    """当前树的四条工具轴（corpus 轴随语料目录另取）。"""
    def _read(rel: str) -> str | None:
        p = repo / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    return _axes_from_yaml_texts(
        _read("spec/spec.yaml"), _read("spec/tolerance.yaml"),
        _read("spec/feature_atoms.yaml"),
    )


def read_axes_at_commit(repo: Path, commit: str) -> dict[str, str]:
    """历史提交的四轴快照；当时不存在的轴如实标注。"""

    def _show(rel: str) -> str | None:
        return _git(repo, "show", f"{commit}:{_repository_path(repo, rel)}")

    return _axes_from_yaml_texts(
        _show("spec/spec.yaml"), _show("spec/tolerance.yaml"),
        _show("spec/feature_atoms.yaml"),
    )


def tag_name_for(corpus_version: str) -> str:
    """`v0.5-cov20-frozen` → `corpus-v0.5`（目录名后缀是生成策略描述）。"""
    m = re.match(r"(v\d+\.\d+)", corpus_version)
    if not m:
        raise ValueError(f"无法从版本名推导标签名：{corpus_version}")
    return f"corpus-{m.group(1)}"


# ---------- 勘误解析与过滤 ----------

def parse_errata(path: Path) -> list[dict]:
    """解析 docs/errata.md 的条目（发布说明按版本过滤收录）。

    格式约定：`### ERR-xxx 标题` 起一段，行内 `- 键：值`。
    """
    if not path.is_file():
        return []
    entries: list[dict] = []
    for block in path.read_text(encoding="utf-8").split("\n### ")[1:]:
        lines = block.splitlines()
        head = lines[0].strip()
        m = re.match(r"(ERR-\d+)\s+(.*)", head)
        if not m:
            continue
        entry = {"id": m.group(1), "title": m.group(2).strip()}
        for line in lines[1:]:
            if line.startswith("- ") and "：" in line:
                key, _, val = line[2:].partition("：")
                entry[key.strip()] = val.strip()
        # 受影响版本拆列表并去掉括注（如"（草稿期）"），供按版本过滤
        versions = [re.sub(r"（[^）]*）", "", v).strip()
                    for v in entry.get("受影响版本", "").split("、")]
        entry["versions"] = [v for v in versions if v]
        entries.append(entry)
    return entries


def errata_for(entries: list[dict], corpus_version: str) -> list[dict]:
    return [e for e in entries if corpus_version in e.get("versions", [])]


# ---------- 生成环境（来自生成日志） ----------

def read_generation_env(corpus_dir: Path) -> dict | None:
    """从 generation-log.yaml 取构建环境；无日志（早于机制上线）返回 None。"""
    path = corpus_dir / LOG_FILENAME
    if not path.is_file():
        return None
    events = read_log(path)
    start = next((e for e in events if e["step"] == "build_start"), None)
    if start is None:
        return None
    return {
        "git_commit": start.get("git_commit"),
        "python": start.get("python"),
        "torch": start.get("torch"),
        "platform": start.get("platform"),
        "events": len(events),
    }


# ---------- 发布说明 ----------

def _diff_summary(prev: CorpusManifest, cur: CorpusManifest) -> str:
    from .evolution import diff_manifests

    d = diff_manifests(prev, cur)
    return (f"用例数 {d['old']['cases']} → {d['new']['cases']}，"
            f"算子 +{len(d['operators_added'])} / −{len(d['operators_removed'])}，"
            f"用例 +{len(d['cases_added'])} / −{len(d['cases_removed'])} / "
            f"改 {len(d['cases_changed'])}")


def render_release_notes(
    *,
    manifest: CorpusManifest,
    axes: dict[str, str],
    gen_env: dict | None,
    errata: list[dict],
    backfill: bool,
    freeze_commit: str | None,
    prev_summary: str | None,
    verify_records: int,
    corpus_dir_name: str,
) -> str:
    """渲染一页发布说明（快照文档：不含当前时间，可复现）。"""
    tag = tag_name_for(manifest.version)
    out: list[str] = []
    out.append(f"# 测试集发布：{tag}")
    out.append("")
    if backfill:
        out.append("> **补录标签**：该版本的正式标签建立晚于版本冻结（2026-10-15 起"
                   "才有发布机制）。标签回溯打在**当前清单内容落地**的提交"
                   f"（{freeze_commit[:12] if freeze_commit else '未定位'}）上，"
                   "五轴快照取自该历史提交；本版本曾以草稿状态被真机轮次消费，"
                   "相关修订按勘误通道登记（docs/errata.md）。")
    else:
        out.append("> 正式发布。版本规则见 docs/versioning.md；本页由 "
                   "scripts/release.py 生成，与标签指向的提交对应。")
    out.append("")
    _render_release_axes(out, manifest, axes)

    _render_release_contents(out, manifest, gen_env, freeze_commit, corpus_dir_name)

    _render_release_changes(out, prev_summary, errata)

    out.append("## 验证状态（发布时点）")
    out.append("")
    out.append(f"- 真机验证记录 {verify_records} 份（verify-records/，"
               "含逐项判定与全量测试日志）")
    out.append("- 验证结论索引：docs/requirements-coverage-report.md"
               "「真机验证与证据归档」一节")
    out.append("")
    return "\n".join(out)


def _render_release_axes(out, manifest, axes):
    out.append("## 版本快照（五轴）")
    out.append("")
    out.append("| 轴 | 版本 |")
    out.append("|---|---|")
    out.append(f"| 测试集 | {manifest.version} |")
    out.append(f"| 评测规范 | {axes['spec_version']} |")
    out.append(f"| 评分器 | {axes['scorer_version']} |")
    out.append(f"| 容差表 | {axes['tolerance_version']} |")
    out.append(f"| 能力点清单 | {axes['fa_version']} |")
    if (manifest.feature_atoms_version
            and axes["fa_version"] != AXIS_ABSENT
            and manifest.feature_atoms_version != axes["fa_version"]):
        out.append("")
        out.append(f"注：清单内记录的能力点清单版本为 "
                   f"{manifest.feature_atoms_version}，与锚定提交的 FA "
                   f"清单（{axes['fa_version']}）不一致——清单内容是 schema "
                   "迁移时保留语义、重算指纹的产物（见 docs/corpus-evolution.md"
                   "与 tests/core/test_corpus.py 的迁移说明），两个记录各自如实保留。")
    out.append("")


def _render_release_contents(out, manifest, gen_env, freeze_commit, corpus_dir_name):
    out.append("## 冻结内容")
    out.append("")
    out.append(f"- 语料目录：`corpus/{corpus_dir_name}`")
    out.append(f"- 清单哈希：`{manifest.manifest_hash}`")
    out.append(f"- 冻结时间：{manifest.frozen_at}")
    out.append(f"- 用例 {len(manifest.cases)} 个 / 算子 {len(manifest.operators)} 个 / "
               f"门禁剔除 {len(manifest.excluded)} 条")
    if gen_env:
        out.append(f"- 生成器提交：`{gen_env['git_commit']}`；构建环境："
                   f"Python {gen_env['python']} / torch {gen_env['torch']} / "
                   f"{gen_env['platform']}（取自 generation-log.yaml，"
                   f"共 {gen_env['events']} 条构建事件）")
    else:
        out.append("- 生成环境：该版构建早于生成日志机制（2026-10-14 起），"
                   "构建环境不可查——构建事实由清单哈希与 git 历史承载")
    if freeze_commit:
        out.append(f"- 清单内容落地提交：`{freeze_commit}`")
    out.append("")


def _render_release_changes(out, prev_summary, errata):
    out.append("## 与上一版的差异")
    out.append("")
    if prev_summary:
        out.append(f"- {prev_summary}")
    else:
        out.append("- 首个版本，无上一版。")
    out.append("- 逐项差异（算子增删 / 用例级增删改 / 规模与 dtype 再分配 / "
               "覆盖原子得失）见 docs/corpus-evolution.md")
    out.append("")
    out.append("## 受影响勘误")
    out.append("")
    if errata:
        for e in errata:
            out.append(f"- **{e['id']}（{e.get('处置', '?')}）**：{e['title']}")
    else:
        out.append("- 无。")
    out.append("- 勘误全文与处置规则见 docs/errata.md")
    out.append("")
