# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Allowlisted, fingerprinted engine bundles and verified result archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from triton_eval.corpus_assets import FRAMEWORK_FILES, ROOT, verify_assets, reference_sources
from triton_eval import framework_path
from triton_eval.corpusmod import load_manifest, verify_corpus_files
from triton_eval.remote_jobs import canonical
from triton_eval.submission_service import ServiceError


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _framework_dependency(name):
    source = framework_path(name, ROOT).resolve()
    if source.is_relative_to(ROOT.resolve()):
        return source.relative_to(ROOT).as_posix(), source
    from triton_eval.fixture_sources import fixture_files

    outputs = fixture_files(FRAMEWORK_FILES, ROOT)
    if name not in outputs or outputs[name].resolve() != source:
        raise ServiceError("BUNDLE-FILE", "冻结资产路径越界")
    return name, source


def _source_tree_members(tree, suffixes):
    result = {}
    for path in (ROOT / tree).rglob("*"):
        if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
            result[path.relative_to(ROOT).as_posix()] = path
    return result


def _engine_members():
    result = dict(_framework_dependency(name) for name in FRAMEWORK_FILES)
    for tree, suffixes in [
        ("src/triton_eval", {".py", ".j2"}),
        ("env", {".sh"}),
        ("spec", {".yaml", ".yml", ".json"}),
        ("scripts/example_sources", {".py", ".txt", ".json"}),
    ]:
        result.update(_source_tree_members(tree, suffixes))
    return result


def _corpus_members(corpus, assets):
    references = reference_sources(corpus)
    result = {
        "job-corpus/manifest.json": corpus / "manifest.json",
        "job-corpus/evaluation-assets.json": corpus / "evaluation-assets.json",
    }
    for name in assets["files"]:
        if name.startswith("@corpus/"):
            relative = name.removeprefix("@corpus/")
            result["job-corpus/" + relative] = references.get(relative, corpus / relative)
        else:
            destination, source = _framework_dependency(name)
            result[destination] = source
    for name, source in references.items():
        result["job-corpus/" + name] = source
    return result


def _stage_bundle_members(files, staging):
    index = {}
    for name, source in sorted(files.items()):
        if source.is_symlink() or not source.is_file():
            raise ServiceError("BUNDLE-FILE", "打包依赖缺失或为符号链接")
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        index[name] = sha256(target)
    return index


def _write_bundle_archive(staging, names, output):
    import gzip

    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                _add_bundle_members(archive, staging, names)


def _add_bundle_members(archive, staging, names):
    for name in sorted(names):
        path = staging / name
        info = tarfile.TarInfo(name)
        info.size = path.stat().st_size
        info.mode = 0o600
        with path.open("rb") as stream:
            archive.addfile(info, stream)


def build_bundle(service, task: dict, job: dict, output: Path) -> dict:
    corpus = service.corpus
    manifest = load_manifest(corpus / "manifest.json")
    verify_corpus_files(corpus, manifest)
    verify_assets(corpus, manifest.manifest_hash)
    if manifest.manifest_hash != task["manifest_hash"]:
        raise ServiceError("CORPUS-CHANGED", "测试集指纹与任务不一致")
    package = service.root / "submissions" / task["submission_id"] / "submission.zip"
    if sha256(package) != task["package_sha256"]:
        raise ServiceError("PACKAGE-CHANGED", "上传后的包指纹发生变化")
    assets = json.loads((corpus / "evaluation-assets.json").read_text())
    files = _engine_members()
    files.update(_corpus_members(corpus, assets))
    files.update(
        {
            "submission.zip": package,
            "pyproject.toml": ROOT / "pyproject.toml",
            "scripts/run-job.sh": ROOT / "scripts/run-job.sh",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bundle-", dir=output.parent) as temporary:
        staging = Path(temporary)
        index = _stage_bundle_members(files, staging)
        spec = dict(
            protocol_version=1,
            job_id=job["id"],
            task=task,
            corpus="job-corpus",
            manifest_hash=manifest.manifest_hash,
            assets_hash=assets["assets_hash"],
            files=index,
        )
        (staging / "job.json").write_text(canonical(spec) + "\n")
        _write_bundle_archive(staging, [*index, "job.json"], output)
    return dict(
        bundle_sha256=sha256(output),
        bundle_bytes=output.stat().st_size,
        manifest_hash=manifest.manifest_hash,
        assets_hash=assets["assets_hash"],
        files=index,
    )


def verify_job(directory: Path) -> dict:
    spec = json.loads((directory / "job.json").read_text())
    if spec.get("protocol_version") != 1:
        raise ValueError("Unsupported bundle protocol")
    for name, digest in spec["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid bundle path")
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or sha256(path) != digest
        ):
            raise ValueError("Bundle fingerprint mismatch")
    corpus = directory / spec["corpus"]
    manifest = load_manifest(corpus / "manifest.json")
    verify_corpus_files(corpus, manifest)
    verify_assets(corpus, manifest.manifest_hash)
    assets = json.loads((corpus / "evaluation-assets.json").read_text())
    if (
        manifest.manifest_hash != spec["manifest_hash"]
        or assets["assets_hash"] != spec["assets_hash"]
    ):
        raise ValueError("Corpus identity mismatch")
    if sha256(directory / "submission.zip") != spec["task"]["package_sha256"]:
        raise ValueError("Submission identity mismatch")
    return spec


def restore_archive(archive_path: Path, destination: Path, final: dict, *, max_bytes: int) -> None:
    if (
        archive_path.stat().st_size > max_bytes
        or archive_path.stat().st_size != final.get("archive_bytes")
        or sha256(archive_path) != final.get("archive_sha256")
    ):
        raise ServiceError("REMOTE-ARCHIVE", "结果包大小或指纹不匹配", 409)
    expected = final.get("files")
    if not isinstance(expected, dict) or not expected or "engine.sqlite" not in expected:
        raise ServiceError("REMOTE-ARCHIVE", "结果包缺少运行身份数据库", 409)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="restore-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        seen, total = set(), 0
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                name, total = _restore_member(archive, member, staging, expected, seen, total, max_bytes)
                seen.add(name)
        if seen != set(expected):
            raise ServiceError("REMOTE-ARCHIVE", "结果包缺少索引成员", 409)
        if destination.exists():
            raise ServiceError("REMOTE-ARCHIVE", "结果目录已存在，不覆盖历史证据", 409)
        os.replace(staging, destination)


def _restore_member(archive, member, staging, expected, seen, total, max_bytes):
    name = member.name
    relative = PurePosixPath(name)
    total += member.size
    invalid_path = not member.isfile() or relative.is_absolute() or ".." in relative.parts
    invalid_membership = name not in expected or name in seen
    invalid_size = total > max_bytes or member.size != expected.get(name, {}).get("bytes")
    if invalid_path or invalid_membership or invalid_size:
        raise ServiceError("REMOTE-ARCHIVE", "结果包路径、成员或解包大小无效", 409)
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.extractfile(member) as source, target.open("wb") as stream:
        shutil.copyfileobj(source, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if sha256(target) != expected[name].get("sha256"):
        raise ServiceError("REMOTE-ARCHIVE", "结果文件指纹不一致", 409)
    return name, total
        # TemporaryDirectory tolerates its staging directory being renamed away.
