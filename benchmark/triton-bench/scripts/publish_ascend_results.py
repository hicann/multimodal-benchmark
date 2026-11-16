#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""将真机证据发布到独立的滚动结果分支，不操作工程工作区的索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_eval.console import emit

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "npu-results"
MAX_BYTES = 50 * 1024 * 1024


def redact(message):
    def clean_url(match):
        url = match.group(0)
        try:
            parsed = urlsplit(url)
            netloc = parsed.netloc.rsplit("@", 1)[-1]
            query = parse_qsl(parsed.query, keep_blank_values=True)
            sensitive = {"token", "access_token", "private_token", "password", "secret", "auth", "api_key"}
            if any(key.lower() in sensitive for key, _ in query):
                query = urlencode([(key, "<redacted>" if key.lower() in sensitive else value)
                                   for key, value in query])
            else:
                query = parsed.query
            if netloc == parsed.netloc and query == parsed.query:
                return url
            return urlunsplit(parsed._replace(netloc=netloc, query=query))
        except ValueError:
            return "<remote-url-redacted>"

    # Public provenance URLs are frozen evidence. Do not consume JSON quotes or
    # replace these URLs: doing so corrupts JSON and its recorded SHA256.
    message = re.sub(r"https?://[^\s<>\"']+", clean_url, message)
    secret = os.environ.get("GITCODE_TOKEN")
    if secret:
        message = message.replace(secret, "<redacted>")
    return message


def git(directory, *arguments, input_text=None):
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("Git executable is unavailable")
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0",
                       GIT_AUTHOR_NAME="ascend-suite", GIT_AUTHOR_EMAIL="ascend-suite@local",
                       GIT_COMMITTER_NAME="ascend-suite", GIT_COMMITTER_EMAIL="ascend-suite@local")
    result = subprocess.run([str(Path(executable).resolve()), "-C", str(directory), *arguments], input=input_text,
                            capture_output=True, text=True, env=environment, timeout=120)
    if result.returncode:
        raise RuntimeError(redact(result.stderr or result.stdout or "git command failed"))
    return result.stdout.strip()


def push_url(repository):
    origin = os.environ.get("TRITON_EVAL_PUSH_URL") or git(repository, "remote", "get-url", "--push", "origin")
    parsed = urlsplit(origin)
    if parsed.scheme in ("http", "https") and (parsed.username or parsed.password):
        raise ValueError("禁止在仓库 URL 中携带凭据；请使用 SSH 或 Git credential helper")
    if any(key.lower() in {"token", "access_token", "private_token", "password", "secret", "auth", "api_key"}
           for key, _ in parse_qsl(parsed.query)):
        raise ValueError("禁止在仓库 URL 查询参数中携带凭据")
    return origin


def copy_evidence(source, destination):
    total = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"结果中不允许符号链接：{path.name}")
        if not path.is_file() or path.suffix in (".sqlite", ".sqlite3", ".db", ".pyc"):
            continue
        relative = path.relative_to(source)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        total += path.stat().st_size
        if total > MAX_BYTES:
            raise ValueError("单轮证据超过 50 MiB，停止上传；完整原件仍在本机")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        try:
            content = redact(raw.decode("utf-8")).encode("utf-8")
        except UnicodeDecodeError:
            content = raw
        target.write_bytes(content)
    verify_published_corpora(destination)


def verify_published_corpora(directory):
    """Reject a publication that changes any portable frozen corpus asset."""
    for path in directory.rglob("evaluation-assets.json"):
        corpus = path.parent
        if not (corpus / "official-catalog.json").is_file():
            continue  # A run snapshot need not contain the original corpus files.
        assets = json.loads(path.read_text(encoding="utf-8"))
        for name, expected in assets.get("files", {}).items():
            if not name.startswith("@corpus/"):
                continue
            asset = (corpus / name.removeprefix("@corpus/")).resolve()
            if not asset.is_relative_to(corpus.resolve()) or not asset.is_file():
                raise ValueError(f"冻结资产缺失或越界：{name}")
            if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
                raise ValueError(f"上传会改变冻结资产指纹，停止发布：{name}")


def trim_records(directory, keep, current):
    records = sorted((path for path in directory.iterdir() if path.is_dir()),
                     key=lambda path: path.name, reverse=True)
    retained = {current}
    for path in records:
        if len(retained) >= keep:
            break
        retained.add(path.name)
    for path in records:
        if path.name not in retained:
            shutil.rmtree(path)


def publish(repository, record, keep=3):
    if keep < 1:
        raise ValueError("keep 必须至少为 1")
    record = record.resolve()
    if not (record / "execution.json").is_file():
        raise ValueError("缺少 execution.json，拒绝上传非运行目录")
    remote = push_url(repository)
    reference = f"refs/heads/{BRANCH}"
    with tempfile.TemporaryDirectory(prefix="ascend-results-") as temporary:
        checkout = Path(temporary)
        git(checkout, "init", "-q")
        git(checkout, "remote", "add", "origin", remote)
        advertised = git(checkout, "ls-remote", "origin", reference)
        previous = advertised.split()[0] if advertised else ""
        if previous:
            git(checkout, "fetch", "--depth=1", "origin", reference)
            git(checkout, "checkout", "--detach", "FETCH_HEAD")
            if git(checkout, "rev-parse", "HEAD") != previous:
                raise RuntimeError("结果分支已被其他运行更新，请重试上传")
            if not (checkout / "rolling-results.json").is_file():
                raise ValueError("npu-results 已存在且不是本机制创建，拒绝覆盖")
        records = checkout / "records"
        records.mkdir(exist_ok=True)
        destination = records / record.name
        if destination.exists():
            shutil.rmtree(destination)
        copy_evidence(record, destination)
        trim_records(records, keep, record.name)
        index = {"format": 1, "latest": record.name, "keep": keep,
                 "records": sorted(path.name for path in records.iterdir())}
        (checkout / "rolling-results.json").write_text(json.dumps(index, indent=2) + "\n")
        git(checkout, "add", "-A")
        tree = git(checkout, "write-tree")
        commit = git(checkout, "commit-tree", tree, input_text=f"npu: publish {record.name}\n")
        git(checkout, "push", f"--force-with-lease={reference}:{previous}",
            "origin", f"{commit}:{reference}")
    (record / ".uploaded").write_text(commit + "\n")
    local_records = sorted((path for path in record.parent.iterdir()
                            if path.is_dir() and (path / ".uploaded").is_file()),
                           key=lambda path: path.name, reverse=True)
    retained = {record} | set([path for path in local_records if path != record][:keep - 1])
    for path in local_records:
        if path not in retained:
            shutil.rmtree(path)
    emit(f"[upload] 已上传 GitCode {BRANCH}：records/{record.name}（最近 {keep} 轮）")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--keep", type=int, default=3)
    args = parser.parse_args()
    try:
        publish(args.repo.resolve(), args.record, args.keep)
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        emit(f"[upload] 失败，原件保留，可用本命令补传：{redact(str(error))}")
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
