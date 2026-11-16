# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reject oversized staged blobs and blobs in commits pending publication."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path


def git(repository, *arguments, content=None):
    executable = shutil.which('git')
    if executable is None:
        raise FileNotFoundError('Git executable is not installed or not on PATH')
    return subprocess.run([str(Path(executable).resolve()), '-C', str(repository), *arguments], input=content,
                          capture_output=True, check=True, timeout=120).stdout


def oversized_blobs(repository: Path, base: str | None, limit: int) -> list[tuple[int, str]]:
    objects = {}
    for entry in git(repository, 'ls-files', '--stage', '-z').split(b'\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        mode, identity, stage = metadata.split()
        if stage != b'0':
            raise ValueError('Unmerged index; do not publish')
        if mode != b'160000':
            objects.setdefault(identity, name)
    if base is not None:
        ancestor = git(repository, 'rev-parse', '--verify', base + '^{commit}').strip()
        git(repository, 'merge-base', '--is-ancestor', ancestor.decode(), 'HEAD')
        for row in git(repository, 'rev-list', '--objects', ancestor.decode() + '..HEAD').splitlines():
            identity, _, name = row.partition(b' ')
            objects.setdefault(identity, name)
    if not objects:
        return []
    result = git(repository, 'cat-file', '--batch-check',
                 content=b'\n'.join(objects) + b'\n')
    failures = []
    for row in result.splitlines():
        identity, kind, size = row.split()
        if kind == b'blob' and int(size) >= limit:
            failures.append((int(size), objects[identity].decode('utf-8', errors='replace')))
    return sorted(failures, reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--base')
    parser.add_argument('--limit-mib', type=int, default=100)
    args = parser.parse_args()
    if args.limit_mib <= 0:
        parser.error('--limit-mib must be positive')
    failures = oversized_blobs(args.repo, args.base, args.limit_mib * 1024 * 1024)
    if failures:
        for size, name in failures:
            logging.info('REJECT %s bytes: %s', size, name)
        logging.info('Amend unpublished oversized commits; a later small-file commit is not sufficient.')
        return 1
    logging.info('PASS: staged files and pending commit blobs are below the size limit.')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    raise SystemExit(main())
