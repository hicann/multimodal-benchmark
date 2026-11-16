#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Offline source secret guard. Reports only file/line/rule, never secret values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
# One-way identifiers of revoked/exposed credentials; never keep their plaintext.
FINGERPRINTS = {
    "9f3cf366b80d7cafe5e1c404615f539f36e2637116232fdeaa2c11aaf82c58bb",
    "6473e5f796ff380eca7eebd344f39bf60630945cc5a41413943888c2e05a099c",
}
RULES = {
    "credential-url": re.compile(rb"https?://[^\s/\x22\x27<>]+:[A-Za-z0-9_-]{20,}@"),
    "token-as-url-user": re.compile(rb"https?://[A-Za-z0-9_-]{20,}@"),
    "credential-default": re.compile(rb"\$\{(?:GITCODE_TOKEN|ACCESS_TOKEN|API_KEY|SECRET_KEY):-[A-Za-z0-9_-]{20,}\}"),
    "credential-assignment": re.compile(
        rb"^\s*(?:GITCODE_TOKEN|ACCESS_TOKEN|API_KEY|SECRET_KEY)\s*=\s*[\x22\x27][A-Za-z0-9_-]{20,}[\x22\x27]", re.M
    ),
    "oauth-json": re.compile(rb'["\x27](?:access_token|refresh_token)["\x27]\s*:\s*["\x27][A-Za-z0-9_.-]{20,}["\x27]'),
    "private-key": re.compile(rb"^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?$", re.M),
}
PLACEHOLDERS = (b"EXPOSED_CREDENTIAL_REMOVED", b"REVOKED_CREDENTIAL_REMOVED")


def findings(data):
    found = []
    for rule, pattern in RULES.items():
        for match in pattern.finditer(data):
            if not any(value in match.group() for value in PLACEHOLDERS):
                found.append((data[: match.start()].count(b"\n") + 1, rule))
    for match in re.finditer(rb"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{24,40}(?![A-Za-z0-9_-])", data):
        if hashlib.sha256(match.group()).hexdigest() in FINGERPRINTS:
            found.append((data[: match.start()].count(b"\n") + 1, "known-exposed-credential"))
    return sorted(set(found))


def git_output(arguments, **options):
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("Git executable is required")
    return subprocess.check_output([str(Path(executable).resolve()), *arguments], cwd=ROOT, **options)


def emit(value):
    import logging
    import sys

    logger = logging.Logger("triton_eval.secret_scan", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.info("%s", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--staged", action="store_true", help="scan the complete index snapshot, not working tree bytes"
    )
    args = parser.parse_args()
    if args.paths:
        paths = args.paths
    else:
        command = (
            ["git", "ls-files", "--cached", "-z"]
            if args.staged
            else ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        )
        paths = [Path(value.decode()) for value in git_output(command[1:]).split(b"\0") if value]
    prefix = git_output(["rev-parse", "--show-prefix"], text=True).strip()
    errors = []
    scanned = 0
    for path in sorted(set(paths)):
        full = path if path.is_absolute() else ROOT / path
        if not args.staged and full.is_symlink():
            continue
        if not args.staged and not full.is_file():
            continue
        try:
            data = (
                git_output(["show", ":" + prefix + path.as_posix()])
                if args.staged
                else full.read_bytes()
            )
        except (OSError, subprocess.CalledProcessError):
            errors.append(dict(path=str(path), line=0, rule="unreadable"))
            continue
        scanned += 1
        errors.extend(dict(path=str(path), line=line, rule=rule) for line, rule in findings(data))
    emit(json.dumps(dict(scanned_files=scanned, findings=errors), ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
