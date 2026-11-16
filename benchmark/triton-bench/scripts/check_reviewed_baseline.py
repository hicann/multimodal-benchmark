#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Reject unreviewed loss of files, headers or structural fixes from a known tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

LICENSE = b"CANN Open Software License Agreement Version 2.0"
COPYRIGHT = b"Huawei Technologies Co., Ltd."


def _has_reason(approval):
    return isinstance(approval.get("reason"), str) and bool(approval["reason"].strip())


def _approved_removal(path, entry, approval):
    if path.exists() or path.is_symlink():
        return False
    if approval.get("removed") is not True or approval.get("before") != entry["sha256"]:
        return False
    if "after" not in approval or approval["after"] is not None:
        return False
    return _has_reason(approval)


def _approved_change(entry, digest, approval):
    if approval.get("before") != entry["sha256"] or approval.get("after") != digest:
        return False
    return _has_reason(approval)


def _has_header(data):
    return LICENSE in data[:3000] and COPYRIGHT in data[:3000]


def _file_findings(platform, name, entry, approval):
    path = platform / name
    if not path.is_file() or path.is_symlink():
        if _approved_removal(path, entry, approval):
            return []
        return [dict(path=name, reason="missing-reviewed-file")]
    data = path.read_bytes()
    errors = []
    if entry["cann_header"] and not _has_header(data):
        errors.append(dict(path=name, reason="lost-cann-header"))
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"] and not _approved_change(entry, digest, approval):
        errors.append(dict(path=name, reason="unreviewed-content-change"))
    return errors


def audit(platform: Path, baseline: dict, approved: dict) -> list[dict]:
    errors = []
    for name, entry in sorted(baseline.items()):
        errors.extend(_file_findings(platform, name, entry, approved.get(name, {})))
    return errors


def _write_report(report):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.Logger("triton_eval.reviewed_baseline", level=logging.INFO)
    logger.addHandler(handler)
    logger.info("%s", json.dumps(report, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--approved", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    approved = json.loads(args.approved.read_text()) if args.approved else {}
    errors = audit(args.platform, baseline["files"], approved)
    _write_report(dict(baseline=baseline["commit"], findings=errors))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
