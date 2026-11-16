#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Record pinned official source hashes without importing any upstream kernel."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
import logging

import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from triton_eval.official_catalog import CANDIDATES, SOURCE_COMMIT, SOURCE_REPOSITORY
from triton_eval.console import emit


NOTES = {
    "Exp": "The official primitive is exp(x); base/scale/shift are explicit CANN Bench attributes.",
    "Log1p": "Use the pure log1p API example; test_log1p.py is an add+log1p fusion.",
    "Mod": "Truncating remainder (fmod), not Python floor remainder.",
    "FloorDiv": "Python floor division; signed Triton truncating division needs a remainder correction.",
    "BitwiseNot": "Use test_invert.py (~integer), not test_not.py (boolean logical NOT).",
    "Gather": "Use the official tl.gather API example, not test_18_gather.py (row embeddings).",
    "Concat": "Use ordered concat in test_cat_dim.py; test_cat.py permits reordering.",
    "SwiGlu": (
        "Official test has separate gate/multiplier inputs; CANN Bench specifies packed halves. The adapter "
        "splits one packed input without changing arithmetic."
    ),
    "SortValues": "Values only, matching the official extension.sort test. No index-output claim.",
    "TopKValues": "Values only, matching official tl.topk. Distinct from CANN Bench TopK with indices.",
    "LayerNorm": "Forward output y, matching the official wrapper; scratch mean/rstd are internal.",
    "Mish": "Secondary source: vendored CANN Bench proto, golden and real Triton example.",
}


def audit(cache=None):
    sources = []
    for candidate in CANDIDATES:
        local = candidate.source.startswith("reference/")
        if local:
            path = ROOT / candidate.source
            data = path.read_bytes()
            url = None
        else:
            url = SOURCE_REPOSITORY + "/blob/" + SOURCE_COMMIT + "/" + candidate.source
            data = _read_upstream_source(candidate, cache)
        entry = {
            "name": candidate.name,
            "path": candidate.source,
            "repository": "vendored CANN Bench" if local else SOURCE_REPOSITORY,
            "commit": None if local else SOURCE_COMMIT,
            "url": url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "expression": candidate.expression,
            "notes": NOTES.get(candidate.name, "Direct mathematical operation; 2D contiguous qualification domain."),
            "upstream_presence_is_npu_qualification": False,
        }
        if not local:
            tree = ast.parse(data.decode("utf-8"))
            entry["triton_calls"] = sorted(_triton_calls(tree))
            entry["skip_annotations"] = _skip_annotations(data)
        if candidate.name in ("Mish", "SwiGlu", "Exp"):
            task = {"Mish": "mish", "SwiGlu": "swi_glu", "Exp": "exp"}[candidate.name]
            asset_root = ROOT / f"reference/cann-bench/tasks/level1/{task}"
            paths = [asset_root / filename for filename in ("proto.yaml", "golden.py")]
            entry["cann_bench_assets"] = _source_digests(paths)
        sources.append(entry)
    return {"format": 1, "source_commit": SOURCE_COMMIT, "operators": sources}


def _read_upstream_source(candidate, cache):
    if cache:
        data = (cache / (candidate.name + ".py")).read_bytes()
    else:
        raw = (
            "https://raw.githubusercontent.com/triton-lang/triton-ascend/"
            + SOURCE_COMMIT
            + "/"
            + candidate.source
        )
        with urllib.request.urlopen(raw, timeout=60) as response:
            data = response.read()
    return data


def _triton_calls(tree):
    calls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = ast.unparse(node.func)
        if name.split(".")[0] in ("tl", "libdevice", "al", "extension"):
            calls.add(name)
    return calls


def _skip_annotations(data):
    annotations = []
    for line in data.decode("utf-8").splitlines():
        if "pytest.mark.skip" in line or "pytest.mark.xfail" in line:
            annotations.append(line.strip())
    return annotations


def _source_digests(paths):
    digests = {}
    for path in paths:
        digests[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reference/ascend-official/sources-v08.json")
    args = parser.parse_args()
    payload = audit(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out.exists() and args.out.read_text(encoding="utf-8") != text:
        raise ValueError("source lock already exists with different contents; publish a new revision")
    args.out.write_text(text, encoding="utf-8")
    emit(f"Verified {len(payload['operators'])} source mappings at {SOURCE_COMMIT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
