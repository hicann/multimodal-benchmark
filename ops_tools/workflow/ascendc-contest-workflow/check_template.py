# Copyright (c) 2026 multimodal-benchmark contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check candidate files against a downloaded CANNJudge template manifest."""

import argparse
import filecmp
import json
import logging
import sys
from pathlib import Path, PurePosixPath


def template_path(name):
    """Return a portable, relative manifest path."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid template path: {name!r}")
    if "\\" in name or ":" in name:
        raise ValueError(f"Invalid template path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts:
        raise ValueError(f"Invalid template path: {name!r}")
    return path


def project_file(root, path):
    """Resolve a manifest entry without following links outside the project."""
    file_path = root / path
    if not file_path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes project directory: {path}")
    if not file_path.is_file():
        raise ValueError(f"Missing project file: {file_path}")
    return file_path


def check_template(manifest, template, candidate):
    """Return changed editable files and modified read-only files."""
    if not isinstance(manifest, dict):
        raise ValueError("Expected a template manifest object")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a nonempty files list in the template manifest")
    editable_changed = []
    readonly_changed = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Expected each manifest entry to be an object")
        editable = entry.get("editable")
        if not isinstance(editable, bool):
            raise ValueError("Expected each manifest entry to have an editable boolean")
        name = entry.get("path")
        path = template_path(name)
        if path in seen:
            raise ValueError(f"Duplicate template entry: {name}")
        seen.add(path)
        original = project_file(template, path)
        proposed = project_file(candidate, path)
        if filecmp.cmp(original, proposed, shallow=False):
            continue
        changed = editable_changed if editable else readonly_changed
        changed.append(name)
    return {"editable_changed": editable_changed, "readonly_changed": readonly_changed}


def main():
    """Run the template comparison and print its JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        result = check_template(manifest, args.template.resolve(), args.candidate.resolve())
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(result.get("readonly_changed")))


if __name__ == "__main__":
    sys.exit(main())
