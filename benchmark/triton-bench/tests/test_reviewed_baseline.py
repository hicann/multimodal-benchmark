# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""A reviewed-content approval cannot authorize a lost license header."""
import hashlib
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'baseline_guard', Path(__file__).resolve().parents[1] / 'scripts/check_reviewed_baseline.py'
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_rejects_loss_and_unreviewed_edits(tmp_path):
    data = guard.LICENSE + b'\n' + guard.COPYRIGHT + b'\npass\n'
    path = tmp_path / 'module.py'
    path.write_bytes(data)
    before = hashlib.sha256(data).hexdigest()
    inventory = {'module.py': {'sha256': before, 'cann_header': True}}
    assert not guard.audit(tmp_path, inventory, {})
    path.write_bytes(data + b'# reviewed edit\n')
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert guard.audit(tmp_path, inventory, {})[0]['reason'] == 'unreviewed-content-change'
    approved = {'module.py': {'before': before, 'after': after, 'reason': 'Add a documented feature'}}
    assert not guard.audit(tmp_path, inventory, approved)
    path.write_bytes(b'pass\n')
    approved['module.py']['after'] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert any(e['reason'] == 'lost-cann-header' for e in guard.audit(tmp_path, inventory, approved))
    path.unlink()
    assert guard.audit(tmp_path, inventory, approved)[0]['reason'] == 'missing-reviewed-file'


def test_removal_requires_exact_baseline_identity_and_explicit_reason(tmp_path):
    before = hashlib.sha256(b'reviewed').hexdigest()
    inventory = {'retired.sh': {'sha256': before, 'cann_header': True}}
    assert guard.audit(tmp_path, inventory, {})
    approval = {'retired.sh': {'before': before, 'after': None, 'removed': True,
                              'reason': 'User explicitly retired the periodic upload mechanism'}}
    assert not guard.audit(tmp_path, inventory, approval)
    approval['retired.sh']['before'] = '0' * 64
    assert guard.audit(tmp_path, inventory, approval)


def test_repository_retains_reviewed_baseline():
    import json

    root = Path(__file__).resolve().parents[1]
    quality = root / "docs/quality"
    baseline = json.loads((quality / "reviewed-baseline.json").read_text())
    approvals = json.loads((quality / "reviewed-changes.json").read_text())
    assert baseline["commit"] == "ba04485ebfedd785f64d5c3721e73d484606fb59"
    assert len(baseline["files"]) == 1403
    assert guard.audit(root, baseline["files"], approvals) == []
