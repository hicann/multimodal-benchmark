# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from collections import Counter
from pathlib import Path
import math
import pytest
from triton_eval.corpusmod import load_manifest
from triton_eval.coverage import _case_signature, size_class
from triton_eval.workload_generation import TARGETS, PARENT, select_completion, candidate_pool

ROOT = Path(__file__).resolve().parents[2]


def test_completion_retains_baseline_and_covers_all_sizes_without_seed_padding():
    parent = load_manifest(ROOT / "corpus" / PARENT / "manifest.json")
    before = parent.model_dump_json()
    added, plan = select_completion(parent.cases)
    again, second = select_completion(parent.cases, list(reversed(candidate_pool())))
    assert [c.code for c in added] == [c.code for c in again]
    assert plan == second
    assert parent.model_dump_json() == before
    assert len(added) == 101
    signatures = [_case_signature(c) for c in [*parent.cases, *added]]
    assert len(signatures) == len(set(signatures))
    for op in TARGETS:
        rows = [c for c in [*parent.cases, *added] if c.operator == op]
        assert len(rows) == 20
        assert set(size_class(math.prod(c.shape)) for c in rows) == {"SMALL", "MEDIUM", "LARGE", "XLARGE"}
    assert all(
        row["new_feature_dtype_tier_pairs"] > 0 or row["previous_tier_count"] >= 1 for row in plan["selection_trace"]
    )


def test_missing_tier_or_budget_pool_does_not_silently_pass():
    parent = load_manifest(ROOT / "corpus" / PARENT / "manifest.json")
    with pytest.raises(ValueError):
        select_completion(parent.cases, [])


def test_new_domain_checks_actual_input_relations():
    torch = pytest.importorskip("torch")
    from triton_eval.workload_contracts import validate_inputs

    x = torch.zeros((256, 32))
    validate_inputs("GroupedMatmul", [x, torch.zeros((2, 32, 16)), torch.zeros((2, 16))], {"group_list": [128, 128]})
    with pytest.raises(ValueError):
        validate_inputs(
            "GroupedMatmul", [x, torch.zeros((1, 32, 16)), torch.zeros((2, 16))], {"group_list": [128, 128]}
        )


@pytest.mark.parametrize("mutation", ["retained_seed", "new_shape", "trace", "cpu_gate"])
def test_refingerprinted_release_tampering_is_not_accepted(tmp_path, mutation):
    import json
    import shutil
    from triton_eval.corpus_assets import freeze_assets
    from triton_eval.corpusmod import compute_case_content_hash, manifest_digest
    from triton_eval.workload_generation import VERSION
    from triton_eval.workload_release import static_readiness

    static_readiness(ROOT / "corpus" / VERSION)  # The authenticated source must pass before mutation.
    corpus = tmp_path / "copy"
    shutil.copytree(ROOT / "corpus" / VERSION, corpus)
    m = load_manifest(corpus / "manifest.json")
    plan = json.loads((corpus / "coverage-plan.json").read_text())
    from triton_eval.buildlog import read_log, write_log

    events = read_log(corpus / "generation-log.yaml")
    if mutation == "retained_seed":
        m.cases[0].seed += 1
    elif mutation == "new_shape":
        m.cases[-1].shape[0] += 1
    elif mutation == "trace":
        plan["selection_trace"] = []
    else:
        next(e for e in events if e.get("step") == "independent_cpu_reference_gate")["failures"] = {"case": "bad"}
    for c in m.cases:
        c.content_hash = compute_case_content_hash(corpus, c)
    m.manifest_hash = manifest_digest(m)
    events[-1]["manifest_hash"] = m.manifest_hash
    (corpus / "manifest.json").write_text(m.model_dump_json())
    (corpus / "coverage-plan.json").write_text(json.dumps(plan))
    write_log(corpus / "generation-log.yaml", events)
    (corpus / "evaluation-assets.json").unlink()
    # A changed case manifest may already be refused by authenticated predecessor validation.
    with pytest.raises(ValueError):
        freeze_assets(corpus, m.manifest_hash)
        static_readiness(corpus)
