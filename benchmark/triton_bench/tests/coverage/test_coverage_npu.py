# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real device tests for new supplements; CPU runs explicitly skip these."""
from pathlib import Path

import pytest

from triton_eval.corpusmod import load_manifest

from triton_eval.fixture_sources import fixture_path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / 'corpus/v0.16-backend-pattern-certification-candidates'
MANIFEST = load_manifest(CORPUS / 'manifest.json')
# Reuse existing carriers; implementation coverage adds no input cases.
IMPLEMENTATION_ATOMS = ('FA-PAT-VEC-LOAD', 'FA-PAT-MASK-TAIL', 'FA-PROG-GRID3D', 'FA-PROG-GRID4D',
                        'FA-PAT-COALESCE-GATHER', 'FA-PAT-TILE-GEMM', 'FA-FUSE-INTERM-ELIDE')
ELIGIBLE_OPERATORS = ('Neg', 'Gather', 'Matmul', 'MaskedSoftmax')
ELIGIBLE_CASES = [case for case in MANIFEST.cases if case.operator in ELIGIBLE_OPERATORS]
PROOF_CASES = [next(case for case in ELIGIBLE_CASES if atom in case.feature_atoms) for atom in IMPLEMENTATION_ATOMS]


def _eligible_case_for(atom):
    return next(case for case in ELIGIBLE_CASES if atom in case.feature_atoms)


CASES = [_eligible_case_for(atom) for atom in IMPLEMENTATION_ATOMS]
CASES = list({c.code: c for c in [*MANIFEST.cases[1360:], *PROOF_CASES]}.values())
pytestmark = pytest.mark.npu


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.code)
def test_coverage_extension_compiles_executes_and_records_artifacts(case, tmp_path):
    from triton_eval.coverage_gate import static_readiness
    from triton_eval.coverage_evidence import observation
    from triton_eval.harness.npu_exec import execute_case
    from triton_eval.judge import decide, tolerance_for_case
    from triton_eval.models import CaseStatus
    from triton_eval.oracle import active_corpus, configure_corpus
    from triton_eval.refcheck import check_case
    from triton_eval.runner import execute_kwargs
    from triton_eval.specload import load_tolerance
    from triton_eval.submission import discover, load_all

    static_readiness(CORPUS)
    previous = active_corpus()
    configure_corpus(CORPUS)
    try:
        assert check_case(case, 'npu:0') is None
        submission = load_all(discover(fixture_path(ROOT / 'examples/coverage-v16-ascend-submission')))
        assert case.operator in submission.loaded, submission.failures
        capture = tmp_path / 'compilation' / case.code
        measurement = execute_case(case, **execute_kwargs(case, submission.loaded[case.operator].func),
                                   compilation_dir=str(capture),
                                   evidence_identity={'manifest_hash': MANIFEST.manifest_hash, 'run_id': 1})
        status = decide(measurement, tolerance_for_case(load_tolerance(), case, measurement))
        assert status == CaseStatus.PASS, (status, measurement.compile_err)
        proof = observation(case, capture / 'observation.json', MANIFEST.manifest_hash, 1)
        assert proof['data']['synchronized']
        assert proof['data']['input_bindings'] and proof['data']['output_bindings']
        # These are real device assertions. Unsupported/missing lowering is an
        # actionable failure; it is never converted into a synthetic PASS.
        from triton_eval.coverage_evidence import feature_witness
        for atom, carrier in zip(IMPLEMENTATION_ATOMS, PROOF_CASES):
            if carrier.code == case.code:
                ok, reason = feature_witness(atom, proof)
                assert ok, (atom, reason, proof.get('implementation_witnesses'))
    finally:
        configure_corpus(previous)
