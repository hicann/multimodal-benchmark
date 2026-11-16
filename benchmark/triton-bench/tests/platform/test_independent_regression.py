# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""New submissions must use all existing frozen cases and carry source only."""
import importlib.util
from collections import Counter
from pathlib import Path
import shutil
import zipfile

import pytest
from test_implementation_kernel_semantics import cpu_kernels as _cpu_kernels_fixture

from triton_eval.corpusmod import load_manifest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_independent_regression',
                                               ROOT / 'scripts/platform_examples/prepare_independent_regression.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_plan_uses_exact_existing_frozen_cases_without_creating_a_corpus(tmp_path):
    corpus = builder.CORPUS
    tracked = {p: p.read_bytes() for p in corpus.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    out = tmp_path / 'prepared'
    plan = builder.prepare(out)
    manifest = load_manifest(corpus / 'manifest.json')
    selected = [c for c in manifest.cases if c.operator in plan['operators']]
    assert len(plan['operators']) == 20 and len(selected) == 406
    assert plan['case_codes'] == [c.code for c in selected]
    assert plan['cases_by_operator'] == dict(Counter(c.operator for c in selected))
    assert plan['manifest_hash'] == manifest.manifest_hash
    assert not (out / 'corpus').exists()
    assert all(p.read_bytes() == original for p, original in tracked.items())


def test_archive_is_source_only_and_cannot_overwrite_an_old_revision(tmp_path):
    source = tmp_path / 'source'
    shutil.copytree(builder.SOURCE, source)
    output = tmp_path / 'submission.zip'
    first = builder.package(source, output, submit_id='regression-r1')
    assert first == builder.package(source, output, submit_id='regression-r1')
    assert first['readiness']['ok'] and first['readiness']['source_checked'] == 20
    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == ['manifest.yaml', 'operators/matrices.py',
                                             'operators/pointwise.py', 'operators/rows.py']
        assert not any(b'triton_eval' in archive.read(name) for name in archive.namelist())
    original = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        builder.package(source, output, submit_id='regression-r2')
    assert output.read_bytes() == original


@pytest.mark.parametrize('cpu_kernels', [builder.SOURCE / 'operators/rows.py'], indirect=True)
@pytest.mark.parametrize('shape', [(1, 1048576), (2, 1048833)])
@pytest.mark.parametrize('keepdim', [False, True])
def test_new_reduce_max_handles_existing_million_element_cases_and_masked_tail(cpu_kernels, shape, keepdim):
    # CPU interpretation of actual new source; it is not compilation/device evidence.
    import torch
    x = -(torch.rand(shape, generator=torch.Generator().manual_seed(123)) + 2)
    x[:, -1] = -.5  # All negative: a zero-padded partial block would give the wrong maximum.
    x[0, 4096] = -.25
    original = x.clone()
    actual = cpu_kernels['evaluate_max'](x, dim=-1, keepdim=keepdim)
    assert torch.equal(actual, torch.amax(x, dim=-1, keepdim=keepdim))
    assert torch.equal(x, original)


# Give the imported fixture an explicit public name without shadowing test arguments.
interpreted_kernels = pytest.fixture(name="cpu_kernels")(_cpu_kernels_fixture.__wrapped__)
