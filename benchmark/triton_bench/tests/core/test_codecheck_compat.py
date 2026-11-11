# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compatibility regressions; synthetic evidence is never new NPU certification."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_eval.buildlog import GenerationLog, read_log, write_log
from triton_eval.console import emit
from triton_eval.corpus_assets import (
    copy_reference_tree, reference_layout, resolve_reference_path,
    snapshot, verify_assets, verify_origin_assets,
)
from triton_eval.corpusmod import load_manifest
from triton_eval.pattern_evidence import pattern_witnesses
from triton_eval.reviewed_kernels import canonical_launch, parameter_groups
from test_implementation_evidence import neg, verified
from test_target_patterns import bias_observation, max_observation, scan_observation, split_observation

from triton_eval.fixture_sources import fixture_path
from triton_eval.migration_metadata import decode_metadata

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus/v0.18-scan-certification-candidates"


def _positive_control_cases(dtype, strided):
    import torch
    import torch.nn.functional as functional

    main = torch.tensor([[-2, -1, 1, 2], [3, 0, 3, -2]], dtype=dtype)
    if strided:
        storage = torch.empty(2, 8, dtype=dtype)
        storage[:, ::2] = main
        main = storage[:, ::2]
    other = main.flip(-1)
    indices = torch.tensor([[0, 2, 1, 3], [1, 3, 0, 2]])
    mask = main > 0
    gamma = torch.tensor([1, 2, -1, 0.5], dtype=dtype)
    weights = torch.arange(24, dtype=dtype).reshape(2, 4, 3) / 24
    bias = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=dtype)
    targets = torch.tensor([1, 3])
    return {
        'Exp': ([main], dict(scale=0.5, shift=-0.25), torch.exp(main * 0.5 - 0.25)),
        'Gelu': ([main], {}, functional.gelu(main.double()).to(dtype)),
        'Cummin': ([main], dict(dim=-1), tuple(torch.cummin(main, -1))),
        'Maximum': ([main, other], {}, torch.maximum(main, other)),
        'MaskedScale': ([main, mask], dict(scale=2.5), main * mask * 2.5),
        'Softmax': ([main], dict(dim=0), torch.softmax(main, 0)),
        'RmsNorm': ([main, gamma], dict(epsilon=0.25),
                    main * torch.rsqrt((main * main).mean(-1, keepdim=True) + 0.25) * gamma),
        'Gather': ([main, indices], dict(dim=-1), torch.gather(main, -1, indices)),
        'Scatter': ([main, indices, other], dict(dim=-1), main.scatter(-1, indices, other)),
        'GroupedMatmul': ([main, weights, bias, [1, 1]], {},
                          torch.stack([main[group] @ weights[group] + bias[group] for group in range(2)])),
        'Transpose': ([main], dict(perm=[1, 0]), main.T.contiguous()),
        'ArgMax': ([main], dict(dim=-1, keepdim=True), torch.argmax(main, -1, keepdim=True)),
        'CrossEntropyLoss': ([main, targets], dict(reduction='sum'),
                             functional.cross_entropy(main, targets, reduction='sum')),
        'SwiGlu': ([main], dict(dim=-1), functional.silu(main[:, :2]) * main[:, 2:]),
    }


@pytest.fixture(scope='module')
def positive_control_submission():
    from triton_eval.submission import discover, load_all
    from triton_eval.validator import validate_submission

    package = fixture_path(ROOT / 'examples/adversarial-submissions/ADV-C1-fullhouse')
    assert validate_submission(package).errors == []
    submission = load_all(discover(package))
    assert submission.failures == {}
    assert len(submission.loaded) == 14
    return submission


@pytest.mark.parametrize('operator', [
    'Exp', 'Gelu', 'Cummin', 'Maximum', 'MaskedScale', 'Softmax', 'RmsNorm',
    'Gather', 'Scatter', 'GroupedMatmul', 'Transpose', 'ArgMax', 'CrossEntropyLoss', 'SwiGlu',
])
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
@pytest.mark.parametrize('strided', [False, True])
def test_consolidated_controls_preserve_each_operator_and_inputs(
        operator, dtype_name, strided, positive_control_submission):
    import torch

    inputs, attrs, expected = _positive_control_cases(getattr(torch, dtype_name), strided)[operator]
    original = [value.clone() if isinstance(value, torch.Tensor) else list(value) for value in inputs]
    actual = positive_control_submission.loaded[operator].func(*inputs, **attrs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    for value, unchanged in zip(inputs, original):
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, unchanged, rtol=0, atol=0)
        else:
            assert value == unchanged


def test_consolidated_control_zip_loads_without_the_repository(tmp_path):
    import zipfile
    from triton_eval.package import ingest_zip

    package = fixture_path(ROOT / 'examples/adversarial-submissions/ADV-C1-fullhouse')
    archive_path = tmp_path / 'positive-controls.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(package.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(package).as_posix())
    extracted = ingest_zip(archive_path, tmp_path / 'ingested')
    program = '''
import importlib.util
from pathlib import Path
import sys
import torch
import yaml

root = Path(sys.argv[1])
manifest = yaml.safe_load((root / 'manifest.yaml').read_text())
for entry in manifest['operators']:
    spec = importlib.util.spec_from_file_location('standalone_' + entry['entry'], root / entry['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, entry['entry']))
assert not any(name == 'triton_eval' or name.startswith('triton_eval.') for name in sys.modules)
torch.testing.assert_close(module.exp(torch.tensor([0.0])), torch.tensor([1.0]))
print('14 standalone entrypoints loaded and Exp executed')
'''
    completed = subprocess.run([sys.executable, '-I', '-c', program, str(extracted.dest)],
                               cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '14 standalone entrypoints loaded and Exp executed' in completed.stdout


def test_argmax_scalar_policy_remains_distinct_from_nonreducing_controls(positive_control_submission):
    import torch

    actual = positive_control_submission.loaded['ArgMax'].func(torch.tensor([2, 5, 5, 0]), dim=0)
    torch.testing.assert_close(actual, torch.tensor(1), rtol=0, atol=0)
    assert actual.ndim == 0


def test_control_gather_preserves_its_existing_partial_index_limitation(positive_control_submission):
    import torch

    main = torch.tensor([[-2.0, -1.0, 1.0, 2.0], [3.0, 0.0, 3.0, -2.0]])
    indices = torch.tensor([[0, 2], [1, 3]])
    actual = positive_control_submission.loaded['Gather'].func(main, indices, dim=-1)
    assert actual.shape == main.shape
    assert actual.shape != torch.gather(main, -1, indices).shape
    torch.testing.assert_close(actual, torch.tensor([[-2.0, 1.0, -1.0, 2.0], [0.0, 0.0, 0.0, 0.0]]),
                               rtol=0, atol=0)


@pytest.mark.parametrize('version', [
    'v0.8-official-candidates', 'v0.8.1-official-candidates', 'v0.9-official-candidates',
    'v0.10-coverage-candidates', 'v0.11-implementation-candidates',
    'v0.11.1-contract-candidates', 'v0.12-compat-candidates', 'v0.13-evidence-candidates',
    'v0.14-backend-evidence-candidates', 'v0.15-backend-capture-fix-candidates',
    'v0.16-backend-pattern-certification-candidates', 'v0.17-pattern-completion-candidates',
    'v0.18-scan-certification-candidates',
])
def test_shared_references_replay_every_archived_source_byte(tmp_path, version):
    corpus = ROOT / 'corpus' / version
    records = reference_layout(corpus)
    assert len(records) == (136 if version.startswith(('v0.8-', 'v0.8.1-', 'v0.9-')) else 148)
    metadata = json.loads((corpus / 'evaluation-assets-migration.json').read_text())
    predecessor = metadata['previous_revisions'][-1]['assets_hash']
    archive = json.loads((corpus / 'evaluation-assets-history' / (predecessor + '.json')).read_text())
    copy_reference_tree(corpus, tmp_path)
    for relative, record in records.items():
        original_digest = archive['files']['@corpus/' + relative]
        assert record['sha256'] == original_digest
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == original_digest
        assert (tmp_path / relative).read_bytes() == resolve_reference_path(corpus, relative).read_bytes()
        assert resolve_reference_path(corpus, relative).name.startswith('ref_cpu' if 'ref_cpu' in relative else 'ref')


@pytest.mark.parametrize('corruption', ['digest', 'path', 'local', 'symbolic'])
def test_reference_mapping_rejects_corruption_before_oracle_loading(tmp_path, corruption):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    payload = json.loads((CORPUS / 'reference-layout.json').read_text())
    relative = next(iter(payload['files']))
    if corruption == 'digest':
        payload['files'][relative]['sha256'] = '0' * 64
    elif corruption == 'path':
        payload['files'][relative]['source'] = '../outside/ref.py'
    else:
        local = corpus / relative
        local.parent.mkdir(parents=True)
        if corruption == 'local':
            local.write_text('def reference(*inputs):\n    return inputs[0]\n')
        else:
            local.symlink_to(resolve_reference_path(CORPUS, relative))
    (corpus / 'reference-layout.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        resolve_reference_path(corpus, relative)


@pytest.mark.parametrize('version', [
    'v0.3-multi-input-frozen', 'v0.4-cov20-frozen', 'v0.5-cov20-frozen',
])
@pytest.mark.parametrize('base, natural', [
    (None, True), (-1, True), (-1.0, True),
    (math.nextafter(-1.0, -math.inf), False),
    (math.nextafter(-1.0, math.inf), False),
    (0.0, False), (-0.0, False), (0.5, False), (1.0, False), (2.0, False),
    (math.nan, False), (math.inf, False), (-math.inf, False),
])
def test_exp_attribute_sentinel_remains_exact(version, base, natural):
    import torch

    path = resolve_reference_path(ROOT / 'corpus' / version, 'operators/exp/ref.py')
    specification = importlib.util.spec_from_file_location('sentinel_reference', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float16)
    before = values.clone()
    transformed = values.float() * 0.5 + 0.25
    expected = torch.exp(transformed) if natural else torch.pow(float(base), transformed)
    actual = module.reference(values, base=base, scale=0.5, shift=0.25)
    torch.testing.assert_close(actual, expected.half(), rtol=0.0, atol=0.0, equal_nan=True)
    assert torch.equal(values, before)


@pytest.mark.parametrize('version', [
    'v0.3-multi-input-frozen', 'v0.4-cov20-frozen', 'v0.5-cov20-frozen',
])
@pytest.mark.parametrize('dimension', [0, -1])
@pytest.mark.parametrize('reduction', [None, 'add'])
@pytest.mark.parametrize('dtype_name', ['int64', 'float32'])
def test_primary_scatter_sharing_preserves_versioned_results(version, dimension, reduction, dtype_name):
    import torch

    source = resolve_reference_path(ROOT / 'corpus' / version, 'operators/scatter/ref.py')
    specification = importlib.util.spec_from_file_location('scatter_primary', source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    data = torch.arange(12, dtype=getattr(torch, dtype_name)).reshape(3, 4).t()
    before = data.clone()
    index_shape = list(data.shape)
    index_shape[dimension] = 2
    indices = torch.zeros(index_shape, dtype=torch.int64)
    updates = torch.arange(indices.numel(), dtype=data.dtype).reshape(index_shape)
    expected = data.clone()
    for coordinates in itertools.product(*(range(size) for size in index_shape)):
        indices[coordinates] = coordinates[dimension] if reduction is None else 0
        target = list(coordinates)
        target[dimension] = int(indices[coordinates])
        if reduction is None:
            expected[tuple(target)] = updates[coordinates]
        else:
            expected[tuple(target)] += updates[coordinates]
    actual = module.reference(data, indices, updates, dim=dimension, reduce=reduction)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(data, before)


def test_logging_console_preserves_stream_protocol_and_root_configuration(capsys):
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    emit("a", 12, sep="|", end="!")
    emit("diagnostic", file=sys.stderr, flush=True)
    captured = capsys.readouterr()
    assert captured.out == "a|12!"
    assert captured.err == "diagnostic\n"
    assert root.handlers == handlers and root.level == level


def test_yaml_log_writers_preserve_each_others_events(tmp_path):
    path = tmp_path / "generation-log.yaml"
    writers = [GenerationLog(path), GenerationLog(path)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda index: writers[index % 2].step("sample", index=index), range(12)))
    records = read_log(path)
    assert sum(record["step"] == "log_header" for record in records) == 1
    assert sorted(record["index"] for record in records if record["step"] == "sample") == list(range(12))
    assert path.read_text().startswith("# Copyright")


def test_legacy_jsonl_is_readable_and_corrupt_yaml_is_not_silently_discarded(tmp_path):
    legacy = tmp_path / "generation-log.jsonl"
    legacy.write_text('{"step": "build_end", "cases": 1385}\n')
    assert read_log(legacy) == [{"step": "build_end", "cases": 1385}]
    bad = tmp_path / "generation-log.yaml"
    bad.write_text("step: not-a-list\n")
    with pytest.raises(ValueError, match="list of event mappings"):
        read_log(bad)


def test_failed_atomic_yaml_replacement_retains_original(tmp_path, monkeypatch):
    import triton_eval.buildlog as buildlog

    path = tmp_path / "generation-log.yaml"
    write_log(path, [{"step": "original"}])
    previous = path.read_bytes()

    def fail_replace(*arguments):
        raise OSError("synthetic replacement failure")

    monkeypatch.setattr(buildlog.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replacement"):
        write_log(path, [{"step": "new"}])
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".generation-*"))


def test_migration_archives_the_old_index_without_changing_case_identity():
    active = json.loads((CORPUS / "evaluation-assets.json").read_text())
    migration = json.loads((CORPUS / "evaluation-assets-migration.json").read_text())
    origin = migration["origin_assets_hash"]
    archive = json.loads((CORPUS / "evaluation-assets-history" / (origin + ".json")).read_text())
    payload = {key: value for key, value in archive.items() if key != "assets_hash"}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == origin
    manifest = load_manifest(CORPUS / "manifest.json")
    assert active["manifest_hash"] == archive["manifest_hash"] == manifest.manifest_hash
    assert len(manifest.cases) == 1385
    assert migration["hardware_validation"] == "NOT_RUN"
    assert verify_origin_assets(CORPUS, origin) == active


def test_migrated_asset_checker_still_rejects_comment_only_changes(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    copy_reference_tree(CORPUS, corpus)
    reference = next((corpus / "operators").glob("*/ref.py"))
    reference.write_text(reference.read_text() + "\n# synthetic tamper\n")
    manifest = load_manifest(corpus / "manifest.json")
    with pytest.raises(ValueError, match="冻结指纹不一致"):
        verify_assets(corpus, manifest.manifest_hash)


def test_refreezing_current_index_cannot_forge_the_archived_origin(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    migration = json.loads((corpus / "evaluation-assets-migration.json").read_text())
    origin = migration["origin_assets_hash"]
    archive = corpus / "evaluation-assets-history" / (origin + ".json")
    data = json.loads(archive.read_text())
    data["files"]["spec/tolerance.yaml"] = "0" * 64
    archive.write_text(json.dumps(data))
    manifest = load_manifest(corpus / "manifest.json")
    (corpus / "evaluation-assets.json").write_text(json.dumps(snapshot(corpus, manifest.manifest_hash)))
    with pytest.raises(ValueError, match="archived asset identity"):
        verify_origin_assets(corpus, origin)


@pytest.mark.parametrize("change", ["index", "metadata", "duplicate", "missing"])
def test_previous_quality_revision_cannot_be_forged_or_silently_omitted(tmp_path, change):
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    migration_path = corpus / "evaluation-assets-migration.json"
    migration = json.loads(migration_path.read_text())
    revision = migration["previous_revisions"][-1]
    archive = corpus / "evaluation-assets-history" / (revision["assets_hash"] + ".json")
    if change == "index":
        data = json.loads(archive.read_text())
        data["files"]["spec/tolerance.yaml"] = "0" * 64
        archive.write_text(json.dumps(data))
    elif change == "missing":
        archive.unlink()
    else:
        if change == "metadata":
            revision["migration_metadata"]["sha256"] = "0" * 64
        else:
            migration["previous_revisions"].append(dict(revision))
        migration_path.write_text(json.dumps(migration))
    manifest = load_manifest(corpus / "manifest.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        snapshot(corpus, manifest.manifest_hash)


def test_new_quality_revision_preserves_immediate_predecessor_and_original_case_identity():
    migration = json.loads((CORPUS / "evaluation-assets-migration.json").read_text())
    assert migration["revision"] == "submission-constexpr-20261110-payload-score"
    assert migration["hardware_validation"] == "NOT_RUN"
    predecessors = [json.loads(decode_metadata(revision["migration_metadata"], CORPUS))["revision"]
                    for revision in migration["previous_revisions"]]
    assert predecessors == ["codecheck-compat-20261102", "codecheck-harness-20261104",
                            "codecheck-structure-20261104", "codecheck-structure-20261104-final",
                            "codecheck-followup-20261104", "codecheck-followup-20261104-final",
                            "codecheck-followup-20261104-verified", "codecheck-present-20261105",
                            "codecheck-complex-20261105", "codecheck-context-20261105",
                            "codecheck-structural-20261106", "codecheck-final-reviewed",
                            "codecheck-20261107-reviewed-repairs", "codecheck-20261107-structural-repairs",
                            "codecheck-20261107-harness-complete-repairs",
                            "codecheck-20261108-explicit-repairs",
                            "codecheck-20261108-instance-and-dependency-repairs",
                            "codecheck-20261109-contract-and-control-flow",
                            "codecheck-20261109-173323-style-only",
                            "codecheck-20261110-preflight-parameter-groups",
                            "submission-system-20261110-dependencies-constexpr-output"]
    for revision in migration["previous_revisions"]:
        archive = CORPUS / "evaluation-assets-history" / (revision["assets_hash"] + ".json")
        historical = json.loads(archive.read_text())
        metadata = historical["files"]["@corpus/evaluation-assets-migration.json"]
        assert hashlib.sha256(decode_metadata(revision["migration_metadata"], CORPUS)).hexdigest() == metadata
        assert historical["manifest_hash"] == load_manifest(CORPUS / "manifest.json").manifest_hash


@pytest.mark.parametrize('kind,groups,count,expected', [
    ('pool', 95, 5492, 'c809f3ec6be974eceb4d70f9ce4d82d5dab0586241b2f0c9e442aedb04e7e85c'),
    ('official', 68, 4912, '695240d15ddf8ecaa06ef7beccae9c1a25f304fef0ab973dcecba90b8cb0638e'),
])
def test_generation_pool_preserves_pre_refactor_values_and_iteration_order(kind, groups, count, expected):
    from triton_eval.casegen import derive_pool
    from triton_eval.official_catalog import CANDIDATES
    from triton_eval.official_generation import cases_for
    from triton_eval.operator_registry import ALL_OPERATORS

    generated = ([derive_pool(operator) for operator in ALL_OPERATORS] if kind == 'pool'
                 else [cases_for(candidate) for candidate in CANDIDATES])
    payload = [[case.model_dump(mode='json') for case in group] for group in generated]
    assert len(payload) == groups
    assert sum(map(len, payload)) == count
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected


@pytest.mark.parametrize('available,failed', [(False, False), (True, False), (True, True)])
def test_shared_builder_reference_gate_preserves_skip_and_failure_audit(monkeypatch, tmp_path, available, failed):

    from triton_eval import refcheck
    from triton_eval.buildlog import run_reference_gate

    cases = [SimpleNamespace(code='alpha'), SimpleNamespace(code='beta')]
    excluded, calls = [], []

    def check(items, device, *, corpus_dir):
        calls.append((items, device, corpus_dir))
        return {'alpha': 'bad-reference'} if failed else {}

    monkeypatch.setattr(refcheck, 'torch_available', lambda: available)
    monkeypatch.setattr(refcheck, 'check_all', check)
    retained, skipped = run_reference_gate(cases, 'cpu', tmp_path, excluded)
    assert skipped is (not available)
    assert calls == ([(cases, 'cpu', tmp_path)] if available else [])
    assert retained == (cases[1:] if available and failed else cases)
    assert excluded == ([{'code': 'alpha', 'reason': '参考实现自检：bad-reference'}]
                        if available and failed else [])


@pytest.mark.parametrize('arguments,quota,device,verify', [
    ([], 20, 'cpu', True),
    (['--per-category', '40', '--device', 'npu:0'], 40, 'npu:0', True),
    (['--no-verify-refs'], 20, 'cpu', False),
])
def test_shared_builder_arguments_preserve_historical_defaults(monkeypatch, arguments, quota, device, verify):
    from triton_eval.buildlog import builder_arguments

    monkeypatch.setattr(sys, 'argv', ['builder', *arguments])
    parsed = builder_arguments()
    assert (parsed.per_category, parsed.device, parsed.verify_refs) == (quota, device, verify)


@pytest.mark.parametrize('entry', [
    'ascend-suite-verify.sh', 'backend-dump-diagnose.sh', 'coverage-patterns-verify.sh',
])
def test_trusted_shell_scripts_are_direct_fixed_interpreter_executables(entry):
    script = ROOT / 'scripts' / entry
    assert not script.is_symlink()
    assert script.read_bytes().splitlines()[0] == b'#!/bin/bash'
    assert script.stat().st_mode & 0o777 == 0o755


def current_bindings(observation):
    """Rebind synthetic fixtures to actual current reviewed parameter names."""
    result = copy.deepcopy(observation)
    registry = json.loads((ROOT / "assets/reviewed-kernels.json").read_text())["kernels"]
    for launch in result["launches"]:
        _rebind_launch(launch, registry)
    return result


def _rebind_launch(launch, registry):
    record = registry[launch["name"]]
    source = fixture_path(ROOT / "scripts/templates" / record["template"])
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == launch["name"])
    groups = {name: len(members) for name, members in parameter_groups(node).items()}
    rename, arguments, position = {}, {}, 0
    for argument in node.args.args:
        size = groups.get(argument.arg, 1)
        roles = record['parameters'][position:position + size]
        position += size
        if argument.arg in groups:
            arguments[argument.arg] = dict(kind='tuple', constexpr=True,
                                           items=[launch['arguments'][role] for role in roles])
        else:
            rename[roles[0]] = argument.arg
            if roles[0] in launch['arguments']:
                arguments[argument.arg] = launch['arguments'][roles[0]]
    launch["source_text"] = ast.unparse(node)
    launch["runtime_arg_names"] = [rename[name] for name in launch["runtime_arg_names"]]
    launch["arguments"] = arguments


@pytest.mark.parametrize("factory,atom", [
    (max_observation, "FA-PAT-MULTIPASS-REDUCE"),
    (scan_observation, "FA-PAT-SCAN-BLOCKED"),
    (split_observation, "FA-PAT-SPLIT-K"),
    (bias_observation, "FA-PAT-VEC-BROADCAST"),
])
def test_current_pattern_arguments_retain_strict_structural_checks(factory, atom):
    observed = current_bindings(factory())
    raw = copy.deepcopy(observed["launches"])
    proof = pattern_witnesses(observed)[atom]
    assert proof["verified"], proof["reason"]
    assert observed["launches"] == raw
    launch = observed["launches"][0]
    launch["arguments"]["not_a_parameter"] = {"kind": "scalar", "constexpr": True, "value": 1}
    assert not pattern_witnesses(observed)[atom]["verified"]


def test_current_implementation_arguments_do_not_relax_tensor_binding():
    observed = current_bindings(neg())
    assert verified(observed, "FA-PAT-MASK-TAIL")
    launch = observed["launches"][0]
    incoming, outgoing = launch["runtime_arg_names"]
    launch["arguments"][incoming], launch["arguments"][outgoing] = (
        launch["arguments"][outgoing], launch["arguments"][incoming])
    assert not verified(observed, "FA-PAT-MASK-TAIL")


def test_registry_identity_is_checked_even_after_an_earlier_read(tmp_path, monkeypatch):
    import triton_eval.reviewed_kernels as reviewed

    launch = neg()["launches"][0]
    assert canonical_launch(launch)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "reviewed-kernels.json").write_text('{"kernels": {}}')
    monkeypatch.setattr(reviewed, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="registry identity differs"):
        canonical_launch(launch)


def test_console_does_not_hide_output_io_failures():
    class BrokenStream:
        @staticmethod
        def write(message):
            raise OSError("synthetic broken output")

        def flush(self):
            pass

    with pytest.raises(OSError, match="synthetic broken output"):
        emit("result", file=BrokenStream())


def test_release_history_supports_a_nested_monorepo_project(tmp_path):
    from triton_eval.release import locate_freeze_commit, read_axes_at_commit, read_toolchain_axes

    repository = tmp_path / "monorepo"
    project = repository / "benchmark/triton_bench"
    project.mkdir(parents=True)
    corpus_name = "v0.3-multi-input-frozen"
    source = ROOT / "corpus" / corpus_name / "manifest.json"
    destination = project / "corpus" / corpus_name / "manifest.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    shutil.copytree(ROOT / "spec", project / "spec")
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "user.name=Regression Fixture",
               "-c", "user.email=fixture@example.invalid", "-C", str(repository)]
    for arguments in (["init", "-q"], ["add", "."], ["commit", "-qm", "Fixture: nested platform"]):
        subprocess.run(command + arguments, check=True, capture_output=True, timeout=60)
    commit = locate_freeze_commit(project, corpus_name)
    assert read_axes_at_commit(project, commit) == read_toolchain_axes(project)


def test_semantically_similar_but_unreviewed_source_does_not_earn_credit():
    launch = neg()["launches"][0]
    tree = ast.parse(launch["source_text"])
    parameter = tree.body[0].args.args[0].arg
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == parameter:
            node.id = "arbitrary_unreviewed_parameter"
        if isinstance(node, ast.arg) and node.arg == parameter:
            node.arg = "arbitrary_unreviewed_parameter"
    launch["source_text"] = ast.unparse(tree)
    with pytest.raises(ValueError, match="unreviewed"):
        canonical_launch(launch)


def test_refactored_anticheat_keeps_pass_order_aliases_and_kernel_exemption():
    from triton_eval.anticheat import scan_source

    source = (
        "import torch as framework\nfrom triton import jit\n\n"
        "@jit\ndef kernel(x):\n    x.data_ptr()\n    framework.exp(x)\n\n"
        "cache = id(value)\nframework.npu.synchronize = callback\n"
        "framework.exp(value)\nvalue.to('cpu')\n"
    )
    findings = scan_source(source, category="Elementwise", entry="new_operator")
    assert [(item.rule_id, item.lineno) for item in findings] == [
        ("SUB-BEH-003", 9), ("SUB-BEH-006", 10), ("SUB-BEH-001", 11), ("SUB-BEH-002", 12),
    ]


@pytest.mark.parametrize("tail,accepted", [
    ("", True),
    ("\n1 warning generated.", True),
    ("\nwarning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]", True),
    ("\nnot a compiler warning", False),
    ("\nfunc.func @other() {\nreturn\n}", False),
])
def test_refactored_snapshot_keeps_exact_function_boundary_and_warning_policy(tail, accepted):
    from triton_eval.compilation_evidence import function_snapshot

    body = "func.func @kernel() {\n  return\n}"
    assert function_snapshot(body + tail, "kernel") == (body if accepted else None)
    assert function_snapshot(body + tail, "different_kernel") is None


def test_refactored_capture_preserves_launch_arguments_options_and_artifact_bytes(tmp_path, monkeypatch):
    import gzip

    from triton_eval.compilation_evidence import CompilationCapture

    monkeypatch.delenv("TRITON_EVAL_CAPTURE_BACKEND", raising=False)
    capture = CompilationCapture(None, tmp_path / "case", {"manifest_hash": "synthetic-local-only"})
    capture.directory.mkdir()
    function = SimpleNamespace(arg_names=["size", "tile"], constexprs=[1], src="def kernel(): pass", name="kernel")
    kernel = SimpleNamespace(asm={"ttir": "module {}", "npubin": b"binary", "unsupported": object()},
                             metadata={"num_warps": 4, "cluster_dims": (1, 2)}, hash="specialization")
    capture.record(function, kernel, (7, 8), {"tile": 16, "grid": (2, 3)}, b"compiler message")
    launch = capture.payload["launches"][0]
    assert launch["arguments"] == {
        "size": {"kind": "scalar", "value": 7, "constexpr": False},
        "tile": {"kind": "scalar", "value": 16, "constexpr": True},
    }
    assert launch["runtime_arg_names"] == ["size"]
    assert launch["compiler_options"] == {"num_warps": 4, "cluster_dims": [1, 2]}
    assert launch["grid"] == [2, 3] and launch["compiler_specialization_hash"] == "specialization"
    artifacts = {entry["stage"]: gzip.decompress((capture.directory / entry["path"]).read_bytes())
                 for entry in launch["artifacts"]}
    assert artifacts == {"ttir": b"module {}", "npubin": b"binary", "compiler_stderr": b"compiler message"}
    assert (capture.directory / launch["source"]["path"]).read_bytes() == function.src.encode()
    capture.record(function, kernel, (), {"warmup": True})
    assert len(capture.payload["launches"]) == 1


@pytest.fixture
def dispatch_recorder(tmp_path):
    from importlib import util

    import torch

    from triton_eval.official_catalog import CANDIDATES

    calls = []

    class Kernel:
        def __init__(self, function):
            self.name = function.__name__

        def __getitem__(self, grid):
            def record(*arguments, **options):
                calls.append((self.name, grid, arguments, options))
            return record

    triton = SimpleNamespace(jit=Kernel, cdiv=lambda count, tile: (count + tile - 1) // tile,
                             next_power_of_2=lambda count: 1 << (count - 1).bit_length())
    source = fixture_path(ROOT / 'scripts/templates/official_kernels.py')
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"torch": torch, "triton": triton, "tl": SimpleNamespace(constexpr=object()),
                     "FAMILIES": {candidate.name: candidate.family for candidate in CANDIDATES}}
    selected = tmp_path / 'host_dispatch.py'
    selected.write_text(ast.unparse(tree))
    specification = util.spec_from_file_location('_codecheck_host_dispatch', selected)
    module = util.module_from_spec(specification)
    module.__dict__.update(namespace)
    specification.loader.exec_module(module)
    return module.dispatch, calls


def _dispatch_candidates():
    from triton_eval.official_catalog import CANDIDATES

    return CANDIDATES


@pytest.mark.parametrize('candidate', _dispatch_candidates(), ids=lambda candidate: candidate.name)
def test_refactored_dispatch_preserves_allocation_and_launch_contract(candidate, dispatch_recorder):
    import torch

    from triton_eval.official_reference import make_inputs, reference

    dispatch, calls = dispatch_recorder
    attrs = {'dim': -1, 'k': 3}
    inputs = make_inputs(candidate.name, seed=19, dtype=candidate.dtypes[0], shape=[2, 8],
                         device='cpu', attrs=attrs)
    result = dispatch(candidate.name, inputs, attrs)
    expected = reference(candidate.name, *inputs, **attrs)
    assert result.shape == expected.shape and result.dtype == expected.dtype
    assert result.device == inputs[0].device
    assert len(calls) == 1
    kernel, grid, arguments, options = calls[0]
    kernels = {'reduce': '_rows', 'arg': '_rows', 'scan': '_rows', 'softmax': '_rows', 'layer': '_rows',
               'rms': '_rows', 'transpose': '_layout', 'concat': '_layout', 'flip': '_layout',
               'swiglu': '_layout', 'gather': '_gather', 'sort': '_sort', 'topk': '_sort', 'matmul': '_matmul'}
    assert kernel == kernels.get(candidate.family, '_elementwise')
    assert all(dimension > 0 for dimension in grid)
    assert any(argument is result for argument in arguments)
    if kernel == '_elementwise':
        assert len(arguments) == 5
        assert arguments[4][:2] == (inputs[0].numel(), candidate.name)
        assert arguments[4][-2:] == (256, 1)
        assert options == {'enable_fp_fusion': False}
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize('stride', [2, 3])
def test_refactored_dispatch_preserves_strided_input_boundary(stride, dispatch_recorder):
    import torch

    dispatch, calls = dispatch_recorder
    inputs = torch.arange(48, dtype=torch.float32).reshape(2, 24)[:, ::stride]
    if stride == 3:
        with pytest.raises(ValueError, match='uniform flat stride=2'):
            dispatch('Neg', [inputs], {})
        assert calls == []
        return
    output = dispatch('Neg', [inputs], {})
    assert output.shape == inputs.shape and calls[0][2][-1][-1] == 2


def test_refactored_dispatch_rejects_noncontiguous_secondary_input(dispatch_recorder):
    import torch

    dispatch, calls = dispatch_recorder
    with pytest.raises(ValueError, match='only the elementwise main input'):
        dispatch('Add', [torch.zeros(2, 4), torch.zeros(2, 8)[:, ::2]], {})
    assert calls == []


@pytest.mark.parametrize('factory', [bias_observation, neg])
def test_grouped_canonical_bindings_keep_historical_semantics_and_raw_records(factory):
    historical = factory()
    current = current_bindings(historical)
    raw = copy.deepcopy(current)
    for old, new in zip(historical['launches'], current['launches']):
        canonical = canonical_launch(new)
        assert canonical['arguments'] == old['arguments']
        assert canonical['runtime_arg_names'] == old['runtime_arg_names']
    assert current == raw


def _corrupt_group_binding(launch, mutation):
    record = launch['arguments']['configuration']
    if mutation == 'missing':
        del launch['arguments']['configuration']
        return
    if mutation == 'short':
        record['items'].pop()
        return
    if mutation == 'extra':
        record['items'].append(copy.deepcopy(record['items'][0]))
        return
    if mutation == 'runtime':
        launch['runtime_arg_names'].append('configuration')
        return
    if mutation == 'opaque':
        record['kind'] = 'scalar'
        return
    if mutation == 'nonconstant':
        record['items'][0]['constexpr'] = False
        return
    if mutation == 'notrecord':
        launch['arguments']['configuration'] = None
        return
    record['items'][0]['value'] = None


@pytest.mark.parametrize('mutation', ['missing', 'short', 'extra', 'runtime', 'opaque',
                                    'nonconstant', 'badvalue', 'notrecord'])
def test_grouped_pattern_bindings_reject_incomplete_or_forged_capture(mutation):
    observed = current_bindings(bias_observation())
    launch = next(item for item in observed['launches'] if item['name'] == '_pattern_bias_tile')
    _corrupt_group_binding(launch, mutation)
    with pytest.raises(ValueError):
        canonical_launch(launch)
    proof = pattern_witnesses(observed)['FA-PAT-VEC-BROADCAST']
    assert not proof['verified']


def test_grouped_capture_preserves_tuple_order_values_and_constexpr_classification(tmp_path):

    from triton_eval.compilation_evidence import CompilationCapture

    capture = CompilationCapture(None, tmp_path, {})
    function = SimpleNamespace(arg_names=['size', 'configuration', 'geometry'], constexprs=[1])
    arguments, runtime = capture.encode_arguments(function, (7, (8, 'Neg', 1.5, True), (3, 5)), {})
    assert runtime == ['size', 'geometry']
    assert arguments['configuration'] == dict(kind='tuple', constexpr=True, items=[
        dict(kind='scalar', value=value, constexpr=True) for value in (8, 'Neg', 1.5, True)])
    assert arguments['geometry'] == dict(kind='tuple', constexpr=False, items=[
        dict(kind='scalar', value=value, constexpr=False) for value in (3, 5)])
    arguments, runtime = capture.encode_arguments(function, (7, (object(),), (3,)), {})
    assert 'configuration' not in arguments


@pytest.mark.parametrize('number,rule', [('005', 'SUB-BEH-006'), ('007', 'SUB-BEH-009')])
def test_quality_repairs_keep_intentional_invalid_submission_rule(number, rule):
    from triton_eval.anticheat import scan_source

    source = fixture_path(ROOT / f'examples/invalid-submissions/SUB-STR-{number}/operators/exp/kernel.py')
    findings = scan_source(source.read_text(), category='Elementwise', entry='exp')
    assert [item.rule_id for item in findings] == [rule]


def interpret_grouped_probe():
    import torch
    from test_independent_submissions import Pointer, load, store

    class ProbePointer(Pointer):
        def __add__(self, offset):
            return ProbePointer(self.tensor, self.offset + offset)

        @property
        def dtype(self):
            return SimpleNamespace(element_ty=self.tensor.dtype)

    programs, unrolled = [0, 0], []

    def unroll(groups):
        unrolled.append(groups)
        return range(groups)

    language = SimpleNamespace(
        program_id=lambda axis: programs[axis], arange=torch.arange, float32=torch.float32,
        zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype), dot=torch.matmul, load=load, store=store,
    )
    source = ROOT / 'corpus/v0.5-cov20-frozen/probes/static_unroll.py'
    names = {'_store_group', '_dynamic_kernel', '_static_kernel'}
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {function.name for function in functions} == names
    for function in functions:
        function.decorator_list.clear()
        for argument in function.args.args:
            argument.annotation = None
    namespace = dict(tl=language, _UNROLL=unroll)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace, ProbePointer, programs, unrolled


@pytest.mark.parametrize('kernel_name', ['_dynamic_kernel', '_static_kernel'])
@pytest.mark.parametrize('inner_count', [5, 8, 9])
@pytest.mark.parametrize('groups', [2, 8])
def test_grouped_probe_shares_math_but_preserves_distinct_group_loops(kernel_name, inner_count, groups):
    import torch

    namespace, pointer, programs, unrolled = interpret_grouped_probe()
    generator = torch.Generator().manual_seed(17)
    left = torch.randn(groups, 8, inner_count, generator=generator)
    right = torch.randn(groups, inner_count, 8, generator=generator)
    bias = torch.randn(groups, 8, generator=generator)
    originals = [tensor.clone() for tensor in (left, right, bias)]
    output = torch.empty(groups, 8, 8)
    for row_program, column_program in itertools.product(range(2), repeat=2):
        programs[:] = [row_program, column_program]
        namespace[kernel_name](pointer(left), pointer(right), pointer(bias), pointer(output),
                               (8, inner_count, 8), (4, 4, 4, groups))
    expected = torch.bmm(left, right) + bias.unsqueeze(1)
    torch.testing.assert_close(output, expected, rtol=5e-3, atol=5e-3)
    assert unrolled == ([groups] * 4 if kernel_name == '_static_kernel' else [])
    assert all(torch.equal(before, after) for before, after in zip(originals, (left, right, bias)))


def test_static_unroll_probe_uses_the_current_grouped_device_signature():

    import torch

    source = ROOT / 'corpus/v0.5-cov20-frozen/probes/static_unroll.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_run')
    namespace = dict(torch=torch, triton=SimpleNamespace(cdiv=lambda count, tile: (count + tile - 1) // tile))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    calls = []

    class Kernel:
        @staticmethod
        def __getitem__(grid):
            def record(*arguments, **options):
                calls.append((grid, arguments, options))
            return record

    inputs = torch.zeros(8, 3, 5), torch.zeros(8, 5, 7), torch.zeros(8, 7)
    result = namespace['_run'](Kernel(), *inputs)
    grid, arguments, options = calls[0]
    assert result.shape == (8, 3, 7)
    assert grid == (1, 1) and options == {}
    assert len(arguments) == 6 and arguments[4:] == ((3, 5, 7), (32, 32, 32, 8))


def test_refactored_dispatch_helpers_match_all_generated_versions():
    names = {'dispatch', '_dispatch_output_metadata', '_validate_dispatch_inputs'}

    def helpers(path):
        tree = ast.parse(path.read_text())
        return {node.name: ast.dump(node) for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names}

    expected = helpers(fixture_path(ROOT / 'scripts/templates/official_kernels.py'))
    assert expected.keys() == names
    paths = sorted((fixture_path(ROOT / 'examples')).glob('coverage*-ascend-submission/operators/kernels.py'))
    paths.append(fixture_path(ROOT / 'examples/official-ascend-submission/operators/kernels.py'))
    assert len(paths) == 11
    for path in paths:
        assert helpers(path) == expected


def test_refactored_backend_stage_filter_does_not_hide_later_shared_dot():
    from triton_eval.implementation_evidence import _has_shared_dot

    launch = {"runtime_arg_names": [], "stage_text": {
        'ttir': 'ignored', 'compiler_stderr': 'ignored', 'compiler_stderr_truncated': 'ignored',
        'ttgir': '%shared = ttg.local_alloc %input : !ttg.memdesc<16xf32, #shared>\n'
                 '%value = ttg.local_load %shared : !ttg.memdesc<16xf32, #shared>\n'
                 '%dot = tt.dot %value, %value, %zero : f32\ntt.store %out, %dot : f32',
    }}
    assert _has_shared_dot(launch) is True


@pytest.mark.parametrize('shape,expected', [
    ([1024], []),
    ([2, 4], [{'perm': [1, 0]}]),
    ([2, 4, 8], [{'perm': [2, 1, 0]}, {'perm': [1, 2, 0]}]),
])
def test_refactored_transpose_attribute_boundaries(shape, expected):
    from triton_eval.casegen import _attr_variants
    from triton_eval.operator_registry import by_name

    assert _attr_variants(by_name('Transpose'), shape) == expected


def _cpu_executor(path, directory, monkeypatch):
    from importlib import util

    import torch

    tree = ast.parse(path.read_text())
    retained = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(item.name.startswith(('triton', 'torch_npu')) for item in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == 'bench.measure':
            continue
        retained.append(node)
    tree.body = retained
    selected = directory / 'cpu_executor.py'
    selected.write_text(ast.unparse(tree))
    specification = util.spec_from_file_location('triton_eval.harness._codecheck_cpu_executor', selected)
    module = util.module_from_spec(specification)
    module.__dict__.update(triton=SimpleNamespace(jit=lambda function: function),
                           tl=SimpleNamespace(constexpr=object()))
    specification.loader.exec_module(module)
    module.torch = SimpleNamespace(**dict(vars(torch), npu=SimpleNamespace(synchronize=lambda: None)))

    def cpu_device():
        return 'cpu'

    monkeypatch.setattr(module, '_device', cpu_device)
    clock = iter(range(1000))
    module.time = SimpleNamespace(time=lambda: next(clock) / 100000.0)
    module.measure = _cpu_measure
    module.measure_baseline = _cpu_baseline
    return module


def _cpu_measure(function, repeats, warmup_ms, rep_ms):
    function()
    return [0.25] * repeats


def _cpu_baseline(function, repeats, warmup_ms, rep_ms):
    function()
    return 0.5


def _synthetic_failure(*arguments, **keywords):
    raise RuntimeError('synthetic execution failure')


def _mutating_reference(tensor):
    tensor.add_(1)
    return tensor.abs()


def _reference_fault_options(mode, torch):
    if mode == 'missing_reference':
        return {'reference': None}
    if mode == 'reference_fail':
        return {'reference': _synthetic_failure}
    if mode == 'reference_mutates':
        return {'reference': _mutating_reference}
    if mode == 'reference_arity':
        return {'reference': lambda tensor: (tensor.abs(), tensor.abs())}
    if mode == 'reference_dtype':
        return {'reference': lambda tensor: tensor.abs().to(torch.float64)}
    if mode == 'cross_wrong':
        return {'cross_reference': lambda tensor: tensor.abs() + 1}
    if mode == 'cross_fail':
        return {'cross_reference': _synthetic_failure}
    if mode == 'cross_mutates':
        return {'cross_reference': _mutating_reference}
    return {}


def _implementation_fault_options(mode, torch):
    if mode == 'missing_impl':
        return {'impl': None}
    if mode == 'runtime_fail':
        return {'impl': _synthetic_failure}
    if mode == 'mutation':
        return {'impl': _mutating_reference}
    if mode == 'wrong_values':
        return {'impl': lambda tensor: tensor.abs() + 1}
    if mode == 'shape':
        return {'impl': lambda tensor: tensor[:, :2].abs()}
    if mode == 'dtype':
        return {'impl': lambda tensor: tensor.abs().to(torch.float64)}
    if mode == 'arity':
        return {'impl': lambda tensor: (tensor.abs(), tensor.abs())}
    if mode == 'tampering':
        return {'tamper_check': lambda: ['synthetic_api']}
    if mode == 'input_fail':
        return {'make_inputs': _synthetic_failure}
    return {}


def _cpu_executor_sample(module, mode):
    import dataclasses

    import torch

    from triton_eval.judge import decide
    from triton_eval.models import CaseSource, CaseSpec
    from triton_eval.specload import load_tolerance

    case = CaseSpec(code='codecheck-cpu-control', operator='Abs', source=CaseSource.VENDOR_SELFTEST,
                    dtype='float32', shape=[2, 4], seed=19)
    options = {'impl': torch.abs, 'reference': torch.abs, 'cross_reference': torch.abs,
               'make_inputs': lambda **keywords: [torch.arange(-4, 4, dtype=torch.float32).reshape(2, 4)],
               'category': 'Elementwise'}
    options.update(_reference_fault_options(mode, torch))
    options.update(_implementation_fault_options(mode, torch))
    if mode == 'performance':
        options.update(measure_perf=True, repeats=2, warmup_ms=1, rep_ms=2)
    result = module.execute_case(case, **options)
    status = decide(result, load_tolerance().for_dtype('float32')).value
    return dataclasses.asdict(result), status


@pytest.mark.parametrize('mode,expected', [
    ('valid', 'PASS'), ('performance', 'PASS'), ('missing_impl', 'SUBMISSION_INVALID'),
    ('missing_reference', 'ORACLE_FAIL'), ('input_fail', 'ORACLE_FAIL'),
    ('reference_fail', 'ORACLE_FAIL'), ('reference_mutates', 'ORACLE_FAIL'),
    ('reference_arity', 'ORACLE_FAIL'), ('reference_dtype', 'ORACLE_FAIL'),
    ('cross_wrong', 'ORACLE_FAIL'), ('cross_fail', 'ORACLE_FAIL'), ('cross_mutates', 'ORACLE_FAIL'),
    ('runtime_fail', 'RUNTIME_FAIL'), ('mutation', 'SUBMISSION_INVALID'),
    ('wrong_values', 'WRONG_RESULT'), ('shape', 'SHAPE_MISMATCH'), ('dtype', 'SUBMISSION_INVALID'),
    ('arity', 'SHAPE_MISMATCH'), ('tampering', 'SUBMISSION_INVALID'),
])
def test_cpu_executor_control_flow_preserves_failure_attribution(mode, expected, tmp_path, monkeypatch):
    module = _cpu_executor(ROOT / 'src/triton_eval/harness/npu_exec.py', tmp_path, monkeypatch)
    measurement, status = _cpu_executor_sample(module, mode)
    assert status == expected
    if mode == 'performance':
        assert measurement['samples_ms'] == [0.25, 0.25]
        assert measurement['baseline_median_ms'] == 0.5


@pytest.mark.parametrize('attrs,contiguous,value_range', [
    ({}, True, None), ({'dim': -1, 'keepdim': True}, False, {'kind': 'uniform', 'low': -2}),
    ({'group_list': [2, 3]}, True, {'kind': 'normal'}),
])
def test_named_signatures_preserve_order_hashing_and_json(attrs, contiguous, value_range):
    from triton_eval.coverage import _case_signature
    from triton_eval.pattern_generation import structural_signature

    case = SimpleNamespace(operator='Neg', dtype='float32', shape=[2, 3], attrs=attrs,
                           contiguous=contiguous, value_range=value_range)
    expected = ('Neg', 'float32', (2, 3), tuple(sorted((key, str(value)) for key, value in attrs.items())),
                json.dumps(value_range or {'kind': 'normal'}, sort_keys=True, separators=(',', ':')), contiguous)
    actual = _case_signature(case)
    assert actual == expected and hash(actual) == hash(expected)
    assert json.dumps(actual) == json.dumps(expected)
    pattern = structural_signature(case)
    previous = ('Neg', 'float32', (2, 3), json.dumps(attrs, sort_keys=True, separators=(',', ':')),
                contiguous, json.dumps(value_range, sort_keys=True))
    assert pattern == previous and hash(pattern) == hash(previous)
    assert actual.operator == pattern.operator == 'Neg'


def test_reference_dependency_record_retains_legacy_unpacking():
    from triton_eval.refcheck import _load_deps

    dependencies = _load_deps()
    assert len(dependencies) == 7
    assert dependencies[0] is dependencies.torch
    assert dependencies[2] is dependencies.dtype_of
    assert dependencies[5] is dependencies.reference_for
    assert list(dependencies) == list(tuple(dependencies))


@pytest.mark.parametrize('artifacts', [[], [{'stage': 'a'}, {'stage': 'a'}]])
def test_backend_artifact_lookup_rejects_missing_and_duplicate_bindings(artifacts):
    from triton_eval.backend_evidence import _stage_artifact

    with pytest.raises(ValueError, match='ambiguous or missing'):
        _stage_artifact({'a': artifacts}, 'a')
    with pytest.raises(ValueError, match='ambiguous or missing'):
        _stage_artifact({}, 'unknown')


def test_backend_artifact_lookup_keeps_original_binding():
    from triton_eval.backend_evidence import _stage_artifact

    artifact = {'stage': 'a', 'sha256': 'identity'}
    stages = {'a': [artifact]}
    assert _stage_artifact(stages, 'a') is artifact
    assert stages == {'a': [artifact]}


def test_implementation_callbacks_bind_each_launch_even_when_deferred(monkeypatch):
    from triton_eval import implementation_evidence as evidence

    callbacks = []
    rejected = []

    def defer_negation(*arguments):
        callbacks.append(arguments[-2])
        rejected.append(arguments[-1])

    def defer_grid(*arguments):
        callbacks.append(arguments[-1])

    monkeypatch.setattr(evidence, '_prepare_implementation_launch',
                        lambda _observed, launch: (launch, None, [], [], set(), []))
    monkeypatch.setattr(evidence, '_negation_witness', defer_negation)
    monkeypatch.setattr(evidence, '_grid3_witness', defer_grid)
    launches = [dict(name='_witness_neg', compiler_specialization_hash='first'),
                dict(name='_witness_neg3', compiler_specialization_hash='second')]
    results = evidence.implementation_witnesses(dict(
        case=SimpleNamespace(operator='Neg', shape=[2, 2, 2]), launches=launches))
    atom = next(iter(evidence.ATOMS))
    rejected[0](atom, 'first rejection')
    assert results[atom]['kernel'] == '_witness_neg'
    for callback, launch in zip(callbacks, launches):
        callback(atom, 'synthetic callback binding test')
        assert results[atom]['kernel'] == launch['name']
        assert results[atom]['compiler_specialization_hash'] == launch['compiler_specialization_hash']


def test_validation_error_class_keeps_legacy_import_and_serialization():
    from dataclasses import asdict

    from triton_eval.validator import SubmissionValidationError, ValidationError_

    assert ValidationError_ is SubmissionValidationError
    error = ValidationError_('RULE', 'load', 'Exp', 'detail')
    assert asdict(error) == dict(rule_id='RULE', stage='load', operator='Exp', detail='detail', lineno=None)
    assert error.render() == SubmissionValidationError('RULE', 'load', 'Exp', 'detail').render()


def test_missing_torch_reference_cli_returns_diagnostic_instead_of_name_error():
    script = ROOT / 'scripts/verify_refs.py'
    source = (
        'import runpy, sys\n'
        'class BlockTorch:\n'
        '    def find_spec(self, fullname, path=None, target=None):\n'
        '        if fullname == "torch":\n'
        '            raise ModuleNotFoundError("deliberately blocked", name="torch")\n'
        'sys.meta_path.insert(0, BlockTorch())\n'
        f'sys.argv = [{str(script)!r}]\n'
        f'runpy.run_path({str(script)!r}, run_name="__main__")\n'
    )
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True,
                            env=environment, timeout=30)
    assert result.returncode == 2
    assert '本脚本需要 torch' in result.stderr and 'NameError' not in result.stderr


def test_shared_builder_candidates_retain_registry_order_and_exact_labels():
    from triton_eval.casegen import derive_pool, manifest_candidates, manifest_operators
    from triton_eval.coverage import label_case
    from triton_eval.models import CaseSource, CaseSpec
    from triton_eval.operator_registry import by_name

    operators = [by_name('Matmul'), by_name('Exp')]
    declarations = manifest_operators(operators)
    assert [declaration.name for declaration in declarations] == ['Matmul', 'Exp']
    assert [declaration.ref_path for declaration in declarations] == ['operators/matmul/ref.py', 'operators/exp/ref.py']
    expected = []
    for operator in operators:
        for derived in derive_pool(operator):
            case = CaseSpec(
                code=derived.code, operator=derived.operator, source=CaseSource.VENDOR_SELFTEST,
                dtype=derived.dtype, shape=derived.shape, attrs=derived.attrs, seed=derived.seed,
                launch={}, timeout_s=derived.timeout_s, feature_atoms=[],
                value_range=dict(derived.value_range.model_dump()), contiguous=derived.contiguous)
            case.feature_atoms = label_case(case, operator)
            expected.append(case.model_dump(mode='json'))
    assert [case.model_dump(mode='json') for case in manifest_candidates(operators)] == expected
    assert manifest_operators([]) == manifest_candidates([]) == []


@pytest.mark.parametrize(
    'source',
    sorted((fixture_path(ROOT / 'examples')).glob('*ascend-submission/operators/kernels.py')),
    ids=lambda path: path.parent.parent.name,
)
def test_renamed_entries_and_manifests_keep_exact_operator_signatures(source):
    import yaml

    from triton_eval.operator_registry import by_name
    from triton_eval.submission_preflight import source_callable
    from triton_eval.validator import signature_errors

    manifest = yaml.safe_load((source.parent.parent / 'manifest.yaml').read_text())
    declared = {entry['name']: entry for entry in manifest['operators']}
    functions = {node.name: node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)}
    assert not {'abs', 'pow'} & set(functions)
    for operator, entry in [('Abs', 'absolute'), ('Pow', 'power')]:
        assert declared[operator]['entry'] == entry
        assert declared[operator]['path'] == 'operators/kernels.py'
        assert signature_errors(source_callable(functions[entry]), by_name(operator)) == []


def test_official_builder_regenerates_matching_non_builtin_entry_names(tmp_path):
    from importlib import util
    import yaml

    from triton_eval.submission_preflight import submission_readiness

    specification = util.spec_from_file_location('context_official_builder',
                                                          ROOT / 'scripts/build_official_corpus.py')
    builder = util.module_from_spec(specification)
    specification.loader.exec_module(builder)
    submission = tmp_path / 'submission'
    builder.prepare_assets(tmp_path / 'corpus', submission)
    manifest = yaml.safe_load((submission / 'manifest.yaml').read_text())
    entries = {row['name']: row['entry'] for row in manifest['operators']}
    assert entries['Abs'] == 'absolute' and entries['Pow'] == 'power'
    report = submission_readiness(submission, ('Abs', 'Pow'))
    assert report['ok'] and report['source_checked'] == report['declared_operators'] == 68
    assert report['load_state'] == 'NOT_RUN'


def test_independent_abs_submission_uses_its_declared_non_builtin_entry(tmp_path):
    from triton_eval.independent_submissions import entries
    from triton_eval.submission_preflight import submission_readiness

    scenario = SimpleNamespace(operators=('Abs',), layout='default', declared_submit_id=None, code='NEW-ABS')
    for name, payload in entries(scenario).items():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    report = submission_readiness(tmp_path, ('Abs',))
    assert report['ok'] and report['source_checked'] == report['declared_operators'] == 1
    assert report['load_state'] == 'NOT_RUN'


def test_missing_log_path_schema_is_rejected_instead_of_silently_documented(monkeypatch):
    from triton_eval import submission_http

    class MissingLogPath:
        @staticmethod
        def model_json_schema(**_options):
            return {'properties': {}}

    monkeypatch.setattr(submission_http, 'CaseResultRecord', MissingLogPath)
    with pytest.raises(ValueError, match='missing its log_path'):
        submission_http.openapi()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0', 'npu:0'])
@pytest.mark.parametrize('missing', [None, 'torch', 'triton', 'torch_npu'])
def test_probe_dependency_check_preserves_import_order_and_failure_diagnostic(device, missing, monkeypatch, capsys):
    from triton_eval.envprobe import collect

    imported = []

    def load(name):
        imported.append(name)
        if name == missing:
            raise ModuleNotFoundError(f'blocked {name}')
        return SimpleNamespace()

    monkeypatch.setattr(collect.importlib, 'import_module', load)
    expected = ['torch', 'triton'] + (['torch_npu'] if device.startswith('npu') else [])
    fails = missing in expected
    assert collect.kernel_dependencies_available(device) is not fails
    assert imported == (expected[:expected.index(missing) + 1] if fails else expected)
    output = capsys.readouterr()
    assert output.out == ''
    assert output.err == (f'探针需要 torch/triton（NPU 设备 {device}）：blocked {missing}\n' if fails else '')


@pytest.mark.parametrize('script', ['dot_precision_probe.py', 'probe_validation.py'])
def test_probe_scripts_return_two_before_device_work_when_dependencies_are_missing(script, monkeypatch, capsys):
    from importlib import util
    from triton_eval.envprobe import collect

    def blocked(_name):
        raise ModuleNotFoundError('blocked torch')

    specification = util.spec_from_file_location('context_probe', ROOT / 'scripts' / script)
    probe = util.module_from_spec(specification)
    monkeypatch.setenv('TRITON_ALL_BLOCKS_PARALLEL', os.environ.get('TRITON_ALL_BLOCKS_PARALLEL', '1'))
    specification.loader.exec_module(probe)
    monkeypatch.setattr(collect.importlib, 'import_module', blocked)
    monkeypatch.setattr(sys, 'argv', [script, '--device', 'npu:0'])
    assert probe.main() == 2
    assert capsys.readouterr().err == '探针需要 torch/triton（NPU 设备 npu:0）：blocked torch\n'


def test_probe_dependency_check_does_not_hide_non_import_failures(monkeypatch):
    from triton_eval.envprobe import collect

    def broken(_name):
        raise RuntimeError('broken runtime')

    monkeypatch.setattr(collect.importlib, 'import_module', broken)
    with pytest.raises(RuntimeError, match='broken runtime'):
        collect.kernel_dependencies_available('npu:0')


@pytest.mark.parametrize('dtype_name', ['float16', 'float32', 'float64', 'bfloat16',
                                     'int8', 'int16', 'int32', 'int64', 'uint8', 'bool'])
@pytest.mark.parametrize('count', [0, 1, 17])
def test_shared_reference_sample_preserves_private_generator(dtype_name, count):
    torch = pytest.importorskip('torch')
    from triton_eval.inputs import reference_sample

    dtype = getattr(torch, dtype_name)
    before = torch.get_rng_state().clone()
    generator = torch.Generator(device='cpu').manual_seed(73)
    if dtype.is_floating_point:
        expected = torch.randn(count, generator=generator).to(dtype)
    else:
        expected = torch.randint(1, 100, (count,), generator=generator, dtype=torch.int64).to(dtype)
    actual = reference_sample(73, count, dtype)
    assert torch.equal(actual, expected)
    assert actual.dtype == dtype and actual.shape == (count,)
    assert torch.equal(torch.get_rng_state(), before)


def test_shared_segment_loader_is_version_local_and_does_not_cache(tmp_path):
    from triton_eval.refcheck import load_reference_segment

    loaded = []
    for version, value in [('old', 3), ('new', 7)]:
        operators = tmp_path / version / 'operators'
        (operators / 'primitive').mkdir(parents=True)
        (operators / 'fusion').mkdir()
        path = operators / 'primitive/ref.py'
        path.write_text(f'value = {value}\n')
        source = operators / 'fusion/ref.py'
        loaded.append(load_reference_segment(str(source), 'primitive'))
        assert loaded[-1].value == value
        assert Path(loaded[-1].__file__) == path
        assert loaded[-1].__name__ == 'te_seg_primitive'
        assert load_reference_segment(str(source), 'primitive') is not loaded[-1]
    assert loaded[0] is not loaded[1]


def test_shared_segment_loader_preserves_failures(tmp_path, monkeypatch):
    from triton_eval import refcheck

    source = tmp_path / 'fusion/ref.py'
    with pytest.raises(FileNotFoundError):
        refcheck.load_reference_segment(str(source), 'missing')
    (tmp_path / 'broken').mkdir()
    (tmp_path / 'broken/ref.py').write_text("raise RuntimeError('broken reference')\n")
    with pytest.raises(RuntimeError, match='broken reference'):
        refcheck.load_reference_segment(str(source), 'broken')
    for specification in [None, SimpleNamespace(loader=None)]:
        monkeypatch.setattr(refcheck.importlib.util, 'spec_from_file_location',
                            lambda *_args, current=specification: current)
        with pytest.raises(ValueError, match='Reference segment has no loader'):
            refcheck.load_reference_segment(str(source), 'broken')


@pytest.mark.parametrize('legacy', [False, True])
def test_shared_routing_retains_shape_contract_and_index_dtype(legacy):
    torch = pytest.importorskip('torch')
    from triton_eval.inputs import expert_routing_inputs

    shape = (5, 3, 4) if legacy else (5, 3)
    actual = expert_routing_inputs(73, shape, 'cpu', {'num_experts': 4, 'top_k': 2}, legacy_shape=legacy)
    assert actual.values.dtype == torch.float16 and actual.values.shape == (5, 3)
    assert actual.experts.dtype == actual.bins.dtype == torch.int32
    assert actual.indices.dtype == torch.int64 and actual.indices.shape == (10,)
    assert actual.expert_count == 4
    assert torch.equal(actual.experts[actual.indices], actual.bins)


@pytest.mark.parametrize('unique', [False, True])
def test_shared_token_pool_retains_historical_request_policy(unique):
    torch = pytest.importorskip('torch')
    from triton_eval.inputs import token_pool_inputs

    requests, pool, starts, ends, locations = token_pool_inputs(
        73, 'cpu', {'bs_upper': 512, 'pool_len': 8}, unique_requests=unique)
    assert (requests.unique().numel() == 512) is unique
    assert pool.shape == (512, 8) and pool.count_nonzero() == 0
    assert all(tensor.dtype == torch.int32 for tensor in [requests, pool, starts, ends, locations])
    assert bool(((starts >= 0) & (ends <= 8) & (ends > starts)).all())
    assert locations.numel() == int((ends - starts).sum())


@pytest.mark.parametrize('capacity', [0, 1, 2, 5])
def test_shared_binned_reference_keeps_empty_experts_and_capacity(capacity):
    torch = pytest.importorskip('torch')
    from triton_eval.reference_execution import binned_gather_reference

    values = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float16)
    indices = torch.tensor([2, 0, 1, 4], dtype=torch.int32)
    bins = torch.tensor([2, 2, 4], dtype=torch.int32)
    original = values.clone()
    actual = binned_gather_reference(values, indices, bins, 2, capacity)
    expected = values.new_zeros((3, capacity, 2))
    count = min(capacity, 2)
    expected[0, :count] = values[torch.tensor([1, 0])[:count]]
    expected[2, :count] = values[torch.tensor([0, 2])[:count]]
    assert torch.equal(actual, expected) and actual.dtype == values.dtype
    assert torch.equal(values, original)


def test_shared_padded_reference_preserves_gaps_and_inputs():
    torch = pytest.importorskip('torch')
    from triton_eval.reference_execution import padded_gather_reference

    values = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    arguments = [values, torch.tensor([2, 0, 1, 4], dtype=torch.int32),
                 torch.tensor([2, 2, 4], dtype=torch.int32), torch.tensor([4, 4, 8], dtype=torch.int32)]
    originals = [tensor.clone() for tensor in arguments]
    actual = padded_gather_reference(*arguments, 2)
    expected = values.new_zeros((8, 2))
    expected[[0, 1, 4, 5]] = values[[1, 0, 0, 2]]
    assert torch.equal(actual, expected)
    assert all(torch.equal(before, after) for before, after in zip(originals, arguments))


def test_shared_token_reference_preserves_sequential_overwrites():
    torch = pytest.importorskip('torch')
    from triton_eval.reference_execution import assign_token_pool_reference

    pool = torch.zeros((1, 4), dtype=torch.int32)
    actual = assign_token_pool_reference(torch.tensor([0, 0]), pool, torch.tensor([0, 1]),
                                         torch.tensor([2, 3]), torch.tensor([1, 2, 3, 4]), 2)
    assert torch.equal(actual, torch.tensor([[1, 3, 4, 0]], dtype=torch.int32))
    assert pool.count_nonzero() == 0 and actual.data_ptr() != pool.data_ptr()


def test_import_exit_fixture_preserves_base_exception_and_has_no_statements_after_exit():
    source = fixture_path(ROOT / 'examples/adversarial-submissions/ADV-E3-systemexit/operators/exp/kernel.py')
    tree = ast.parse(source.read_text())
    assert isinstance(tree.body[-1], ast.Raise)
    probe = "import runpy, sys; runpy.run_path(sys.argv[1]); print('EXIT-WAS-NOT-RAISED')"
    result = subprocess.run([sys.executable, '-c', probe, str(source)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert result.stdout == result.stderr == ''
    from triton_eval.safeexec import SubmissionEscape, guard_call

    with pytest.raises(SubmissionEscape) as failure:
        guard_call(exec, compile(tree, str(source), 'exec'), {})
    assert failure.value.origin.code == 0
    assert not isinstance(failure.value.origin, Exception)


@pytest.mark.parametrize('module_name,record_name,fields', [
    ('triton_eval.coverage_evidence', 'AuditInputs', 'manifest active root snapshot functions summary'),
    ('triton_eval.pattern_evidence', 'PairScan', 'left left_index right right_index definitions predicate'),
    ('triton_eval.pattern_release', 'PatternCases', 'parent_cases cases active plan selection added'),
    ('triton_eval.cli', 'ComparisonChanges', 'func_regress func_fix perf_regress perf_gain only_old only_new'),
])
def test_named_quality_results_preserve_tuple_protocol(module_name, record_name, fields):
    module = importlib.import_module(module_name)
    record = getattr(module, record_name)
    values = tuple(range(6))
    result = record(*values)
    assert isinstance(result, tuple)
    assert result == values
    assert list(result) == list(values)
    assert all(getattr(result, name) == value for name, value in zip(fields.split(), values))


def test_shared_slice_and_routing_plans_preserve_order_and_attributes():
    path = ROOT / 'scripts/build_corpus_v06_ascend_ops.py'
    names = {'_slice_variants', '_routing_variants', '_plan_insert_slice', '_plan_extract_slice',
             '_plan_gather_scatter', '_plan_binned_gather_scatter', '_plan_padded_gather_scatter'}
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(product=itertools.product)
    exec(compile(tree, str(path), 'exec'), namespace)
    slices = [((rows, width), 'float32', {}, 'normal', True)
              for rows, width in itertools.product([65, 256, 1024, 4096, 8192], [32, 64, 128, 256])]
    assert namespace['_plan_insert_slice']() == namespace['_plan_extract_slice']() == slices
    for name in ('gather', 'binned_gather', 'padded_gather'):
        top_k_values = [2, 4] if name == 'binned_gather' else [2, 4, 8]
        parameters = itertools.product([256, 1024, 4096], [128, 512, 2048], [8, 32], top_k_values)
        expected = [((length, hidden, experts), 'float16', {'top_k': top_k}, 'normal', True)
                    for length, hidden, experts, top_k in parameters]
        assert namespace['_plan_' + name + '_scatter']() == expected


def test_license_matcher_preserves_live_and_historical_cann_declarations():
    from xml.etree import ElementTree

    configuration = ElementTree.parse(ROOT.parents[1] / 'OAT.xml')
    matchers = configuration.findall('./oatconfig/licensematcherlist/licensematcher')
    assert len(matchers) == 1
    assert matchers[0].attrib['name'] == 'CANN-2.0'
    texts = [entry.attrib['name'] for entry in matchers[0].findall('licensetext')]
    assert len(texts) == 3
    assert any('NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS' in text for text in texts)
    assert any('NON-INFRINGEMENT, MERCHANTABILITY, FITNESS' in text for text in texts)
    assert any(text.lstrip().startswith('CANN Open Software License Agreement Version 2.0') for text in texts)
