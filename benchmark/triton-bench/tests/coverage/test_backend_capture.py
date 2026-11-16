# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Synthetic subprocesses test provenance plumbing, never supply real NPU credit."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_eval.backend_capture import BackendCapture
from triton_eval.backend_evidence import backend_snapshots
from triton_eval.compilation_evidence import CompilationCapture, compiler_passes
from triton_eval.coverage_evidence import observation
from test_evidence_gaps import ASCEND_VECTOR
from test_coverage_evidence import coverage_suite as coverage_suite
from test_implementation_evidence import neg, verified


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    executable = tmp_path / 'bishengir-compile'
    executable.write_text(f'''#!{sys.executable}
import pathlib, sys
if '--help' in sys.argv:
    print('  --mlir-print-ir-after-all  Print snapshots')
    print('  --mlir-print-ir-module-scope  Print full modules')
    sys.exit(0)
source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[sys.argv.index('-o')+1])
output.write_bytes(b'synthetic-unit-binary')
print('synthetic compiler stdout')
if '--mlir-print-ir-after-all' in sys.argv:
    print('// IR Dump After Lowering\\n'+source.read_text(), file=sys.stderr)
if source.name == 'failure.mlir':
    sys.exit(9)
''')
    executable.chmod(0o755)
    source = tmp_path / 'input.mlir'
    source.write_text(ASCEND_VECTOR)
    output = tmp_path / 'output.o'
    monkeypatch.setenv('TRITON_EVAL_CAPTURE_BACKEND', '1')
    writer = CompilationCapture(None, tmp_path / 'capture/case', {})
    writer.directory.mkdir(parents=True)
    return SimpleNamespace(executable=executable, source=source, output=output, writer=writer,
                           command=lambda: [str(executable), str(source), '-o', str(output)])


def launch_for(compiler, capture):
    return dict(
        name='_witness_neg',
        artifacts=[
            compiler.writer.artifact(
                'ttadapter',
                ASCEND_VECTOR),
            compiler.writer.artifact(
                'npubin',
                compiler.output.read_bytes()),
            *capture.artifacts],
        backend_compilations=capture.records)


def test_external_compiler_pipes_are_captured_bound_and_do_not_change_return_values(compiler):
    original = subprocess.run
    with BackendCapture(compiler.writer.artifact) as capture:
        result = subprocess.run(compiler.command(), capture_output=True, text=True, check=True)
        assert result.stdout == 'synthetic compiler stdout\n'
        assert 'IR Dump After Lowering' in result.stderr
    assert subprocess.run is original
    launch = launch_for(compiler, capture)
    snapshots, bindings = backend_snapshots(launch, compiler.writer.directory)
    assert bindings[0]['verified'] and bindings[0]['snapshot_count'] == 1
    assert 'hivm.hir.load' in next(iter(snapshots.values()))
    observed = neg(8192, lowered=snapshots)
    assert verified(observed, 'FA-PAT-VEC-LOAD')
    record = capture.records[0]
    assert record['input_sha256'] == hashlib.sha256(ASCEND_VECTOR.encode()).hexdigest()
    assert record['output_sha256'] == hashlib.sha256(compiler.output.read_bytes()).hexdigest()
    assert record['executable_sha256'] == hashlib.sha256(compiler.executable.read_bytes()).hexdigest()


def test_vendor_filters_capture_real_function_scoped_ir_without_module_or_tree_flags(compiler):
    from test_physical_backend_evidence import FIXTURES

    passes = 'hivm-inject-sync,hivm-inject-block-sync,hivm-graph-sync-solver'
    compiler.executable.write_text(f'''#!{sys.executable}
import pathlib, sys
if '--help' in sys.argv:
    print('  --bishengir-print-ir-before=<pass-arg>  Before snapshots')
    print('  --bishengir-print-ir-after=<pass-arg>  After snapshots')
    print('  --mlir-print-ir-after-all  Generic snapshots')
    print('  --mlir-print-ir-module-scope  Modules')
    print('  --mlir-print-ir-tree-dir=<string>  Directory')
    sys.exit(0)
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'synthetic-unit-binary')
if '--bishengir-print-ir-before={passes}' in sys.argv and '--bishengir-print-ir-after={passes}' in sys.argv:
    sys.stderr.write(pathlib.Path({str(FIXTURES / 'Neg.txt')!r}).read_text())
''')
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    record = capture.records[0]
    assert record['command'] == compiler.command() + [
        '--bishengir-print-ir-before=' + passes, '--bishengir-print-ir-after=' + passes]
    assert not record.get('capture_directory')
    launch = launch_for(compiler, capture)
    snapshots, binding = backend_snapshots(launch, compiler.writer.directory)
    assert binding[0]['verified'] and binding[0]['snapshot_count'] == 2
    assert all(stage.endswith('-hivm-graph-sync-solver') for stage in snapshots)
    assert verified(neg(8192, lowered=snapshots), 'FA-PAT-VEC-LOAD')
    for artifact in launch['artifacts']:
        if artifact['stage'] == 'backend-0000-stderr':
            artifact['stage'] = 'backend-0001-stderr'
    record['artifacts'] = ['backend-0001-stderr' if stage == 'backend-0000-stderr' else stage
                           for stage in record['artifacts']]
    assert not backend_snapshots(launch, compiler.writer.directory)[0]


@pytest.mark.parametrize('change', ['input', 'binary', 'kernel', 'returncode', 'truncated', 'executable', 'ambiguous'])
def test_unbound_failed_foreign_or_truncated_backend_text_never_enters_recognizer(compiler, change):
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    launch = launch_for(compiler, capture)
    record = launch['backend_compilations'][0]
    if change == 'input':
        record['input_sha256'] = '0' * 64
    elif change == 'binary':
        record['output_sha256'] = '0' * 64
    elif change == 'kernel':
        launch['name'] = 'unrelated'
    elif change == 'returncode':
        record['returncode'] = 9
    elif change == 'executable':
        record['executable_sha256'] = ''
    elif change == 'ambiguous':
        launch['artifacts'].append(copy.deepcopy(launch['artifacts'][0]))
    else:
        for artifact in launch['artifacts']:
            if artifact['stage'].endswith('-stderr'):
                previous = artifact['stage']
                artifact['stage'] += '-truncated'
                record['artifacts'] = [artifact['stage'] if stage == previous else stage
                                       for stage in record['artifacts']]
    snapshots, bindings = backend_snapshots(launch, compiler.writer.directory)
    assert not snapshots


def test_unsupported_dump_flags_are_not_guessed_or_injected(compiler, monkeypatch):
    executable = compiler.executable
    executable.write_text(
        executable.read_text().replace(
            '--mlir-print-ir-after-all  Print snapshots',
            '--unrelated  Some option'))
    with BackendCapture(compiler.writer.artifact) as capture:
        result = subprocess.run(compiler.command(), capture_output=True, check=True)
    assert result.stderr == b''
    assert '--mlir-print-ir-after-all' not in capture.records[0]['command']
    assert capture.records[0]['notes']
    assert not backend_snapshots(launch_for(compiler, capture), compiler.writer.directory)[0]


def test_failed_compiler_preserves_exception_streams_provenance_and_restores_hook(compiler):
    compiler.source.rename(compiler.source.with_name('failure.mlir'))
    compiler.source = compiler.source.with_name('failure.mlir')

    def failed_command():
        return [str(compiler.executable), str(compiler.source), '-o', str(compiler.output)]

    compiler.command = failed_command
    original = subprocess.run
    with pytest.raises(subprocess.CalledProcessError) as failure:
        with BackendCapture(compiler.writer.artifact) as capture:
            subprocess.run(compiler.command(), capture_output=True, check=True)
    assert subprocess.run is original and failure.value.returncode == 9
    assert b'IR Dump After Lowering' in failure.value.stderr
    assert capture.records[0]['returncode'] == 9
    assert any(entry['stage'].endswith('-stderr') for entry in capture.artifacts)


def test_no_hook_without_opt_in_and_unrelated_processes_remain_unmodified(compiler, monkeypatch):
    original = subprocess.run
    monkeypatch.delenv('TRITON_EVAL_CAPTURE_BACKEND')
    with BackendCapture(compiler.writer.artifact) as capture:
        assert subprocess.run is original
    assert not capture.records
    monkeypatch.setenv('TRITON_EVAL_CAPTURE_BACKEND', '1')
    with BackendCapture(compiler.writer.artifact) as capture:
        result = subprocess.run([sys.executable, '-c', 'print(42)'], capture_output=True, text=True)
    assert result.stdout == '42\n' and not capture.records


def test_valid_before_snapshot_is_supported_without_treating_generic_adapter_as_proof():
    header = '// IR Dump Before Scheduling\n'
    assert not compiler_passes(header + ASCEND_VECTOR, '_witness_neg')
    snapshots = compiler_passes(header + ASCEND_VECTOR, '_witness_neg', include_before=True)
    assert verified(neg(8192, lowered=snapshots), 'FA-PAT-VEC-LOAD')
    assert not compiler_passes(header + ASCEND_VECTOR[:-2], '_witness_neg', include_before=True)
    generic = ASCEND_VECTOR.replace('hivm.hir.load', 'memref.copy').replace('#hivm.address_space<ub>', '')
    assert not verified(neg(8192, lowered=compiler_passes(header + generic, '_witness_neg', include_before=True)),
                        'FA-PAT-VEC-LOAD')


def test_exception_inside_jit_restores_both_hooks_and_saves_failed_backend_artifacts(compiler):
    class JIT:
        def __init__(self, command):
            self.command = command

        def run(self, *args, **kwargs):
            subprocess.run(self.command, capture_output=True, check=True)
            raise RuntimeError('synthetic JIT failure')
    original = JIT.run
    original_process = subprocess.run
    triton = SimpleNamespace(runtime=SimpleNamespace(jit=SimpleNamespace(JITFunction=JIT)))
    with pytest.raises(RuntimeError, match='JIT failure'):
        with CompilationCapture(triton, compiler.writer.directory, {}) as capture:
            JIT(compiler.command()).run()
    assert JIT.run is original and subprocess.run is original_process
    assert capture.payload['failed_backend_compilations'][0]['returncode'] == 0
    assert capture.payload['failed_compilation_artifacts']


def test_control_flow_without_path_proof_is_not_linearized_into_memory_evidence():
    body = ASCEND_VECTOR.replace(' hivm.hir.load', ' scf.if %condition {\n hivm.hir.load', 1)
    body = body.replace(' hivm.hir.store', ' }\n hivm.hir.store', 1)
    assert not verified(neg(8192, lowered={'compiler-pass-0001': body}), 'FA-PAT-VEC-LOAD')


def test_raw_unbound_backend_logs_cannot_bypass_snapshot_identity_checks(suite):
    payload = json.loads(suite.path.read_text())
    writer = CompilationCapture(None, suite.path.parent, {})
    raw = writer.artifact('backend-0000-stderr', '// IR Dump After Lowering\n' + ASCEND_VECTOR)
    payload['launches'][0]['artifacts'].append(raw)
    suite.path.write_text(json.dumps(payload))
    captured = observation(suite.case, suite.path, payload['manifest_hash'], payload['run_id'])
    assert raw['stage'] not in captured['launches'][0]['stage_text']
    assert not captured['launches'][0]['backend_binding']


def test_generated_snapshots_are_captured_without_inventing_a_pass_phase(compiler):
    program = compiler.executable.read_text().replace("output.write_bytes(b'synthetic-unit-binary')",
                                                      ("output.write_bytes(b'synthetic-unit-binary')"
                                                       "\nsource.with_name('snapshot.mlir')"
                                                       ".write_text(source.read_text())"))
    compiler.executable.write_text(program)
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    snapshots, bindings = backend_snapshots(launch_for(compiler, capture), compiler.writer.directory)
    assert bindings[0]['verified']
    assert any('compiler-snapshot-saved' in name for name in snapshots)
    assert not any('IR Dump After Saved' in text for text in snapshots.values())


def test_backend_evidence_reuse_requires_identical_input_binary_manifest_and_run_directory(compiler):
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    function, kernel = _assert_specialization_reuse(compiler, capture)
    changed = CompilationCapture(None, compiler.writer.directory.with_name('third'), {})
    other = SimpleNamespace(asm={'ttadapter': ASCEND_VECTOR, 'npubin': b'changed-binary'})
    changed.record(function, other, (), dict(grid=(1,)))
    assert not changed.payload['launches'][0]['backend_compilations']
    different_manifest = CompilationCapture(
        None, compiler.writer.directory.with_name('fourth'), {
            'manifest_hash': 'other'})
    different_manifest.record(function, kernel, (), dict(grid=(1,)))
    assert not different_manifest.payload['launches'][0]['backend_compilations']
    other_run = CompilationCapture(None, compiler.writer.directory.parent / 'another-run/case', {})
    other_run.directory.mkdir(parents=True)
    other_run.record(function, kernel, (), dict(grid=(1,)))
    assert not other_run.payload['launches'][0]['backend_compilations']


def _assert_specialization_reuse(compiler, capture):
    function = SimpleNamespace(
        arg_names=[],
        constexprs=[],
        src='def _witness_neg():\n    pass\n',
        __name__='_witness_neg')
    kernel = SimpleNamespace(asm={'ttadapter': ASCEND_VECTOR, 'npubin': compiler.output.read_bytes()})
    compiler.writer.record(function, kernel, (), dict(grid=(1,)), backend=capture)
    reused = CompilationCapture(None, compiler.writer.directory.with_name('second'), {})
    reused.record(function, kernel, (), dict(grid=(1,)))
    launch = reused.payload['launches'][0]
    assert launch['backend_cache_reused']
    assert backend_snapshots(launch, reused.directory)[1][0]['verified']
    return function, kernel


@pytest.fixture
def prefix_compiler(compiler):
    compiler.executable.write_text(f'''#!{sys.executable}
import pathlib, sys
if '--help' in sys.argv:
    print('  --mlir-print-ir-after-all  Print snapshots')
    print('  --mlir-print-ir-module-scope  Print modules')
    print('  --mlir-print-ir-tree-dir=<string>  Snapshot directory')
    print('  --bishengir-print-ir-after=<pass-arg>  Backend snapshots')
    sys.exit(0)
source = pathlib.Path(sys.argv[1])
argument = next((value[3:] for value in sys.argv if value.startswith('-o=')), None)
output = pathlib.Path(argument or sys.argv[sys.argv.index('-o')+1])
output.with_suffix('.o').write_bytes(b'synthetic-unit-binary')
output.with_suffix('.json').write_text('{{"not": "a binary"}}')
directory = next(value.split('=',1)[1] for value in sys.argv if value.startswith('--mlir-print-ir-tree-dir='))
snapshot = pathlib.Path(directory)/'kernel'/'passes'/'after.mlir'
snapshot.parent.mkdir(parents=True, exist_ok=True)
if '--bishengir-print-ir-after=canonicalize' in sys.argv:
    snapshot.write_text(source.read_text())
''')
    compiler.output = compiler.source.with_name('kernel.o')
    compiler.prefix = compiler.source.with_name('kernel')

    def snapshot_command():
        return [str(compiler.executable), str(compiler.source), '-o', str(compiler.prefix)]

    compiler.command = snapshot_command
    return compiler


@pytest.mark.parametrize('query', ['--help', '--help-hidden', '--version', '-h'])
def test_compiler_queries_are_forwarded_unchanged_and_not_counted_as_compilations(compiler, query):
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, b'help')
    capture = BackendCapture(compiler.writer.artifact)
    command = [str(compiler.executable), query]
    result = capture.invoke(original, command, capture_output=True)
    assert result.stdout == b'help'
    assert calls == [((command,), {'capture_output': True})]
    assert not capture.records and not capture.artifacts


@pytest.mark.parametrize('equal_output_option', [False, True])
def test_output_prefix_and_nested_file_snapshots_bind_to_the_exact_returned_binary(
        prefix_compiler, equal_output_option):
    compiler = prefix_compiler
    command = compiler.command()
    if equal_output_option:
        command = command[:-2] + ['-o=' + str(compiler.prefix)]
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(command, capture_output=True, check=True)
    assert not compiler.prefix.exists()
    record = capture.records[0]
    assert len(record['output_candidates']) == 2
    assert '--bishengir-print-ir-after=canonicalize' in record['command']
    assert not any('triton-eval-ir-' in path.name for path in compiler.source.parent.iterdir())
    assert 'existing_files' not in record
    snapshots, bindings = backend_snapshots(launch_for(compiler, capture), compiler.writer.directory)
    assert bindings[0]['verified'] and bindings[0]['matched_output_stages']
    assert bindings[0]['snapshot_count'] == 1
    assert verified(neg(8192, lowered=snapshots), 'FA-PAT-VEC-LOAD')
    assert all('compiler-snapshot-saved' in stage for stage in snapshots)


@pytest.mark.parametrize('damage', ['foreign_binary', 'tampered_candidate', 'missing_candidate', 'duplicate_candidate'])
def test_prefix_outputs_require_indexed_unambiguous_byte_exact_binding(prefix_compiler, damage):
    compiler = prefix_compiler
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    launch = launch_for(compiler, capture)
    record = launch['backend_compilations'][0]
    if damage == 'foreign_binary':
        launch['artifacts'] = [artifact for artifact in launch['artifacts'] if artifact['stage'] != 'npubin']
        launch['artifacts'].append(compiler.writer.artifact('npubin', b'foreign'))
    elif damage == 'tampered_candidate':
        record['output_candidates'][0]['sha256'] = '0' * 64
    elif damage == 'missing_candidate':
        launch['artifacts'] = [artifact for artifact in launch['artifacts']
                               if artifact['stage'] != record['output_candidates'][0]['stage']]
    else:
        stage = record['output_candidates'][0]['stage']
        launch['artifacts'].append(next(copy.deepcopy(artifact) for artifact in launch['artifacts']
                                        if artifact['stage'] == stage))
    if damage == 'foreign_binary':
        snapshots, bindings = backend_snapshots(launch, compiler.writer.directory)
        assert not snapshots and not bindings[0]['verified']
    else:
        with pytest.raises(ValueError):
            backend_snapshots(launch, compiler.writer.directory)


def test_unchanged_stale_output_is_not_evidence_of_this_invocation(prefix_compiler):
    compiler = prefix_compiler
    compiler.output.write_bytes(b'synthetic-unit-binary')
    compiler.executable.write_text(compiler.executable.read_text().replace(
        "output.with_suffix('.o').write_bytes(b'synthetic-unit-binary')", 'pass'))
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    snapshots, bindings = backend_snapshots(launch_for(compiler, capture), compiler.writer.directory)
    assert not snapshots and not bindings[0]['verified']


def test_symlink_output_cannot_supply_backend_identity(prefix_compiler):
    compiler = prefix_compiler
    outside = compiler.source.with_name('unrelated.bin')
    outside.write_bytes(b'synthetic-unit-binary')
    compiler.output.symlink_to(outside)
    compiler.executable.write_text(compiler.executable.read_text().replace(
        "output.with_suffix('.o').write_bytes(b'synthetic-unit-binary')", 'pass'))
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    assert not backend_snapshots(launch_for(compiler, capture), compiler.writer.directory)[0]


def test_prefix_output_identity_allows_same_run_specialization_reuse(prefix_compiler):
    compiler = prefix_compiler
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    _assert_specialization_reuse(compiler, capture)


def test_output_tree_limits_leave_explicit_missing_evidence(prefix_compiler, monkeypatch):
    import triton_eval.backend_capture as module
    original = module.compiler_files
    calls = []

    def bounded(directory):
        calls.append(directory)
        if len(calls) > 1:
            raise ValueError('compiler output directory depth exceeds capture limit')
        return original(directory)
    monkeypatch.setattr(module, 'compiler_files', bounded)
    with BackendCapture(prefix_compiler.writer.artifact) as capture:
        result = subprocess.run(prefix_compiler.command(), capture_output=True)
    assert result.returncode == 0
    assert not capture.records[0].get('captured')
    assert any('backend output unavailable' in note for note in capture.records[0]['notes'])


def test_output_prefix_can_be_a_directory_without_guessing_a_binary_extension(prefix_compiler):
    compiler = prefix_compiler
    compiler.executable.write_text(compiler.executable.read_text().replace(
        "output.with_suffix('.o').write_bytes(b'synthetic-unit-binary')",
        "output.mkdir()\n(output/'payload.custom').write_bytes(b'synthetic-unit-binary')"))
    compiler.output = compiler.prefix / 'payload.custom'
    with BackendCapture(compiler.writer.artifact) as capture:
        subprocess.run(compiler.command(), capture_output=True, check=True)
    snapshots, bindings = backend_snapshots(launch_for(compiler, capture), compiler.writer.directory)
    assert bindings[0]['verified'] and snapshots
    assert any(candidate['path'].endswith('/payload.custom') for candidate in capture.records[0]['output_candidates'])


def test_directory_scan_rejects_depth_and_count_exhaustion(tmp_path, monkeypatch):
    import triton_eval.backend_capture as module
    (tmp_path / 'nested').mkdir()
    monkeypatch.setattr(module, 'MAX_DEPTH', 0)
    with pytest.raises(ValueError, match='depth'):
        module.compiler_files(tmp_path)
    monkeypatch.setattr(module, 'MAX_DEPTH', 8)
    monkeypatch.setattr(module, 'MAX_FILES', 0)
    with pytest.raises(ValueError, match='count'):
        module.compiler_files(tmp_path)


def test_prefix_compile_failure_keeps_collected_outputs_but_cannot_certify(prefix_compiler):
    compiler = prefix_compiler
    compiler.executable.write_text(compiler.executable.read_text() + '\nsys.exit(9)\n')
    original = subprocess.run
    with pytest.raises(subprocess.CalledProcessError):
        with BackendCapture(compiler.writer.artifact) as capture:
            subprocess.run(compiler.command(), capture_output=True, check=True)
    assert subprocess.run is original
    assert capture.records[0]['output_candidates'] and capture.records[0]['returncode'] == 9
    snapshots, bindings = backend_snapshots(launch_for(compiler, capture), compiler.writer.directory)
    assert not snapshots and not bindings[0]['verified']
    assert not any('triton-eval-ir-' in path.name for path in compiler.source.parent.iterdir())
