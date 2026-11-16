#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Replay two archived compiler inputs to diagnose dumps, never certify coverage."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import logging

import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


from triton_eval.backend_capture import BackendCapture, LIMIT, file_digest
from triton_eval.compilation_evidence import CompilationCapture
from triton_eval.coverage_evidence import evidence_path, observation, read_blob, sha
from triton_eval.coverage_overlay import _read
from triton_eval.console import emit


MANIFEST = 'ad023d11150696ac84bdf3cf13a05f8575ef1d266702090acf2997cb55364138'
TARGETS = {'Neg': 'FA-PAT-VEC-LOAD', 'Matmul': 'FA-PAT-TILE-GEMM'}
VARIANTS = ('v15-control', 'vendor-whitelist')
PASSES = ('hivm-inject-sync', 'hivm-inject-block-sync', 'hivm-graph-sync-solver')
SEMANTIC_OPTIONS = frozenset((
    '--target=Ascend910B3', '--enable-auto-multi-buffer=True',
    '--enable-auto-bind-sub-block=True', '--enable-hfusion-compile=true',
    '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true'))
DUMP_OPTIONS = frozenset(('--mlir-print-ir-after-all', '--mlir-print-ir-module-scope',
                          '--bishengir-print-ir-after', '--mlir-print-ir-tree-dir'))


def _invalid_replay_signature(inputs, outputs, options):
    if (len(inputs) != 1):
        return True
    if (outputs != 1):
        return True
    if (set(options) != SEMANTIC_OPTIONS):
        return True
    return bool(len(options) != len(SEMANTIC_OPTIONS))


def _consume_option_value(arguments, description):
    if not arguments or arguments[0].startswith('-'):
        raise ValueError('missing archived ' + description)
    return arguments.pop(0)


def semantic_options(command):
    arguments = list(command[1:])
    options, inputs, outputs = [], [], 0
    while arguments:
        argument = arguments.pop(0)
        if argument == '-o':
            _consume_option_value(arguments, 'output prefix')
            outputs += 1
            continue
        if argument.startswith('-o='):
            outputs += 1
            continue
        if argument in SEMANTIC_OPTIONS:
            options.append(argument)
            continue
        if argument.split('=', 1)[0] in DUMP_OPTIONS:
            if argument in ('--bishengir-print-ir-after', '--mlir-print-ir-tree-dir'):
                _consume_option_value(arguments, 'dump option value')
            continue
        if argument.endswith('.mlir') and not argument.startswith('-'):
            inputs.append(argument)
            continue
        raise ValueError('unreviewed archived compiler argument: ' + argument)
    if _invalid_replay_signature(inputs, outputs, options):
        raise ValueError('archived compiler command differs from reviewed v15 configuration')
    return options


def load_inputs(results, submission):
    data = _read(results, submission)
    if (data['report']['manifest_hash'] != MANIFEST
            or set(data['summary']['selected_operators']) != set(TARGETS)
            or data['report'].get('selected_operator_validation_passed') is not True):
        raise ValueError('requires the archived v15 Neg/Matmul passing regression')
    prepared = []
    for operator, atom in TARGETS.items():
        candidates = sorted((case for case in data['cases'].values()
                             if case.operator == operator and atom in case.feature_atoms), key=lambda case: case.code)
        for case in candidates:
            paths = list((data['root'] / operator).glob('run-*/compilation/' + case.code + '/observation.json'))
            if len(paths) != 1 or str(paths[0].relative_to(data['root'])) not in data['report']['evidence_sha256']:
                raise ValueError('missing indexed target observation')
            captured = observation(case, paths[0], MANIFEST, data['rows'][case.code]['run_id'])
            launches = captured['launches']
            if len(launches) != 1 or launches[0].get('backend_cache_reused'):
                continue
            launch = launches[0]
            _check_target_artifacts(launch, paths[0], data)
            records = launch.get('backend_compilations', [])
            bindings = launch.get('backend_binding', [])
            if len(records) != 1 or len(bindings) != 1 or bindings[0].get('verified') is not True:
                raise ValueError('target input/binary binding is not verified')
            artifacts = {entry['stage']: entry for entry in launch['artifacts']}
            prepared.append(dict(operator=operator, atom=atom, case_code=case.code,
                                 kernel=launch['name'], input=read_blob(paths[0].parent, artifacts['ttadapter']),
                                 binary=read_blob(paths[0].parent, artifacts['npubin']),
                                 tool_sha256=records[0]['executable_sha256'],
                                 options=semantic_options(records[0]['command'])))
            break
        else:
            raise ValueError('no actual bound compilation for ' + operator)
    return data, prepared


def _check_target_artifacts(launch, path, data):
    for entry in [launch['source'], *launch['artifacts']]:
        path = evidence_path(path.parent, entry['path'], data['root'])
        if str(path.relative_to(data['root'])) not in data['report']['evidence_sha256']:
            raise ValueError('unindexed target compiler artifact')


def installed_compiler(explicit=None):
    if explicit is not None:
        path = Path(explicit)
    else:
        spec = importlib.util.find_spec('triton')
        if spec is None or spec.origin is None:
            raise ValueError('installed Triton is unavailable; use --prepare-only on CPU')
        path = Path(spec.origin).parent / 'backends/ascend/bishengir/bin/bishengir-compile'
    path = path.absolute()
    if path.name != 'bishengir-compile' or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('expected an installed executable named bishengir-compile')
    return path


def compiler_help(tool, output):
    writer = CompilationCapture(None, output / 'tool', {})
    writer.directory.mkdir(parents=True)
    artifacts, queries, texts = [], [], []
    flags = ['--help']
    while flags:
        flag = flags.pop(0)
        query = dict(option=flag, returncode=None)
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            _run_help_query(tool, flag, (stdout, stderr), query)
            chunks = _read_help_streams(writer, flag, (('stdout', stdout), ('stderr', stderr)),
                                        query, artifacts)
        queries.append(query)
        text = '\n'.join(chunks)
        if not (query['returncode'] == 0 and not query.get('truncated')):
            continue
        texts.append(text)
        if flag == '--help' and '--help-hidden' in text:
            flags.append('--help-hidden')
    metadata = dict(queries=queries, artifacts=artifacts)
    (writer.directory / 'probe.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if any(query['returncode'] or query.get('truncated') for query in queries):
        raise ValueError('compiler help failed or exceeded the diagnostic limit; see tool/probe.json')
    return '\n'.join(texts), metadata


def _run_help_query(tool, flag, streams, query):
    stdout, stderr = streams
    try:
        result = subprocess.run([str(tool), flag], stdout=stdout, stderr=stderr, timeout=15)
        query['returncode'] = result.returncode
    except subprocess.TimeoutExpired:
        query.update(returncode=124, error='compiler help timed out')


def _read_help_streams(writer, flag, streams, query, artifacts):
    chunks = []
    for name, stream in streams:
        stream.seek(0)
        raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            query['truncated'] = True
        chunks.append(raw[:1024 * 1024].decode(errors='replace'))
        artifacts.append(writer.artifact(flag[2:] + '-' + name, raw[:1024 * 1024]))
    return chunks


def compare_replays(replays):
    control, targeted = (replays[variant] for variant in VARIANTS)
    if not all(result['returncode'] == 0 for result in (control, targeted)):
        conclusion = 'compiler_replay_failed'
    elif not all(result['tool_identity_unchanged'] for result in (control, targeted)):
        conclusion = 'compiler_identity_changed'
    elif not all(result['binary_matches_archived'] for result in (control, targeted)):
        conclusion = 'archived_binary_not_reproduced'
    elif targeted['ready_for_review'] and not control['dump_observed']:
        conclusion = 'vendor_filter_restores_dump_in_this_replay'
    elif targeted['ready_for_review'] and control['ready_for_review']:
        conclusion = 'both_variants_produce_dump'
    elif control['ready_for_review'] and not targeted['dump_observed']:
        conclusion = 'only_control_produces_dump'
    elif control['dump_observed'] or targeted['dump_observed']:
        conclusion = 'dump_present_but_capture_needs_review'
    else:
        conclusion = 'backend_dump_still_missing'
    return dict(conclusion=conclusion, diagnostic_only=True, coverage_certified=False,
                execution_validation_performed=False)


def replay(item, tool, help_text, variant, output, timeout):
    if variant not in VARIANTS:
        raise ValueError('unknown diagnostic variant')
    if file_digest(tool) != item['tool_sha256']:
        raise ValueError('compiler identity changed before replay')
    writer = CompilationCapture(None, output / item['operator'] / variant, {})
    writer.directory.mkdir(parents=True)
    targeted = variant == 'vendor-whitelist'
    names = ','.join(PASSES)
    flags = (['--bishengir-print-ir-before=' + names, '--bishengir-print-ir-after=' + names]
             if targeted else ['--mlir-print-ir-after-all', '--mlir-print-ir-module-scope',
                               '--bishengir-print-ir-after=canonicalize'])
    probe = dict(executable=str(tool), executable_sha256=item['tool_sha256'], help=help_text,
                 flags=flags, tree_dump=not targeted, notes=[])
    capture = BackendCapture(writer.artifact, {(str(tool.resolve()), item['tool_sha256']): probe})
    capture.original = subprocess.run
    outcome = dict(variant=variant, diagnostic_only=True, coverage_certified=False,
                   flags=flags, timeout_seconds=timeout, environment_overrides={'MLIR_ENABLE_DUMP': '1'})
    environment = dict(os.environ, MLIR_ENABLE_DUMP='1')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='backend-dump-probe-') as temporary:
        directory = Path(temporary).resolve()
        source = directory / 'kernel.ttadapter.mlir'
        source.write_bytes(item['input'])
        command = [str(tool), str(source), *item['options'], '-o', str(directory / 'kernel')]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            outcome.update(_replay_command(capture, command, (stdout, stderr),
                                           dict(cwd=directory, env=environment, timeout=timeout)))
            _capture_replay_streams(capture, writer, (('stdout', stdout), ('stderr', stderr)))
    _check_replayed_binary(capture, writer, item, outcome)

    snapshots = _replay_snapshot_inventory(capture, writer, item)

    outcome.update(elapsed_seconds=round(time.monotonic() - started, 3), records=capture.records,
                   artifacts=capture.artifacts, snapshot_inventory=snapshots)
    outcome['dump_observed'] = any(snapshot['has_target_function'] and snapshot['hivm_operations']
                                   for snapshot in snapshots)
    _record_replay_readiness(outcome, capture, tool, item)

    (writer.directory / 'replay.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + '\n')
    return outcome


def _replay_command(capture, command, streams, options):
    stdout, stderr = streams
    outcome = {}
    try:
        result = capture.invoke(subprocess.run, command, stdout=stdout, stderr=stderr,
                                **options)
        outcome['returncode'] = result.returncode
    except subprocess.TimeoutExpired:
        outcome.update(returncode=124, error='compiler replay timed out')
    return outcome


def _capture_replay_streams(capture, writer, streams):
    for name, stream in streams:
        stream.seek(0)
        raw = stream.read(LIMIT + 1)
        if raw:
            stage = 'probe-' + name + ('-truncated' if len(raw) > LIMIT else '')
            capture.artifacts.append(writer.artifact(stage, raw[:LIMIT]))


def diagnose(results, submission, output, *, compiler=None, prepare_only=False, timeout=120):
    if not 0 < timeout <= 120:
        raise ValueError('diagnostic timeout must be between 1 and 120 seconds')
    data, inputs = load_inputs(results, submission)
    report = dict(scope='backend_dump_diagnostic', diagnostic_only=True, affects_coverage=False,
                  execution_validation_performed=False, coverage_certified=False, state='prepared',
                  source_manifest_hash=MANIFEST, source_report_sha256=sha(
                      (data['root'] / 'coverage-validation.json').read_bytes()),
                  source_evidence_files_checked=len(data['report']['evidence_sha256']),
                  source_environments=data.get('environments', []),
                  diagnostic_script_sha256=file_digest(Path(__file__)),
                  hypotheses_are_not_installed_build_facts=True, operators={}, ready_for_review=False)
    report['operators'] = _prepare_diagnostic_sources(inputs, output)
    (output / 'diagnostic-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if prepare_only:
        return report
    tool = installed_compiler(compiler)
    tool_hash = file_digest(tool)
    if any(item['tool_sha256'] != tool_hash for item in inputs):
        raise ValueError('installed compiler hash differs from the archived actual compiler')
    help_text, report['tool_probe'] = compiler_help(tool, output)
    options = set(re.findall(r'(?m)^\s*(--[\w-]+)(?=[=\s])', help_text))
    required = DUMP_OPTIONS | {'--bishengir-print-ir-before'}
    report['tool'] = dict(path=str(tool), sha256=tool_hash, advertised_options=sorted(options))
    (output / 'diagnostic-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not required <= options:
        raise ValueError('installed compiler help does not advertise the diagnostic options')
    for item in inputs:
        for variant in VARIANTS:
            result = replay(item, tool, help_text, variant, output, timeout)
            report['operators'][item['operator']]['replays'][variant] = result
            (output / 'diagnostic-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            emit(
                f"{item['operator']} {variant}: rc={result['returncode']}, "
                f"binary_match={result['binary_matches_archived']}, dump_observed={result['dump_observed']}",
                flush=True)
        entry = report['operators'][item['operator']]
        entry['comparison'] = compare_replays(entry['replays'])
        emit(item['operator'] + ': ' + entry['comparison']['conclusion'], flush=True)
    report['state'] = 'finished'
    report['ready_for_review'] = all(entry['replays']['vendor-whitelist']['ready_for_review']
                                     for entry in report['operators'].values())
    report['note'] = ('Dump availability and byte identity are diagnostics, not live dataflow certification or '
                      'execution validation.')
    return report


def _prepare_diagnostic_sources(inputs, output):
    operators = {}
    for item in inputs:
        writer = CompilationCapture(None, output / item['operator'] / 'source', {})
        writer.directory.mkdir(parents=True)
        operators[item['operator']] = dict(case_code=item['case_code'],
                                                     atom=item['atom'],
                                                     kernel=item['kernel'],
                                                     tool_sha256=item['tool_sha256'],
                                                     options=item['options'],
                                                     source_artifacts=[writer.artifact('ttadapter',
                                                                                       item['input']),
                                                                       writer.artifact('npubin',
                                                                                       item['binary'])],
                                                     replays={})
    return operators


def _repository_commit():
    executable = shutil.which('git')
    if executable is None:
        raise FileNotFoundError('Git executable is unavailable')
    return subprocess.check_output(
        [str(Path(executable).resolve()), '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='archived v15 evidence directory')
    parser.add_argument('--submission', type=Path, default=ROOT / 'examples/coverage-v15-ascend-submission')
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    record = (args.out or ROOT / 'runs/backend-diagnostics' / stamp).resolve()
    if record.is_relative_to(args.results.resolve()) or record.is_relative_to(args.submission.resolve()):
        parser.error('diagnostic output must not modify the archived results or submission')
    output = record / 'evidence'
    output.mkdir(parents=True, exist_ok=False)
    commit = _repository_commit()
    execution = dict(code_commit=commit, state='starting', scope='backend_dump_diagnostic')
    (record / 'execution.json').write_text(json.dumps(execution, indent=2) + '\n')
    report, status = {}, 1
    try:
        report = diagnose(args.results.resolve(), args.submission.resolve(), output,
                          compiler=args.compiler, prepare_only=args.prepare_only)
        status = 0 if args.prepare_only or report['ready_for_review'] else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        path = output / 'diagnostic-report.json'
        report = json.loads(path.read_text()) if path.is_file() else report
        report.update(scope='backend_dump_diagnostic', diagnostic_only=True, affects_coverage=False,
                      execution_validation_performed=False, coverage_certified=False, ready_for_review=False,
                      state='failed', error=str(error))
        emit('诊断失败：' + str(error), file=sys.stderr)
    (output / 'diagnostic-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    index = [dict(path=str(path.relative_to(output)), sha256=sha(path.read_bytes()))
             for path in sorted(output.rglob('*')) if path.is_file()]
    (output / 'artifacts.json').write_text(json.dumps(dict(files=index), indent=2) + '\n')
    execution.update(state='finished', returncode=status)
    (record / 'execution.json').write_text(json.dumps(execution, indent=2) + '\n')
    emit('结果目录：' + str(record))
    emit('本工具不执行 NPU 内核，不新增编译/执行认证覆盖。')
    if args.publish:
        from publish_ascend_results import publish
        try:
            publish(ROOT, record)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            from publish_ascend_results import redact
            emit('上传失败，原始诊断保留在本机：' + redact(str(error)), file=sys.stderr)
            return 2
    return status




def _replay_snapshot_inventory(capture, writer, item):
    snapshots = []
    for entry in capture.artifacts:
        stage = entry['stage']
        if stage not in ('probe-stdout', 'probe-stderr') and '-saved-' not in stage:
            continue
        text = read_blob(writer.directory, entry).decode(errors='replace')
        snapshots.append(dict(stage=stage, before=text.count('IR Dump Before'), after=text.count('IR Dump After'),
                              has_target_function=bool(
            re.search(
                r'\b(?:func\.func|tt\.func)\s+(?:public\s+)?@' +
                re.escape(
                    item['kernel']) +
                r'\(',
                text)),
            has_module=bool(re.search(r'(?m)^\s*(?:builtin\.)?module\b', text)),
            hivm_operations=sorted(set(re.findall(r'\bhivm\.hir\.[A-Za-z_0-9]+', text)))))
    return snapshots




def _check_replayed_binary(capture, writer, item, outcome):
    outputs = [entry for entry in capture.artifacts if re.fullmatch(r'backend-\d{4}-output-\d{4}', entry['stage'])]
    outcome['binary_matches_archived'] = any(read_blob(writer.directory, entry) == item['binary'] for entry in outputs)




def _record_replay_readiness(outcome, capture, tool, item):
    outcome['tool_identity_unchanged'] = file_digest(tool) == item['tool_sha256']
    outcome['ready_for_review'] = (
        outcome.get('returncode') == 0 and outcome['binary_matches_archived'] and outcome['dump_observed'] and not any(
            record.get('notes') for record in capture.records) and outcome['tool_identity_unchanged'] and not any(
            entry['stage'].endswith('-truncated') for entry in capture.artifacts))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
