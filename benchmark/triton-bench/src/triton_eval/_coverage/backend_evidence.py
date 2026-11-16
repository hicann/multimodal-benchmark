# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bind external compiler snapshots to the actual JIT input and returned binary."""
from __future__ import annotations

import hashlib
import re


def _stage_artifact(stages, stage):
    artifacts = stages.get(stage, [])
    if len(artifacts) != 1:
        raise ValueError('ambiguous or missing backend artifact: ' + stage)
    return artifacts[0]


def _missing_compiler_output_binding_part_1(outputs, matches, record, binary_hash):
    if not (outputs):
        return False
    return not (matches)


def _missing_compiler_output_binding_part_2(outputs, matches, record, binary_hash):
    if outputs:
        return False
    return record.get('output_sha256') != binary_hash


def _missing_compiler_output_binding(outputs, matches, record, binary_hash):
    if (_missing_compiler_output_binding_part_1(outputs, matches, record, binary_hash)):
        return True
    return bool(_missing_compiler_output_binding_part_2(outputs, matches, record, binary_hash))


def _output_matches(record, root, stages, prefix, binary):
    from .coverage_evidence import read_blob

    matches = []
    for output in record.get('output_candidates', []):
        stage = output.get('stage', '')
        if not re.fullmatch(re.escape(prefix) + r'-output-\d{4}', stage) or len(stages.get(stage, [])) != 1:
            raise ValueError('ambiguous or missing backend output artifact')
        content = read_blob(root, _stage_artifact(stages, stage))
        if hashlib.sha256(content).hexdigest() != output.get('sha256'):
            raise ValueError('backend output candidate SHA256 mismatch')
        if content == binary:
            matches.append(stage)
    return matches


def _bind_binary(launch, root, stages, prefix, record, result):
    from .coverage_evidence import read_blob

    if record.get('returncode') != 0 or record.get('captured') is not True:
        return False
    if (record.get('input_stage') != prefix + '-input'
            or not re.fullmatch(r'[0-9a-f]{64}', record.get('executable_sha256', ''))):
        return False
    required = [record.get('input_stage'), 'ttadapter', 'npubin']
    if any(len(stages.get(stage, [])) != 1 for stage in required):
        return False
    source = read_blob(root, _stage_artifact(stages, required[0]))
    adapter = read_blob(root, _stage_artifact(stages, 'ttadapter'))
    binary = read_blob(root, _stage_artifact(stages, 'npubin'))
    if source != adapter or hashlib.sha256(source).hexdigest() != record.get('input_sha256'):
        result['reason'] = 'compiler input differs from captured adapter'
        return False
    binary_hash = hashlib.sha256(binary).hexdigest()
    outputs = record.get('output_candidates', [])
    matches = _output_matches(record, root, stages, prefix, binary)
    if _missing_compiler_output_binding(outputs, matches, record, binary_hash):
        result['reason'] = 'no captured compiler output equals returned npubin'
        return False
    if record.get('output_sha256', binary_hash) != binary_hash:
        result['reason'] = 'compiler output hash differs from returned npubin'
        return False
    pattern = r'(?:tt\.func|func\.func)\s+(?:public\s+)?@' + re.escape(launch['name']) + r'\('
    if not re.search(pattern, adapter.decode()):
        return False
    result.update(verified=True, reason='compiler input equals captured adapter; output equals returned npubin',
                  executable_sha256=record['executable_sha256'], input_sha256=record['input_sha256'],
                  output_sha256=binary_hash, matched_output_stages=matches)
    return True


def _compiler_snapshots(launch, root, stages, prefix, record):
    from .compilation_evidence import compiler_passes, complete_module, function_snapshot
    from .coverage_evidence import read_blob

    snapshots = {}
    for stage in record.get('artifacts', []):
        if not re.fullmatch(re.escape(prefix) + r'-(?:stdout|stderr|saved-\d{4})', stage):
            continue
        if len(stages.get(stage, [])) != 1:
            raise ValueError('ambiguous or missing backend snapshot artifact')
        text = read_blob(root, _stage_artifact(stages, stage)).decode()
        if '-saved-' in stage:
            snapshot = text if complete_module(text, launch['name']) else function_snapshot(text, launch['name'])
            modules = {'compiler-snapshot-saved': snapshot} if snapshot else {}
        else:
            modules = compiler_passes(text, launch['name'], include_before=True, allow_function=True)
        snapshots.update({f'{prefix}-{stage}-{name}': module for name, module in modules.items()})
    return snapshots


def backend_snapshots(launch, root):
    stages = {}
    for artifact in launch['artifacts']:
        stages.setdefault(artifact['stage'], []).append(artifact)
    snapshots, binding = {}, []
    for index, record in enumerate(launch.get('backend_compilations', [])):
        prefix = f'backend-{index:04d}'
        result = dict(invocation=index, verified=False, reason='backend input/binary identity is not established')
        binding.append(result)
        if not _bind_binary(launch, root, stages, prefix, record, result):
            continue
        snapshots.update(_compiler_snapshots(launch, root, stages, prefix, record))
        result['snapshot_count'] = sum(name.startswith(f'backend-{index:04d}-') for name in snapshots)
    return snapshots, binding
