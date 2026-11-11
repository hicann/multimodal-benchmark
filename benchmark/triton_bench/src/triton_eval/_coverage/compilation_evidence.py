# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Optional NPU-side capture of JIT launches and actual returned compiler artifacts.

No Triton import on a developer CPU. An unsupported compiler hook produces an
explicit diagnostic, never a successful coverage observation. The wrapper is
installed only around the correctness launch, restored before performance runs.
"""
from __future__ import annotations
import gzip
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


BACKEND_CACHE = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def compiler_stderr():
    if os.environ.get('TRITON_EVAL_CAPTURE_PASSES') != '1':
        yield None
        return
    with tempfile.TemporaryFile() as stream:
        sys.stderr.flush()
        original = os.dup(2)
        try:
            os.dup2(stream.fileno(), 2)
            yield stream
        finally:
            sys.stderr.flush()
            os.dup2(original, 2)
            os.close(original)


def complete_module(text, kernel_name):
    signature = r'(?:tt\.func|func\.func)\s+(?:public\s+)?@' + re.escape(kernel_name) + r'\('
    stripped = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    stripped = re.sub(r'//[^\n]*', '', stripped)
    if (len(re.findall(signature, stripped)) != 1
            or not re.search(r'(?m)^\s*(?:builtin\.)?module\b', stripped)):
        return False
    depth = 0
    for character in stripped:
        if character == '{':
            depth += 1
        elif character == '}':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and '{' in stripped


def function_snapshot(text, kernel_name):
    text = text.strip()
    lines = text.splitlines()
    signature = r'func\.func\s+(?:public\s+)?@' + re.escape(kernel_name) + r'\('
    if not lines or not re.match(signature, lines[0]) or not lines[0].endswith('{'):
        return None
    masked = re.sub(r'"(?:\\.|[^"\\])*"', lambda match: ' ' * len(match[0]), text)
    masked = re.sub(r'//[^\n]*', lambda match: ' ' * len(match[0]), masked)
    if (len(re.findall(r'\b(?:func\.func|tt\.func)\b', masked)) != 1
            or 'hacc.function_kind<HOST>' in masked):
        return None
    start = lines[0].rfind('{')
    if masked[:start].count('{') != masked[:start].count('}'):
        return None
    return _snapshot_body(text, masked, start)


def _snapshot_body(text, masked, start):
    depth = 0
    for index in range(start, len(masked)):
        if masked[index] == '{':
            depth += 1
        elif masked[index] == '}':
            depth -= 1
            if depth == 0:
                return _snapshot_tail(text, index)
    return None


def _snapshot_tail(text, index):
    tail = text[index + 1:].strip().splitlines()
    if any(_unexpected_snapshot_warning(line) for line in tail if line.strip()):
        return None
    body = text[:index + 1]
    return body if re.search(r'(?m)^\s*(?:func\.)?return\b[^\n]*\n\s*}$', body) else None


def _unexpected_snapshot_warning(line):
    pattern = (
        r'(?:warning: overriding the module target triple with '
        r'aarch64-unknown-linux-gnu \[-Woverride-module\]|\d+ warnings? generated\.)')
    return not re.fullmatch(pattern, line.strip())


def compiler_passes(text, kernel_name, *, include_before=False, allow_function=False):
    sections = re.split(r'(?m)^//[^\n]*IR Dump (After|Before)[^\n]*$', text)
    headers = re.findall(r'(?m)^//[^\n]*IR Dump (?:After|Before)[^\n]*$', text)
    result = {}
    for index in range(1, len(sections) - 1, 2):
        module = sections[index + 1]
        if sections[index] != 'After' and not include_before:
            continue
        if not complete_module(module, kernel_name):
            module = function_snapshot(module, kernel_name) if allow_function else None
            if module is None:
                continue
        phase = sections[index].lower() + '-' if include_before else ''
        pass_name = re.search(r'\(([\w.-]+)\)', headers[index // 2]) if allow_function else None
        suffix = '-' + pass_name[1] if pass_name else ''
        result[f'compiler-pass-{phase}{index // 2:04d}{suffix}'] = module
    return result


def _matching_backend_records(records, key):
    matching = []
    for record in records:
        if record.get('returncode') != 0:
            continue
        if record.get('input_sha256') != key[-2]:
            continue
        if record.get('output_sha256') == key[-1]:
            matching.append(record)
    return matching


class CompilationCapture:
    def __init__(self, triton, directory, identity):
        self.triton, self.directory = triton, Path(directory)
        self.payload = dict(format=2, **identity, launches=[], diagnostics=[], synchronized=False)
        self.original = None
        self.cls = None
        self.tensor_ids = {}
        self.storage_ids = {}
        self.backend_probes = {}

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self.cls = self.triton.runtime.jit.JITFunction
            self.original = self.cls.run
            capture = self

            def observed(function, *args, **kwargs):
                from .backend_capture import BackendCapture
                backend = BackendCapture(capture.artifact, capture.backend_probes)
                dump = None
                try:
                    with backend, compiler_stderr() as stream:
                        try:
                            kernel = capture.original(function, *args, **kwargs)
                        finally:
                            dump = _read_compiler_stream(stream)
                except BaseException as exc:
                    capture.payload['diagnostics'].append(f"JIT invocation failed: {type(exc).__name__}")
                    if dump:
                        entry = capture.artifact(
                            'failed_compiler_stderr', dump[:64 * 1024 * 1024].decode(errors='replace'))
                        capture.payload.setdefault('failed_compilation_artifacts', []).append(entry)
                    if backend.records:
                        capture.payload.setdefault('failed_backend_compilations', []).extend(backend.records)
                        capture.payload.setdefault('failed_compilation_artifacts', []).extend(backend.artifacts)
                    raise
                try:
                    capture.record(function, kernel, args, kwargs, dump, backend)
                except Exception as exc:
                    capture.payload['diagnostics'].append(f"artifact capture unsupported: {type(exc).__name__}: {exc}")
                return kernel
            self.cls.run = observed
        except Exception as exc:
            self.payload['diagnostics'].append(f"JIT hook unsupported: {type(exc).__name__}: {exc}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original is not None:
            self.cls.run = self.original
        if exc_type:
            self.payload['diagnostics'].append(f"implementation exit: {exc_type.__name__}")
        (self.directory / 'observation.json').write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return False

    @staticmethod
    def _compiler_options(metadata):
        options = {}
        for name in ('num_warps', 'num_stages', 'num_ctas', 'shared', 'cluster_dims'):
            value = metadata.get(name) if isinstance(metadata, dict) else getattr(metadata, name, None)
            if isinstance(value, (bool, int, float, str)):
                options[name] = value
            elif isinstance(value, (tuple, list)) and all(isinstance(v, int) for v in value):
                options[name] = list(value)
        return options

    def tensor_metadata(self, tensor):
        key = (str(tensor.device), tensor.data_ptr())
        storage = (str(tensor.device), tensor.untyped_storage().data_ptr())
        return dict(tensor_id=self.tensor_ids.setdefault(key, len(self.tensor_ids)),
                    storage_id=self.storage_ids.setdefault(storage, len(self.storage_ids)),
                    dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride()))

    def bind_inputs(self, inputs):
        self._bind('input_bindings', inputs)

    def bind_outputs(self, outputs):
        self._bind('output_bindings', outputs)

    def artifact(self, stage, value):
        data = value.encode() if isinstance(value, str) else bytes(value)
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', str(stage))
        packed = gzip.compress(data, mtime=0)
        filename = f"{digest(data)}-{safe}.artifact.gz"
        blob_dir = self.directory.parent / 'blobs'
        blob_dir.mkdir(exist_ok=True)
        (blob_dir / filename).write_bytes(packed)
        return dict(stage=str(stage), path='../blobs/' + filename, sha256=digest(packed),
                    content_sha256=digest(data), compression='gzip',
                    kind='text' if isinstance(value, str) else 'binary')

    def record(self, function, kernel, args, kwargs, pass_dump=None, backend=None):
        if kwargs.get('warmup', False):
            return
        arguments, runtime_names = self.encode_arguments(function, args, kwargs)
        blob_dir = self.directory.parent / 'blobs'
        blob_dir.mkdir(exist_ok=True)
        stages = dict(getattr(kernel, 'asm', {}))
        backend_records, backend_artifacts, reused = self._backend_records(stages, backend)
        artifacts = self._stage_artifacts(stages, pass_dump)
        artifacts.extend(backend_artifacts)
        source = getattr(function, 'src', '')
        source_data = source.encode() if isinstance(source, str) else b''
        source_file = f"{digest(source_data)}-source.py"
        (blob_dir / source_file).write_bytes(source_data)
        grid = kwargs.get('grid')
        grid = list(grid) if isinstance(grid, (tuple, list)) and all(isinstance(v, int) for v in grid) else None
        options = self._compiler_options(getattr(kernel, 'metadata', None))
        kernel_hash = getattr(kernel, 'hash', None)
        self.payload['launches'].append(dict(
            name=getattr(function, '__name__', getattr(function, 'name', 'unknown')),
            arguments=arguments, runtime_arg_names=runtime_names,
            grid=grid, grid_observation='explicit' if grid else 'unresolved',
            compiler_specialization_hash=kernel_hash if isinstance(kernel_hash, str) else None,
            compiler_options=options,
            backend_compilations=backend_records, backend_cache_reused=reused,
            source=dict(path="../blobs/" + source_file, sha256=digest(source_data)), artifacts=artifacts,
            returned_compiled_kernel=bool(artifacts),
            note='artifact may be loaded from compiler cache; no claim of a fresh frontend invocation'))

    def encode_arguments(self, function, args, kwargs):
        names = list(getattr(function, 'arg_names', []))
        constexprs = set(getattr(function, 'constexprs', []))
        values = dict(zip(names, args))
        values.update({k: v for k, v in kwargs.items() if k in names})
        arguments, runtime_names = {}, []
        for i, name in enumerate(names):
            value = values.get(name)
            if i not in constexprs:
                runtime_names.append(name)
            record = self._argument(value, i in constexprs)
            if record is not None:
                arguments[name] = record
        return arguments, runtime_names

    def _bind(self, key, tensors):
        try:
            self.payload[key] = [self.tensor_metadata(t) for t in tensors]
        except Exception as exc:
            self.payload['diagnostics'].append(f"tensor binding unsupported: {type(exc).__name__}: {exc}")

    def _argument(self, value, constexpr):
        if hasattr(value, 'data_ptr'):
            return dict(kind='tensor', **self.tensor_metadata(value))
        if isinstance(value, (bool, int, float, str)):
            return dict(kind='scalar', value=value, constexpr=constexpr)
        if isinstance(value, tuple) and all(type(item) in (bool, int, float, str) for item in value):
            items = []
            for item in value:
                items.append(dict(kind='scalar', value=item, constexpr=constexpr))
            return dict(kind='tuple', constexpr=constexpr, items=items)
        return None

    def _backend_records(self, stages, backend):
        backend_records = backend.records if backend is not None else []
        backend_artifacts = backend.artifacts if backend is not None else []
        adapter, binary = stages.get('ttadapter'), stages.get('npubin')
        reused = False
        if (os.environ.get('TRITON_EVAL_CAPTURE_BACKEND') == '1' and isinstance(adapter, str)
                and isinstance(binary, (bytes, bytearray))):
            key = (str(self.directory.parent.resolve()), self.payload.get('manifest_hash'),
                   digest(adapter.encode()), digest(binary))
            for record in backend_records:
                if any(candidate.get('sha256') == key[-1] for candidate in record.get('output_candidates', [])):
                    record['output_sha256'] = key[-1]
            valid = _matching_backend_records(backend_records, key)
            if valid:
                BACKEND_CACHE[key] = copy.deepcopy((backend_records, backend_artifacts))
            elif not backend_records and key in BACKEND_CACHE:
                backend_records, backend_artifacts = copy.deepcopy(BACKEND_CACHE[key])
                reused = True
        return backend_records, backend_artifacts, reused

    def _stage_artifacts(self, stages, pass_dump):
        artifacts = []
        if isinstance(pass_dump, bytes) and pass_dump:
            if len(pass_dump) <= 64 * 1024 * 1024:
                stages['compiler_stderr'] = pass_dump.decode(errors='replace')
            else:
                stages['compiler_stderr_truncated'] = pass_dump[:64 * 1024 * 1024].decode(errors='replace')
        for key, value in stages.items():
            if not isinstance(value, (str, bytes, bytearray)):
                continue
            artifacts.append(self.artifact(key, value))
        return artifacts



def _read_compiler_stream(stream):
    if stream is None:
        return None
    stream.flush()
    stream.seek(0)
    return stream.read(64 * 1024 * 1024 + 1)
