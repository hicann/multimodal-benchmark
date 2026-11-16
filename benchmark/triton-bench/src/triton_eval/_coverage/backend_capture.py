# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Capture the external Ascend compiler inside one correctness JIT invocation."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


LIMIT = 64 * 1024 * 1024
MAX_FILES = 512
MAX_DEPTH = 8


def _unsupported_compiler_command(kwargs, command):
    if (kwargs.get('shell')):
        return True
    if (not isinstance(command, (tuple, list))):
        return True
    if (not command):
        return True
    return bool(Path(os.fsdecode(command[0])).name != 'bishengir-compile')


def compiler_files(directory):
    directory = Path(directory)
    pending, files = [(directory, 0)], {}
    visited = 0
    while pending:
        parent, depth = pending.pop()
        for path in sorted(parent.iterdir()):
            visited += 1
            if visited > MAX_FILES:
                raise ValueError('compiler output file count exceeds capture limit')
            if path.is_symlink():
                continue
            is_directory = path.is_dir()
            if is_directory and depth >= MAX_DEPTH:
                raise ValueError('compiler output directory depth exceeds capture limit')
            if is_directory:
                pending.append((path, depth + 1))
                continue
            if path.is_file():
                status = path.stat()
                files[path] = (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)
    return files


def option_value(command, option):
    for index, argument in enumerate(command[1:], 1):
        if argument == option and index + 1 < len(command):
            return command[index + 1]
        if argument.startswith(option + '='):
            return argument[len(option) + 1:]
    return None


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _mlir_input_paths(command, working):
    paths = []
    for argument in command[1:]:
        if argument.endswith('.mlir') and not argument.startswith('-'):
            paths.append((working / argument).resolve())
    return paths


def _missing_compiler_flags(command, flags):
    missing = []
    for flag in flags:
        present = any(argument.split('=', 1)[0] == flag.split('=', 1)[0] for argument in command[1:])
        if not present:
            missing.append(flag)
    return missing


class BackendCapture:
    def __init__(self, artifact, probes=None):
        self.artifact = artifact
        self.probes = {} if probes is None else probes
        self.records, self.artifacts = [], []
        self.original = None

    def __enter__(self):
        if os.environ.get('TRITON_EVAL_CAPTURE_BACKEND') == '1':
            self.original = subprocess.run
            original = self.original

            def captured_run(*arguments, **keywords):
                return self.invoke(original, *arguments, **keywords)

            subprocess.run = captured_run
        return self

    def __exit__(self, *exception):
        if self.original is not None:
            subprocess.run = self.original
        return False

    @staticmethod
    def cleanup(record):
        record.pop('existing_files', None)
        directory = record.pop('capture_directory', None)
        if directory:
            shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _prepare_outputs(command, working, input_path, record):
        output = option_value(command, '-o')
        directories = {input_path.parent}
        if output:
            output_path = Path(os.path.abspath(working / output))
            if not any(parent.is_symlink() for parent in (output_path, *output_path.parents)):
                record['output_prefix'] = str(output_path)
                directories.add(output_path.parent)
        record['existing_files'] = {str(directory): compiler_files(directory) for directory in directories}

    def save(self, stage, value):
        entry = self.artifact(stage, value)
        self.artifacts.append(entry)
        return stage

    def probe(self, executable, kwargs):
        environment = kwargs.get('env')
        path = shutil.which(executable, path=(environment or os.environ).get('PATH'))
        if path is None:
            return dict(notes=['compiler executable not resolved'], flags=[])
        path = str(Path(path).resolve())
        key = (path, file_digest(path))
        if key not in self.probes:
            self._probe_compiler_help(path, key, kwargs)
        return self.probes[key]

    def invoke(self, original, *args, **kwargs):
        command = args[0] if args else kwargs.get('args')
        if (_unsupported_compiler_command(kwargs, command)):
            return original(*args, **kwargs)
        command = list(map(os.fsdecode, command))
        if any(argument in ('--help', '--help-hidden', '--version', '-h') for argument in command[1:]):
            return original(*args, **kwargs)
        prefix = f'backend-{len(self.records):04d}'
        record = dict(command=command[:], artifacts=[], notes=[], returncode=None)
        self.records.append(record)
        try:
            self._capture_input(command, kwargs, record, prefix)
        except (OSError, ValueError) as exc:
            record['notes'].append('backend provenance unavailable: ' + type(exc).__name__)
            command = list(map(os.fsdecode, args[0] if args else kwargs['args']))
        invocation = (command, *args[1:]) if args else ()
        invocation_kwargs = dict(kwargs)
        if not args:
            invocation_kwargs['args'] = command
        try:
            result = original(*invocation, **invocation_kwargs)
        except subprocess.CalledProcessError as exc:
            self.complete(record, prefix, command, kwargs, exc.returncode, exc.stdout, exc.stderr)
            raise
        except BaseException:
            self.cleanup(record)
            raise
        else:
            self.complete(record, prefix, command, kwargs, result.returncode, result.stdout, result.stderr)
            return result

    def complete(self, record, prefix, command, kwargs, returncode, stdout, stderr):
        record['returncode'] = returncode
        try:
            for name, value in (('stdout', stdout), ('stderr', stderr)):
                if not (value):
                    continue
                raw = value.encode() if isinstance(value, str) else value
                if len(raw) > LIMIT:
                    name += '-truncated'
                record['artifacts'].append(self.save(prefix + '-' + name, raw[:LIMIT].decode(errors='replace')))
            self._capture_outputs(record, prefix)
            if not record['output_candidates']:
                record['notes'].append('no fresh bounded output file found under compiler output prefix')
            record['captured'] = True
        except (OSError, ValueError, IndexError) as exc:
            record['notes'].append('backend output unavailable: ' + type(exc).__name__)
        finally:
            self.cleanup(record)

    def _probe_compiler_help(self, path, key, kwargs):
        environment = kwargs.get('env')
        try:
            result = self.original([path, '--help'], capture_output=True, timeout=15,
                                   cwd=kwargs.get('cwd'), env=environment)
            help_text = (result.stdout or b'') + (result.stderr or b'')
            if b'--help-hidden' in help_text:
                hidden = self.original([path, '--help-hidden'], capture_output=True, timeout=15,
                                       cwd=kwargs.get('cwd'), env=environment)
                if hidden.returncode == 0:
                    help_text += b'\n' + (hidden.stdout or b'') + (hidden.stderr or b'')
            text = help_text[:1024 * 1024].decode(errors='replace')
            options = set(re.findall(r'(?m)^\s*(--[\w-]+)(?=[=\s])', text)) if result.returncode == 0 else set()
            after_flags = ('--bishengir-print-ir-after-all', '--mlir-print-ir-after-all', '--print-ir-after-all')
            scope_flags = ('--bishengir-print-ir-module-scope', '--mlir-print-ir-module-scope',
                           '--print-ir-module-scope')
            after = next((flag for flag in after_flags if flag in options), None)
            scope = next((flag for flag in scope_flags if flag in options), None)
            flags = ([after] if after else []) + ([scope] if scope else [])
            vendor_filters = {'--bishengir-print-ir-before', '--bishengir-print-ir-after'} <= options
            if vendor_filters:
                passes = 'hivm-inject-sync,hivm-inject-block-sync,hivm-graph-sync-solver'
                flags = ['--bishengir-print-ir-before=' + passes, '--bishengir-print-ir-after=' + passes]
            elif '--bishengir-print-ir-after' in options:
                flags.append('--bishengir-print-ir-after=canonicalize')
            self.probes[key] = dict(executable=path, executable_sha256=key[1], help=text,
                                    flags=flags,
                                    tree_dump=not vendor_filters and '--mlir-print-ir-tree-dir' in options,
                                    notes=[] if after or '--bishengir-print-ir-after' in options else
                                    ['compiler help does not advertise a supported IR dump flag'])
        except (OSError, subprocess.SubprocessError) as exc:
            self.probes[key] = dict(executable=path, executable_sha256=key[1], flags=[],
                                    notes=['compiler help unavailable: ' + type(exc).__name__])

    def _capture_input(self, command, kwargs, record, prefix):
        probe = self.probe(command[0], kwargs)
        record.update({key: value for key, value in probe.items() if key != 'help'})
        record['notes'] = list(record['notes'])
        if 'help' in probe:
            record['artifacts'].append(self.save(prefix + '-help', probe['help']))
        working = Path(kwargs.get('cwd') or os.getcwd()).resolve()
        inputs = _mlir_input_paths(command, working)
        inputs = [path for path in inputs if path.is_file()]
        if len(inputs) != 1 or inputs[0].stat().st_size > LIMIT:
            record['notes'].append('requires one bounded MLIR compiler input')
            return
        content = inputs[0].read_bytes()
        record['input_sha256'] = hashlib.sha256(content).hexdigest()
        record['input_stage'] = self.save(prefix + '-input', content.decode())
        record['artifacts'].append(record['input_stage'])
        record['input_directory'] = str(inputs[0].parent)
        self._prepare_outputs(command, working, inputs[0], record)
        flags = _missing_compiler_flags(command, probe['flags'])
        if '--' not in command:
            if probe.get('tree_dump') and not option_value(command, '--mlir-print-ir-tree-dir'):
                record['capture_directory'] = tempfile.mkdtemp(prefix='triton-eval-ir-', dir=inputs[0].parent)
                flags.append('--mlir-print-ir-tree-dir=' + record['capture_directory'])
            command += flags
        record['command'] = command[:]

    def _capture_outputs(self, record, prefix):
        files = {}
        for directory, previous in record.get('existing_files', {}).items():
            for path, status in compiler_files(directory).items():
                if previous.get(path) != status:
                    files[path] = status
        output_prefix = Path(record['output_prefix']) if record.get('output_prefix') else None
        saved_bytes = 0
        record['output_candidates'] = []
        for path, status in sorted(files.items()):
            output_related = output_prefix is not None and (
                path == output_prefix or path.is_relative_to(output_prefix)
                or (path.parent == output_prefix.parent and path.name.startswith(output_prefix.name + '.')))
            if not output_related and path.suffix != '.mlir':
                continue
            if status[0] > LIMIT or saved_bytes + status[0] > LIMIT:
                record['notes'].append('compiler output exceeds capture byte limit: ' + path.name)
                continue
            content = path.read_bytes()
            saved_bytes += len(content)
            if output_related:
                stage = self.save(prefix + f'-output-{len(record["output_candidates"]):04d}', content)
                record['output_candidates'].append(dict(stage=stage, path=str(path),
                                                        sha256=hashlib.sha256(content).hexdigest()))
                record['artifacts'].append(stage)
                if path == output_prefix:
                    record['output_sha256'] = hashlib.sha256(content).hexdigest()
            if not (path.suffix == '.mlir'):
                continue
            try:
                text = content.decode()
            except UnicodeError:
                continue
            stage = self.save(prefix + f'-saved-{len(record["artifacts"]):04d}', text)
            record['artifacts'].append(stage)

