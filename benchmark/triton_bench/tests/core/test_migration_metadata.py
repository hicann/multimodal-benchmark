# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Storage compaction retains authenticated history and bounded decoding."""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import subprocess
import zlib
from pathlib import Path

import pytest

from triton_eval import corpus_assets
from triton_eval import migration_metadata as codec

ROOT = Path(__file__).absolute().parents[2]


@pytest.mark.parametrize('content', [b'', b'plain\n', '中文\n'.encode(), b'\\"' * 100000])
def test_metadata_roundtrip_and_legacy_compatibility(content):
    assert codec.decode_metadata(codec.encode_metadata(content)) == content
    assert codec.decode_metadata(content.decode()) == content


@pytest.mark.parametrize('change', ['encoding', 'size', 'boolean-size', 'negative-size', 'digest',
                                   'extra', 'base64', 'truncated', 'tail', 'stream', 'bound'])
def test_compressed_metadata_rejects_invalid_or_forged_content(change):
    value = codec.encode_metadata(b'authenticated predecessor\n')
    if change == 'encoding':
        value['encoding'] = 'gzip'
    elif change == 'size':
        value['size'] -= 1
    elif change == 'boolean-size':
        value['size'] = True
    elif change == 'negative-size':
        value['size'] = -1
    elif change == 'digest':
        value['sha256'] = '0' * 64
    elif change == 'extra':
        value['unexpected'] = 1
    elif change == 'base64':
        value['data'] = '$not-base64'
    elif change == 'bound':
        value['size'] = codec.MAX_METADATA_BYTES + 1
    else:
        compressed = base64.b64decode(value['data'])
        if change == 'truncated':
            compressed = compressed[:-1]
        elif change == 'tail':
            compressed += b'trailing-data'
        else:
            compressed += zlib.compress(b'second-stream')
        value['data'] = base64.b64encode(compressed).decode()
    with pytest.raises(ValueError):
        codec.decode_metadata(value)


def test_metadata_decode_enforces_output_bound(monkeypatch):
    value = codec.encode_metadata(b'large' * 10000)
    monkeypatch.setattr(codec, 'MAX_METADATA_BYTES', 100)
    with pytest.raises(ValueError):
        codec.decode_metadata(value)
    with pytest.raises(ValueError):
        codec.decode_metadata('large' * 10000)
    with pytest.raises(ValueError):
        codec.encode_metadata(b'large' * 10000)


def test_detached_metadata_roundtrip_requires_explicit_corpus(tmp_path):
    content = b'preserve exact predecessor bytes\n'
    reference, payload = codec.detached_metadata(content)
    path = codec.metadata_path(reference, tmp_path)
    path.parent.mkdir()
    path.write_bytes(payload)
    assert codec.decode_metadata(reference, tmp_path) == content
    with pytest.raises(ValueError, match='requires corpus'):
        codec.decode_metadata(reference)


@pytest.mark.parametrize('change', ['missing', 'identity', 'size', 'boolean-size', 'path',
                                   'digest', 'payload', 'directory-symlink', 'file-symlink'])
def test_detached_metadata_rejects_missing_corrupt_and_escaping_storage(tmp_path, change):
    reference, payload = codec.detached_metadata(b'authenticated bytes')
    path = codec.metadata_path(reference, tmp_path)
    path.parent.mkdir()
    path.write_bytes(payload)
    if change == 'missing':
        path.unlink()
    elif change == 'identity':
        reference['sha256'] = '0' * 64
    elif change == 'size':
        reference['size'] += 1
    elif change == 'boolean-size':
        reference['size'] = True
    elif change == 'path':
        reference['sha256'] = '../outside'
    elif change in {'digest', 'payload'}:
        envelope = json.loads(payload)
        envelope['sha256' if change == 'digest' else 'data'] = '0' * 64
        path.write_text(json.dumps(envelope))
    elif change == 'directory-symlink':
        moved = tmp_path / 'outside'
        path.parent.rename(moved)
        path.parent.symlink_to(moved, target_is_directory=True)
    else:
        moved = tmp_path / 'outside.json'
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises((ValueError, FileNotFoundError)):
        codec.decode_metadata(reference, tmp_path)


def load_script(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_repeated_revisions_preserve_exact_predecessors_without_recursive_escape(tmp_path, monkeypatch):
    source = ROOT / 'corpus/v0.18-scan-certification-candidates'
    corpus = tmp_path / 'corpus/candidate'
    shutil.copytree(source, corpus)
    helper = tmp_path / 'src/triton_eval/migration_metadata.py'
    helper.parent.mkdir(parents=True)
    shutil.copy2(Path(codec.__file__), helper)
    writer = load_script('codecheck_revision')
    monkeypatch.setattr(writer, 'load_assets', lambda platform: corpus_assets)
    historical = {path.name: path.read_bytes() for path in (corpus / 'evaluation-assets-history').glob('*.json')}
    migration = corpus / 'evaluation-assets-migration.json'
    for iteration in range(4):
        before = migration.read_bytes()
        writer.revise(tmp_path, f'size-regression-{iteration}')
        data = json.loads(migration.read_bytes())
        assert codec.decode_metadata(data['previous_revisions'][-1]['migration_metadata'], corpus) == before
        assert migration.stat().st_size < 16 * 1024
        corpus_assets.verify_origin_assets(corpus, data['origin_assets_hash'])
    assert all((corpus / 'evaluation-assets-history' / name).read_bytes() == content
               for name, content in historical.items())
    before = {path.relative_to(corpus): path.read_bytes() for path in corpus.rglob('*') if path.is_file()}
    assert writer.revise(tmp_path, 'size-regression-3') == []
    assert before == {path.relative_to(corpus): path.read_bytes() for path in corpus.rglob('*') if path.is_file()}


def test_revision_failure_rolls_back_new_metadata_and_indexes(tmp_path, monkeypatch):
    source = ROOT / 'corpus/v0.18-scan-certification-candidates'
    corpus = tmp_path / 'corpus/candidate'
    shutil.copytree(source, corpus)
    helper = tmp_path / 'src/triton_eval/migration_metadata.py'
    helper.parent.mkdir(parents=True)
    shutil.copy2(Path(codec.__file__), helper)
    writer = load_script('codecheck_revision')
    monkeypatch.setattr(writer, 'load_assets', lambda platform: corpus_assets)
    before = {path.relative_to(corpus): path.read_bytes() for path in corpus.rglob('*') if path.is_file()}

    def refuse_snapshot(*arguments):
        raise ValueError('injected snapshot failure')

    monkeypatch.setattr(corpus_assets, 'snapshot', refuse_snapshot)
    with pytest.raises(ValueError, match='injected snapshot failure'):
        writer.revise(tmp_path, 'rollback-check')
    assert before == {path.relative_to(corpus): path.read_bytes() for path in corpus.rglob('*') if path.is_file()}


def git(repository, *arguments):
    executable = shutil.which('git')
    if executable is None:
        raise FileNotFoundError('Git executable is required for size-gate tests')
    return subprocess.run([str(Path(executable).resolve()), '-C', str(repository), *arguments],
                          capture_output=True, check=True).stdout


def commit(repository):
    git(repository, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-qm', 'size-test')
    return git(repository, 'rev-parse', 'HEAD').decode().strip()


def test_size_gate_checks_index_and_all_unpublished_history(tmp_path):
    checker = load_script('check_git_blob_sizes')
    git(tmp_path, 'init', '-q')
    path = tmp_path / 'history.json'
    path.write_text('small')
    git(tmp_path, 'add', '.')
    baseline = commit(tmp_path)
    path.write_bytes(b'large' * 100)
    git(tmp_path, 'add', '.')
    assert checker.oversized_blobs(tmp_path, baseline, 100) == [(500, 'history.json')]
    commit(tmp_path)
    path.write_text('small again')
    git(tmp_path, 'add', '.')
    commit(tmp_path)
    assert checker.oversized_blobs(tmp_path, None, 100) == []
    assert checker.oversized_blobs(tmp_path, baseline, 100) == [(500, 'history.json')]


def test_size_gate_fails_for_unknown_base(tmp_path):
    checker = load_script('check_git_blob_sizes')
    git(tmp_path, 'init', '-q')
    with pytest.raises(subprocess.CalledProcessError):
        checker.oversized_blobs(tmp_path, 'does-not-exist', 100)


@pytest.mark.parametrize('size', [True, False, -1, 0.5, '1', None])
def test_metadata_refactor_keeps_strict_size_rejection(size):
    envelope = codec.encode_metadata(b'content')
    envelope['size'] = size
    with pytest.raises(ValueError):
        codec.decode_metadata(envelope)


@pytest.mark.parametrize('data', [None, 12, [], {}])
def test_metadata_refactor_rejects_non_text_compressed_payload(data):
    envelope = codec.encode_metadata(b'content')
    envelope['data'] = data
    with pytest.raises(ValueError):
        codec.decode_metadata(envelope)


def test_size_gate_refuses_missing_git_instead_of_launching_a_bare_command(tmp_path, monkeypatch):
    checker = load_script('check_git_blob_sizes')
    monkeypatch.setattr(checker.shutil, 'which', lambda name: None)
    with pytest.raises(FileNotFoundError, match='Git executable'):
        checker.git(tmp_path, 'status')


def test_size_gate_resolves_git_to_an_absolute_executable(tmp_path, monkeypatch):
    from types import SimpleNamespace

    checker = load_script('check_git_blob_sizes')
    command = []

    def capture(arguments, **options):
        command.extend(arguments)
        return SimpleNamespace(stdout=b'ok')

    monkeypatch.setattr(checker.subprocess, 'run', capture)
    assert checker.git(tmp_path, 'status') == b'ok'
    assert Path(command[0]).is_absolute()
    assert command[1:] == ['-C', str(tmp_path), 'status']
