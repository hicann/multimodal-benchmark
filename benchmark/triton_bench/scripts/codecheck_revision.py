# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Publish explicit local asset revisions without changing case manifests or old indexes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def index_hash(index):
    payload = {key: value for key, value in index.items() if key != 'assets_hash'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_assets(platform):
    specification = importlib.util.spec_from_file_location(
        'revision_assets', platform / 'src/triton_eval/corpus_assets.py')
    assets = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(assets)
    return assets


def load_metadata_codec(platform):
    specification = importlib.util.spec_from_file_location(
        'revision_metadata', platform / 'src/triton_eval/migration_metadata.py')
    metadata_codec = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(metadata_codec)
    return metadata_codec


def read_predecessor(path, assets):
    corpus = path.parent
    migration = corpus / 'evaluation-assets-migration.json'
    if not migration.is_file():
        raise ValueError('No authenticated predecessor migration: ' + str(corpus))
    previous_bytes, metadata_bytes = path.read_bytes(), migration.read_bytes()
    previous, metadata = json.loads(previous_bytes), json.loads(metadata_bytes)
    manifest = json.loads((corpus / 'manifest.json').read_bytes())
    if index_hash(previous) != previous['assets_hash']:
        raise ValueError('Predecessor index digest differs')
    if (previous['manifest_hash'] != manifest['manifest_hash']
            or metadata['manifest_hash'] != manifest['manifest_hash']):
        raise ValueError('Case manifest identity differs')
    expected = previous['files'].get('@corpus/evaluation-assets-migration.json')
    if hashlib.sha256(metadata_bytes).hexdigest() != expected:
        raise ValueError('Predecessor migration digest differs')
    assets.revision_history_paths(corpus, metadata, manifest['manifest_hash'])
    return previous_bytes, metadata_bytes, previous, metadata


def plan_metadata(corpus, metadata, predecessor, metadata_codec, planned):
    entries = [(entry['assets_hash'], metadata_codec.decode_metadata(entry['migration_metadata'], corpus))
               for entry in metadata.get('previous_revisions', [])]
    entries.append(predecessor)
    revisions = []
    for identity, content in entries:
        reference, payload = metadata_codec.detached_metadata(content)
        stored = metadata_codec.metadata_path(reference, corpus)
        before = stored.read_bytes() if stored.exists() else None
        if before is not None and before != payload:
            raise ValueError('Historical metadata storage differs')
        if before is None:
            planned[stored] = (None, payload)
        revisions.append({'assets_hash': identity, 'migration_metadata': reference})
    return revisions


def plan_corpus(path, revision, assets, metadata_codec, planned, reason):
    previous_bytes, metadata_bytes, previous, metadata = read_predecessor(path, assets)
    corpus = path.parent
    if metadata.get('revision') == revision:
        assets.verify_assets(corpus, previous['manifest_hash'])
        return None
    archive = corpus / 'evaluation-assets-history' / (previous['assets_hash'] + '.json')
    if archive.exists() and archive.read_bytes() != previous_bytes:
        raise ValueError('Archive already exists with different bytes')
    updated = dict(metadata)
    predecessor = (previous['assets_hash'], metadata_bytes)
    updated['previous_revisions'] = plan_metadata(corpus, metadata, predecessor, metadata_codec, planned)
    updated['metadata_storage'] = 'detached-zlib-base64-v1'
    updated.update(revision=revision, hardware_validation='NOT_RUN',
                   reason=reason)
    planned[path] = (previous_bytes, None)
    planned[corpus / 'evaluation-assets-migration.json'] = (metadata_bytes, encoded(updated))
    planned[archive] = (archive.read_bytes() if archive.exists() else None, previous_bytes)
    return {'corpus': corpus.name, 'previous_assets_hash': previous['assets_hash'],
            'origin_assets_hash': metadata['origin_assets_hash'],
            'manifest_hash': previous['manifest_hash'], 'hardware_validation': 'NOT_RUN'}


def rollback(planned, written):
    for path in reversed(written):
        before = planned[path][0]
        if before is None:
            path.unlink()
        else:
            path.write_bytes(before)


def write_plan(platform, planned, records, assets):
    written = []
    try:
        for path, (before, after) in planned.items():
            if after is None:
                continue
            if (path.read_bytes() if path.exists() else None) != before:
                raise ValueError('Concurrent asset edit detected')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(after)
            written.append(path)
        for record in records:
            corpus = platform / 'corpus' / record['corpus']
            path = corpus / 'evaluation-assets.json'
            current = assets.snapshot(corpus, record['manifest_hash'])
            if path.read_bytes() != planned[path][0]:
                raise ValueError('Concurrent active-index edit detected')
            path.write_bytes(encoded(current))
            written.append(path)
            assets.verify_origin_assets(corpus, record['origin_assets_hash'])
            record['current_assets_hash'] = current['assets_hash']
    except BaseException:
        rollback(planned, written)
        raise


def revise(platform, revision, *, reason="Context-reviewed quality repairs; case identities and scoring retained."):
    assets, metadata_codec = load_assets(platform), load_metadata_codec(platform)
    planned, records = {}, []
    for path in sorted((platform / 'corpus').glob('*/evaluation-assets.json')):
        record = plan_corpus(path, revision, assets, metadata_codec, planned, reason)
        if record is not None:
            records.append(record)
    write_plan(platform, planned, records, assets)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reason', default='Context-reviewed quality repairs; case identities and scoring retained.')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Revision receipt must be a new file')
    records = revise(args.platform.resolve(), args.revision, reason=args.reason)
    args.out.write_bytes(encoded(records))
    logging.info('%s', json.dumps({'revised': len(records), 'hardware_validation': 'NOT_RUN'}))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    raise SystemExit(main())
