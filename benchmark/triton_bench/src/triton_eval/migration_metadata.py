# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Losslessly store authenticated predecessor metadata without recursive escaping."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

MAX_METADATA_BYTES = 512 * 1024 * 1024


def encode_metadata(content: bytes) -> dict:
    if len(content) > MAX_METADATA_BYTES:
        raise ValueError("asset migration metadata exceeds decoded size limit")
    return {"encoding": "zlib-base64", "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "data": base64.b64encode(zlib.compress(content, 9)).decode("ascii")}


def metadata_path(value: dict, corpus: Path) -> Path:
    identity = value.get('sha256')
    if set(value) != {'encoding', 'size', 'sha256'} or value['encoding'] != 'zlib-base64-file':
        raise ValueError('invalid detached asset migration metadata')
    if not valid_digest(identity):
        raise ValueError('invalid detached asset migration metadata')
    directory = corpus / 'evaluation-assets-metadata'
    path = directory / (identity + '.json')
    if directory.is_symlink() or path.is_symlink():
        raise ValueError('symlink in asset migration metadata')
    return path


def detached_metadata(content: bytes) -> tuple[dict, bytes]:
    encoded = encode_metadata(content)
    reference = {'encoding': 'zlib-base64-file', 'size': encoded['size'], 'sha256': encoded['sha256']}
    return reference, (json.dumps(encoded, indent=2) + '\n').encode()


def valid_digest(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in '0123456789abcdef' for character in value)


def valid_size(value) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return 0 <= value <= MAX_METADATA_BYTES


def read_detached(value: dict, corpus: Path | None) -> dict:
    if corpus is None:
        raise ValueError('detached metadata requires corpus directory')
    path = metadata_path(value, corpus)
    if path.stat().st_size > MAX_METADATA_BYTES * 2:
        raise ValueError('asset migration storage exceeds size limit')
    envelope = json.loads(path.read_bytes())
    if not isinstance(envelope, dict) or envelope.get('encoding') != 'zlib-base64':
        raise ValueError('detached asset migration identity differs')
    if not valid_size(value['size']):
        raise ValueError('detached asset migration identity differs')
    if envelope.get('size') != value['size'] or envelope.get('sha256') != value['sha256']:
        raise ValueError('detached asset migration identity differs')
    return envelope


def validate_envelope(value) -> None:
    if not isinstance(value, dict) or set(value) != {'encoding', 'size', 'sha256', 'data'}:
        raise ValueError("invalid compressed asset migration metadata")
    if value['encoding'] != 'zlib-base64' or not valid_size(value['size']):
        raise ValueError("invalid compressed asset migration metadata")
    if not valid_digest(value['sha256']):
        raise ValueError("invalid compressed asset migration metadata")
    if not isinstance(value['data'], str) or len(value['data']) > MAX_METADATA_BYTES * 2:
        raise ValueError("invalid compressed asset migration metadata")


def decompress_metadata(value: dict) -> bytes:
    validate_envelope(value)
    try:
        compressed = base64.b64decode(value['data'], validate=True)
        decoder = zlib.decompressobj()
        content = decoder.decompress(compressed, value['size'] + 1)
    except (ValueError, zlib.error) as error:
        raise ValueError("invalid compressed asset migration metadata") from error
    if len(content) != value['size'] or not decoder.eof:
        raise ValueError("compressed asset migration metadata integrity differs")
    if decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError("compressed asset migration metadata integrity differs")
    if hashlib.sha256(content).hexdigest() != value['sha256']:
        raise ValueError("compressed asset migration metadata integrity differs")
    return content


def decode_metadata(value: str | dict, corpus: Path | None = None) -> bytes:
    if isinstance(value, dict) and value.get('encoding') == 'zlib-base64-file':
        value = read_detached(value, corpus)
    if isinstance(value, str):
        content = value.encode('utf-8')
        if len(content) > MAX_METADATA_BYTES:
            raise ValueError("asset migration metadata exceeds decoded size limit")
        return content
    return decompress_metadata(value)
