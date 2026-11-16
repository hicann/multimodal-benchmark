# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Local-only credential admission. Never return, log or serialize token values."""

from __future__ import annotations


import json
import os
from pathlib import Path
import stat

from triton_eval.console import emit

ERROR = (
    "远端 Worker 启动失败：必须提供仓库外、归当前用户所有且权限为 600 的 CANNLab 凭据文件；"
    "JSON 必须包含非空 access_token 和 refresh_token。使用 --credentials-file、"
    "CANNLAB_CREDENTIALS_FILE 或远端配置的 credentials 指定路径。"
)


def _repository_root(repository):
    if repository is not None:
        return repository.resolve()
    source = Path(__file__).resolve()
    from triton_eval import PROJECT_ROOT

    return next((parent for parent in source.parents if (parent / ".git").exists()), PROJECT_ROOT).resolve()


def _require_private_file(info):
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(ERROR)
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        raise ValueError(ERROR)
    if info.st_size > 65536:
        raise ValueError(ERROR)


def _require_token_pair(value):
    if not isinstance(value, dict):
        raise ValueError(ERROR)
    for key in ("access_token", "refresh_token"):
        token = value.get(key)
        if not isinstance(token, str) or not token:
            raise ValueError(ERROR)
        if any(character.isspace() for character in token):
            raise ValueError(ERROR)


def validate_credentials(path: Path, *, repository: Path | None = None) -> Path:
    """Validate local ownership, containment and token shape without any cloud request."""
    repository = _repository_root(repository)
    candidate = Path(path).expanduser().absolute()
    try:
        if candidate.resolve().is_relative_to(repository):
            raise ValueError(ERROR)
        fd = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, mode=0o600)
        with os.fdopen(fd, "rb") as stream:
            _require_private_file(os.fstat(stream.fileno()))
            value = json.loads(stream.read(65537))
        _require_token_pair(value)
    except (OSError, ValueError, TypeError):
        raise ValueError(ERROR) from None
    return candidate


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=os.environ.get("CANNLAB_CREDENTIALS_FILE") or Path.home() / "cann-lab-http-retry/.credentials.json",
    )
    args = parser.parse_args()
    try:
        validate_credentials(args.credentials_file)
    except ValueError:
        parser.error(ERROR)
    emit("CANNLab 外部凭据本地校验通过；未执行云请求。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
