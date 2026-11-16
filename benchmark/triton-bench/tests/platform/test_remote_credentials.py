# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Remote startup must reject unsafe credentials before creating service state."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from triton_eval.remote_credentials import ERROR, validate_credentials


def credential_file(tmp_path):
    path = tmp_path / "oauth.json"
    path.write_text(
        json.dumps(dict(access_token="synthetic-access", refresh_token="synthetic-refresh"))
    )
    path.chmod(0o600)
    return path


def test_external_private_credentials_accepted(tmp_path):
    path = credential_file(tmp_path)
    assert validate_credentials(path, repository=tmp_path / "repo") == path


@pytest.mark.parametrize(
    "kind",
    ["missing", "public", "malformed", "empty", "whitespace", "symlink", "directory", "repository"],
)
def test_unsafe_credentials_rejected_without_values(tmp_path, kind):
    path = credential_file(tmp_path)
    repository = tmp_path / "repo"
    if kind == "missing":
        path.unlink()
    elif kind == "public":
        path.chmod(0o644)
    elif kind == "malformed":
        path.write_text("synthetic-secret-invalid-json")
    elif kind == "empty":
        path.write_text("{}")
    elif kind == "whitespace":
        path.write_text(
            json.dumps(dict(access_token="synthetic secret", refresh_token="synthetic-refresh"))
        )
    elif kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(path)
        path = link
    elif kind == "directory":
        path = tmp_path
    elif kind == "repository":
        repository = tmp_path
    with pytest.raises(ValueError) as exc:
        validate_credentials(path, repository=repository)
    assert str(exc.value) == ERROR
    assert "synthetic" not in str(exc.value)


def test_cli_fails_before_creating_service_directory(tmp_path):
    config = tmp_path / "remote.json"
    config.write_text(
        json.dumps(
            dict(instance="0" * 32, ssh_host="test-host", credentials=str(tmp_path / "absent"))
        )
    )
    root = tmp_path / "service"
    environment = dict(os.environ)
    environment.pop("CANNLAB_CREDENTIALS_FILE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "triton_eval.submission_http",
            "--root",
            str(root),
            "--remote-config",
            str(config),
            "--acknowledge-isolation",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert ERROR in result.stderr
    assert not root.exists()


def test_explicit_credentials_override_config_and_environment(tmp_path, monkeypatch):
    from triton_eval import submission_http

    path = credential_file(tmp_path)
    config = tmp_path / "remote.json"
    config.write_text(
        json.dumps(dict(instance="0" * 32, ssh_host="test-host", credentials="/absent-config"))
    )
    monkeypatch.setenv("CANNLAB_CREDENTIALS_FILE", "/absent-env")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server",
            "--remote-config",
            str(config),
            "--credentials-file",
            str(path),
            "--acknowledge-isolation",
        ],
    )

    class ReachedService(Exception):
        pass

    def service(*args, **kwargs):
        assert kwargs["remote_config"].credentials == path
        raise ReachedService

    monkeypatch.setattr(submission_http, "EvaluationService", service)
    with pytest.raises(ReachedService):
        submission_http.main()
