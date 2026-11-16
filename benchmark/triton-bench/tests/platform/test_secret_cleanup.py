# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Security regressions; synthetic values are deliberately assembled in memory."""

import importlib.util
from pathlib import Path
import subprocess
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("scan_secrets", ROOT / "scripts/scan_secrets.py")
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


def program(name):
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(name)
    return str(Path(path).resolve())


def test_secret_patterns_and_safe_examples():
    value = b"unittest" + b"0123456789abcdefghijklmnop"
    assert scanner.findings(b"https://oauth2:" + value + b"@example.test/repo")
    assert scanner.findings(b"https://" + value + b"@example.test/repo")
    assert scanner.findings(b"${GITCODE_TOKEN:-" + value + b"}")
    assert scanner.findings(b'GITCODE_TOKEN="' + value + b'"')
    assert scanner.findings(b'{"access_token":"' + value + b'"}')
    assert not scanner.findings(b"https://example.test/repo ${GITCODE_TOKEN:?required} git@example.test:team/repo")


@pytest.mark.parametrize(
    "remote,expected",
    [
        ("https://example.test/repo", 0),
        ("git@example.test:team/repo", 0),
        ("https://user:dummy@example.test/repo", 1),
        ("https://example.test/repo?access_token=dummy", 1),
    ],
)
def test_shell_remote_guard_does_not_print_credentials(remote, expected, tmp_path):
    command = 'source "$1"; require_safe_git_remote "$2"'
    result = subprocess.run(
        [program("bash"), "-c", command, "test", str(ROOT / "scripts/git-auth.sh"), remote],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected
    assert "dummy" not in result.stdout + result.stderr


def test_persistent_runner_does_not_reset_or_upload_periodically():
    text = (ROOT / "scripts/ascend-ops-verify.sh").read_text()
    for dangerous in ("git reset", "git clean", "git clone", "git push", "upload-heartbeat", "GITCODE_TOKEN"):
        assert dangerous not in text
    assert not (ROOT / "scripts/upload-heartbeat.sh").exists()
    assert not (ROOT / "scripts/save-compiler-wheel.sh").exists()


def test_repository_source_secret_guard():
    result = subprocess.run([sys.executable, "scripts/scan_secrets.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("script", ["upload-verify-record.sh", "upload-verify-results.sh"])
def test_upload_rejects_embedded_credentials_in_target_repository_before_side_effects(tmp_path, script):
    import os

    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    for name in (script, "git-auth.sh"):
        shutil.copyfile(ROOT / "scripts" / name, project / "scripts" / name)
    subprocess.run([program("git"), "init", str(project)], check=True, capture_output=True)
    subprocess.run(
        [program("git"), "-C", str(project), "remote", "add", "origin", "https://user:dummy@example.test/repo"],
        check=True,
    )
    environment = dict(os.environ, WORKDIR=str(project))
    environment.pop("TRITON_EVAL_PUSH_URL", None)
    result = subprocess.run(
        [program("bash"), str(project / "scripts" / script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "拒绝" in result.stderr
    assert "dummy" not in result.stdout + result.stderr
    assert not (project / "verify-records").exists()


def test_staged_guard_reads_index_even_if_worktree_was_cleaned(tmp_path):

    (tmp_path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/scan_secrets.py", tmp_path / "scripts/scan_secrets.py")
    subprocess.run([program("git"), "init", str(tmp_path)], check=True, capture_output=True)
    secret = "synthetic" + "0123456789abcdefghijklmnop"
    sample = tmp_path / "sample.sh"
    sample.write_text('GITCODE_TOKEN="' + secret + '"\n')
    subprocess.run([program("git"), "-C", str(tmp_path), "add", "sample.sh"], check=True)
    sample.write_text("# cleaned working tree\n")
    result = subprocess.run(
        [sys.executable, "scripts/scan_secrets.py", "--staged"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 1 and "credential-assignment" in result.stdout
    assert secret not in result.stdout + result.stderr
