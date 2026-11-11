# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""用本地裸仓库验证滚动上传、失败留档及工作区隔离。"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "publish_ascend_results", Path(__file__).resolve().parents[2] / "scripts/publish_ascend_results.py")
UPLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPLOAD)


def test_publisher_cli_bootstraps_source_without_site_packages_or_pythonpath(tmp_path):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-S', str(Path(UPLOAD.__file__)), '--help'],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'record' in result.stdout


def command(directory, *arguments):
    executable = shutil.which('git')
    if executable is None:
        raise FileNotFoundError('Git executable is required for this regression')
    return subprocess.check_output([str(Path(executable).resolve()), "-C", str(directory), *arguments],
                                   text=True).strip()


@pytest.fixture
def repositories(tmp_path, monkeypatch):
    monkeypatch.delenv("TRITON_EVAL_PUSH_URL", raising=False)
    monkeypatch.delenv("GITCODE_TOKEN", raising=False)
    remote = tmp_path / "remote.git"
    remote.mkdir()
    command(remote, "init", "--bare", "-q")
    repository = tmp_path / "project"
    repository.mkdir()
    command(repository, "init", "-q")
    command(repository, "remote", "add", "origin", str(remote))
    (repository / "user-change.txt").write_text("staged user work")
    command(repository, "add", "user-change.txt")
    return repository, remote


def record(repository, stamp, returncode=0):
    root = repository / "runs" / stamp
    root.mkdir(parents=True)
    (root / "execution.json").write_text(json.dumps({"returncode": returncode}))
    (root / "terminal.log").write_text("startup output\n")
    evidence = root / "evidence"
    evidence.mkdir()
    (evidence / "run.log").write_text("failure details\n")
    (evidence / "eval.sqlite").write_bytes(b"database excluded")
    return root


def test_rolling_upload_keeps_three_snapshots_without_parent_history(repositories):
    repository, remote = repositories
    original_index = command(repository, "diff", "--cached")
    for number in range(5):
        UPLOAD.publish(repository, record(repository, f"20261023-{number:06d}"))
    index = json.loads(command(remote, "show", "npu-results:rolling-results.json"))
    assert index["latest"] == "20261023-000004"
    assert index["records"] == ["20261023-000002", "20261023-000003", "20261023-000004"]
    assert command(remote, "rev-list", "--count", "npu-results") == "1"
    files = command(remote, "ls-tree", "-r", "--name-only", "npu-results")
    assert "eval.sqlite" not in files
    assert "user-change.txt" not in files
    assert "failure details" in command(remote, "show", "npu-results:records/20261023-000004/evidence/run.log")
    assert command(repository, "diff", "--cached") == original_index
    assert sorted(path.name for path in (repository / "runs").iterdir()) == index["records"]


def test_failed_evaluation_is_uploaded_and_unsent_records_are_preserved(repositories):
    repository, remote = repositories
    pending = record(repository, "000000-pending", 1)
    failed = record(repository, "20261023-failed", 1)
    UPLOAD.publish(repository, failed, keep=1)
    payload = json.loads(command(remote, "show", "npu-results:records/20261023-failed/execution.json"))
    assert payload["returncode"] == 1
    assert pending.exists()
    assert (failed / ".uploaded").is_file()


def test_push_failure_does_not_prune_local_results(repositories, monkeypatch):
    repository, _ = repositories
    previous = record(repository, "20261023-000000")
    UPLOAD.publish(repository, previous)
    current = record(repository, "20261023-000001")
    original = UPLOAD.git

    def fail_push(directory, *arguments, **kwargs):
        if arguments[0] == "push":
            raise RuntimeError("simulated network failure")
        return original(directory, *arguments, **kwargs)

    monkeypatch.setattr(UPLOAD, "git", fail_push)
    with pytest.raises(RuntimeError, match="network failure"):
        UPLOAD.publish(repository, current, keep=1)
    assert previous.exists() and current.exists()
    assert not (current / ".uploaded").exists()


def test_concurrent_update_is_not_overwritten(repositories, monkeypatch):
    repository, remote = repositories
    UPLOAD.publish(repository, record(repository, "20261023-000000"))
    losing = record(repository, "20261023-000001")
    winning = record(repository, "20261023-000002")
    original = UPLOAD.git

    def update_before_push(directory, *arguments, **kwargs):
        if arguments[0] == "push":
            monkeypatch.setattr(UPLOAD, "git", original)
            UPLOAD.publish(repository, winning)
        return original(directory, *arguments, **kwargs)

    monkeypatch.setattr(UPLOAD, "git", update_before_push)
    with pytest.raises(RuntimeError):
        UPLOAD.publish(repository, losing)
    latest = json.loads(command(remote, "show", "npu-results:rolling-results.json"))
    assert latest["latest"] == winning.name
    assert losing.exists() and not (losing / ".uploaded").exists()


def test_redacts_credentials_before_publishing(repositories, monkeypatch):
    repository, remote = repositories
    monkeypatch.setenv("GITCODE_TOKEN", "private-test-token")
    current = record(repository, "20261023-redacted")
    (current / "terminal.log").write_text("https://user:password@example.test/repo private-test-token")
    UPLOAD.publish(repository, current)
    uploaded = command(remote, "show", "npu-results:records/20261023-redacted/terminal.log")
    assert "password" not in uploaded and "private-test-token" not in uploaded


def test_rejects_oversized_evidence_without_deleting_source(repositories, monkeypatch):
    repository, remote = repositories
    current = record(repository, "20261023-large")
    monkeypatch.setattr(UPLOAD, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="50 MiB"):
        UPLOAD.publish(repository, current)
    assert current.exists()
    assert not command(remote, "for-each-ref", "refs/heads/npu-results")


def test_rejects_symlink_evidence(repositories, tmp_path):
    repository, _ = repositories
    current = record(repository, "20261023-symlink")
    external = tmp_path / "external.txt"
    external.write_text("unrelated data")
    (current / "external.log").symlink_to(external)
    with pytest.raises(ValueError, match="符号链接"):
        UPLOAD.publish(repository, current)


def test_legacy_credentials_only_apply_to_matching_repository(tmp_path, monkeypatch):
    monkeypatch.delenv("TRITON_EVAL_PUSH_URL", raising=False)
    monkeypatch.delenv("GITCODE_TOKEN", raising=False)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "ascend-ops-verify.sh").write_text('REPO_URL="https://user:secret@gitcode.com/team/project.git"\n')
    monkeypatch.setattr(UPLOAD, "git", lambda *args, **kwargs: "https://gitcode.com/team/project.git")
    assert "user:secret@" in UPLOAD.push_url(tmp_path)
    monkeypatch.setattr(UPLOAD, "git", lambda *args, **kwargs: "https://gitcode.com/another/project.git")
    assert "secret" not in UPLOAD.push_url(tmp_path)


@pytest.mark.parametrize("environment_rc,evaluation_rc,upload_rc,expected", [
    (7, 0, 0, 7), (0, 1, 0, 1), (0, 0, 2, 2), (0, 0, 0, 0)])
def test_shell_entry_uploads_startup_and_evaluation_failures(
        tmp_path, environment_rc, evaluation_rc, upload_rc, expected):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[2] / "scripts/ascend-suite-verify.sh"
    (scripts / source.name).write_text(source.read_text())
    (scripts / source.name).chmod(0o755)
    (scripts / "env-bootstrap.sh").write_text(
        "echo environment-started\npython() { python3 \"$@\"; }\n"
        + (f"exit {environment_rc}\n" if environment_rc else ""))
    (scripts / "run_ascend_suite.py").write_text(f"raise SystemExit({evaluation_rc})\n")
    (scripts / "publish_ascend_results.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "assert (root / 'terminal.log').exists()\n"
        "status = json.loads((root / 'execution.json').read_text())\n"
        "assert status['state'] == 'finished'\n"
        "Path('uploaded.json').write_text(json.dumps(status))\n"
        f"raise SystemExit({upload_rc})\n")
    command(tmp_path, "init", "-q")
    command(tmp_path, "add", "scripts")
    command(tmp_path, "-c", "user.name=test", "-c", "user.email=test@local",
            "commit", "-qm", "fixture")
    result = subprocess.run([str(scripts / source.name)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, result.stdout + result.stderr
    status = json.loads((tmp_path / "uploaded.json").read_text())
    assert status["returncode"] == (environment_rc or evaluation_rc)
    assert status["code_commit"] == command(tmp_path, "rev-parse", "HEAD")


def test_publication_preserves_json_provenance_and_frozen_hashes(repositories):
    import hashlib

    repository, remote = repositories
    current = record(repository, "20261023-frozen")
    frozen = current / "evidence/frozen-official50"
    frozen.mkdir(parents=True)
    # Keep CRLF and a non-ASCII field byte-for-byte as part of the fingerprint.
    raw = json.dumps({"repository": "https://github.com/triton-lang/triton-ascend",
                      "url": "https://example.test/source?revision=abc", "说明": "来源"},
                     ensure_ascii=False, indent=2).replace("\n", "\r\n").encode() + b"\r\n"
    (frozen / "official-catalog.json").write_bytes(raw)
    (frozen / "evaluation-assets.json").write_text(json.dumps({"format": 2, "files": {
        "@corpus/official-catalog.json": hashlib.sha256(raw).hexdigest()}}))
    UPLOAD.publish(repository, current)
    executable = shutil.which('git')
    if executable is None:
        raise FileNotFoundError('Git executable is required for this regression')
    uploaded = subprocess.run([str(Path(executable).resolve()), "-C", str(remote), "show",
                               "npu-results:records/20261023-frozen/evidence/frozen-official50/official-catalog.json"],
                              check=True, capture_output=True).stdout
    assert uploaded == raw
    assert json.loads(uploaded)["repository"] == "https://github.com/triton-lang/triton-ascend"


def test_redaction_keeps_json_valid_and_removes_url_authentication(monkeypatch):
    monkeypatch.setenv("GITCODE_TOKEN", "private-test-token")
    text = json.dumps({"url": "https://oauth2:password@example.test/repo?token=private-test-token&revision=abc",
                       "source": "https://github.com/triton-lang/triton-ascend"})
    redacted = UPLOAD.redact(text)
    assert "password" not in redacted and "private-test-token" not in redacted
    payload = json.loads(redacted)
    assert payload["source"] == "https://github.com/triton-lang/triton-ascend"
    assert "revision=abc" in payload["url"]


def test_publication_rejects_redaction_of_fingerprinted_assets(repositories, monkeypatch):
    import hashlib

    repository, remote = repositories
    current = record(repository, "20261023-private-asset")
    frozen = current / "evidence/frozen-official50"
    frozen.mkdir(parents=True)
    raw = b'{"url":"https://user:password@example.test/source"}'
    (frozen / "official-catalog.json").write_bytes(raw)
    (frozen / "evaluation-assets.json").write_text(json.dumps({"format": 2, "files": {
        "@corpus/official-catalog.json": hashlib.sha256(raw).hexdigest()}}))
    with pytest.raises(ValueError, match="冻结资产指纹"):
        UPLOAD.publish(repository, current)
    assert current.exists()
    assert not command(remote, "for-each-ref", "refs/heads/npu-results")
