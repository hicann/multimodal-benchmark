# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Synthetic dump diagnostics never establish NPU compilation or execution credit."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_eval.compilation_evidence import CompilationCapture
from triton_eval.coverage_evidence import sha
from test_evidence_gaps import ASCEND_VECTOR


SPEC = importlib.util.spec_from_file_location(
    "diagnose_backend_dump", Path(__file__).resolve().parents[2] / "scripts/diagnose_backend_dump.py"
)
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


def test_archived_commands_keep_only_the_exact_reviewed_semantic_configuration():
    options = sorted(DIAG.SEMANTIC_OPTIONS)
    command = [
        "bishengir-compile",
        "/old/kernel.ttadapter.mlir",
        *options,
        "-o",
        "/old/kernel",
        "--mlir-print-ir-after-all",
        "--mlir-print-ir-module-scope",
        "--bishengir-print-ir-after=canonicalize",
        "--mlir-print-ir-tree-dir=/old/tree",
    ]
    assert DIAG.semantic_options(command) == options
    assert DIAG.semantic_options(command[:-4] + ["--bishengir-print-ir-after", "canonicalize"]) == options
    assert DIAG.semantic_options(command[:2] + options + ["-o=/old/kernel"]) == options
    for changed in (
        command + ["--hivmc-args=unreviewed"],
        command + ["--load=/untrusted"],
        command + ["other.mlir"],
        command + ["-o", "other"],
        command + ["--enable-hivm-compile=false"],
        command + ["--target=Ascend910B3"],
        command[:-6],
    ):
        with pytest.raises(ValueError):
            DIAG.semantic_options(changed)


def synthetic_observation(launches, case, *_):
    try:
        launch = launches[case.code]
    except KeyError as exc:
        raise AssertionError(f"Missing synthetic compilation for {case.code}") from exc
    return dict(launches=[launch])


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "old"
    root.mkdir()
    (root / "coverage-validation.json").write_text("{}")
    data = dict(
        root=root,
        report=dict(manifest_hash=DIAG.MANIFEST, selected_operator_validation_passed=True, evidence_sha256={}),
        summary=dict(selected_operators=["Neg", "Matmul"]),
        cases={},
        rows={},
    )
    launches = {}
    for operator, atom in DIAG.TARGETS.items():
        code = "synthetic-" + operator
        case = SimpleNamespace(code=code, operator=operator, feature_atoms=[atom])
        data["cases"][code] = case
        data["rows"][code] = dict(run_id=1)
        path = root / operator / "run-1/compilation" / code / "observation.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        writer = CompilationCapture(None, path.parent, {})
        entries = [writer.artifact("ttadapter", ASCEND_VECTOR), writer.artifact("npubin", b"synthetic-binary")]
        entry = writer.artifact("source", "def _witness_neg():\n    pass\n")
        launches[code] = dict(
            name="_witness_neg",
            source=entry,
            artifacts=entries,
            backend_binding=[dict(verified=True)],
            backend_compilations=[
                dict(
                    executable_sha256="a" * 64,
                    command=[
                        "bishengir-compile",
                        "/old/input.mlir",
                        *sorted(DIAG.SEMANTIC_OPTIONS),
                        "-o",
                        "/old/kernel",
                    ],
                )
            ],
        )
        for indexed in [path, *[(path.parent / artifact["path"]).resolve() for artifact in [entry, *entries]]]:
            data["report"]["evidence_sha256"][str(indexed.relative_to(root))] = sha(indexed.read_bytes())
    monkeypatch.setattr(DIAG, "_read", lambda *_: data)
    monkeypatch.setattr(DIAG, "observation", lambda case, *_: synthetic_observation(launches, case))
    return SimpleNamespace(root=root, data=data, launches=launches)


def test_target_selection_uses_indexed_bound_non_cached_compilations_only(source):
    data, items = DIAG.load_inputs(source.root, source.root / "submission")
    assert data is source.data
    assert [(item["operator"], item["atom"]) for item in items] == list(DIAG.TARGETS.items())
    assert all(item["input"] == ASCEND_VECTOR.encode() and item["binary"] == b"synthetic-binary" for item in items)


def test_missing_synthetic_observation_is_a_hard_failure(source):
    with pytest.raises(AssertionError, match="Missing synthetic compilation for missing") as failure:
        DIAG.observation(SimpleNamespace(code="missing"))
    assert isinstance(failure.value.__cause__, KeyError)


@pytest.mark.parametrize("change", ["manifest", "scope", "validation", "observation", "artifact", "binding", "cache"])
def test_untrusted_wrong_scope_or_missing_binding_never_produces_replay_inputs(source, change):
    if change == "manifest":
        source.data["report"]["manifest_hash"] = "foreign"
    elif change == "scope":
        source.data["summary"]["selected_operators"].append("Abs")
    elif change == "validation":
        source.data["report"]["selected_operator_validation_passed"] = False
    elif change == "observation":
        source.data["report"]["evidence_sha256"].pop("Neg/run-1/compilation/synthetic-Neg/observation.json")
    elif change == "artifact":
        source.data["report"]["evidence_sha256"].pop(
            next(key for key in source.data["report"]["evidence_sha256"] if key.endswith(".gz"))
        )
    elif change == "binding":
        source.launches["synthetic-Neg"]["backend_binding"][0]["verified"] = False
    else:
        source.launches["synthetic-Neg"]["backend_cache_reused"] = True
    with pytest.raises(ValueError):
        DIAG.load_inputs(source.root, source.root / "submission")


@pytest.fixture
def tool(tmp_path):
    path = tmp_path / "bishengir-compile"
    help_text = "".join(
        "  " + flag + "=<value>  Synthetic option\n"
        for flag in sorted(DIAG.DUMP_OPTIONS | {"--bishengir-print-ir-before"})
    )
    function_text = ASCEND_VECTOR.removeprefix("module {").removesuffix("}")
    path.write_text(f"""#!{sys.executable}
import pathlib, sys
if '--help' in sys.argv:
    print({help_text!r})
    sys.exit(0)
output = pathlib.Path(sys.argv[sys.argv.index('-o')+1])
output.with_suffix('.o').write_bytes(b'synthetic-binary')
if any(arg.startswith('--bishengir-print-ir-before=hivm-inject-sync,') for arg in sys.argv):
    print('// -----// IR Dump Before InjectSync //----- //', file=sys.stderr)
    print({function_text!r}, file=sys.stderr)
""")
    path.chmod(0o755)
    item = dict(
        operator="Neg",
        atom="FA-PAT-VEC-LOAD",
        case_code="synthetic-Neg",
        kernel="_witness_neg",
        tool_sha256=DIAG.file_digest(path),
        options=sorted(DIAG.SEMANTIC_OPTIONS),
        input=ASCEND_VECTOR.encode(),
        binary=b"synthetic-binary",
    )
    help_text = subprocess.check_output([str(path), "--help"]).decode()
    return SimpleNamespace(path=path, item=item, help=help_text, output=tmp_path / "diagnostic")


def test_vendor_whitelist_can_produce_function_scope_dump_but_never_certifies_it(tool):
    original = subprocess.run
    control = DIAG.replay(tool.item, tool.path, tool.help, "v15-control", tool.output, 10)
    targeted = DIAG.replay(tool.item, tool.path, tool.help, "vendor-whitelist", tool.output, 10)
    assert subprocess.run is original
    assert control["binary_matches_archived"] and not control["dump_observed"]
    assert targeted["binary_matches_archived"] and targeted["dump_observed"] and targeted["ready_for_review"]
    inventory = next(entry for entry in targeted["snapshot_inventory"] if entry["hivm_operations"])
    assert inventory["has_target_function"] and not inventory["has_module"]
    assert inventory["before"] == 1 and inventory["after"] == 0
    assert not targeted["coverage_certified"]
    assert not any("canonicalize" in flag for flag in targeted["flags"])


@pytest.mark.parametrize("change", ["binary", "failure", "timeout", "kernel"])
def test_binary_mismatch_failed_compilation_timeout_or_foreign_function_are_not_ready(tool, change):
    content = tool.path.read_text()
    if change == "binary":
        tool.item["binary"] = b"foreign-binary"
    elif change == "failure":
        content += "\nsys.exit(9)\n"
    elif change == "timeout":
        content += "\nimport time\ntime.sleep(5)\n"
    else:
        tool.item["kernel"] = "foreign_kernel"
    tool.path.write_text(content)
    tool.item["tool_sha256"] = DIAG.file_digest(tool.path)
    outcome = DIAG.replay(
        tool.item, tool.path, tool.help, "vendor-whitelist", tool.output, 0.05 if change == "timeout" else 10
    )
    assert not outcome["ready_for_review"] and not outcome["coverage_certified"]
    if change == "timeout":
        assert outcome["returncode"] == 124
    assert (tool.output / "Neg/vendor-whitelist/replay.json").is_file()


def test_changed_compiler_is_rejected_before_execution(tool):
    tool.item["tool_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        DIAG.replay(tool.item, tool.path, tool.help, "vendor-whitelist", tool.output, 10)
    assert not tool.output.exists()


def test_prepare_only_does_not_require_triton_or_execute_any_compiler(source, monkeypatch, tmp_path):
    monkeypatch.setattr(
        DIAG, "installed_compiler", lambda *_: pytest.fail("CPU preparation must not resolve a compiler")
    )
    report = DIAG.diagnose(source.root, source.root / "submission", tmp_path / "prepared", prepare_only=True)
    assert report["state"] == "prepared" and not report["ready_for_review"]
    assert not report["execution_validation_performed"] and not report["affects_coverage"]
    assert set(report["operators"]) == set(DIAG.TARGETS)


def test_two_operators_two_variants_are_fixed_and_readiness_is_not_certification(tool, monkeypatch, tmp_path):
    items = [dict(tool.item, operator=operator, atom=atom) for operator, atom in DIAG.TARGETS.items()]
    root = tmp_path / "source"
    root.mkdir()
    (root / "coverage-validation.json").write_text("{}")
    data = dict(root=root, report=dict(evidence_sha256={"checked": "hash"}))
    monkeypatch.setattr(DIAG, "load_inputs", lambda *_: (data, items))
    output = tmp_path / "out"
    output.mkdir()
    report = DIAG.diagnose(root, root, output, compiler=tool.path)
    assert report["ready_for_review"] and not report["coverage_certified"]
    assert not report["execution_validation_performed"] and not report["affects_coverage"]
    for item in report["operators"].values():
        assert set(item["replays"]) == {"v15-control", "vendor-whitelist"}
    tool.path.write_text(tool.path.read_text() + "\n")
    with pytest.raises(ValueError, match="compiler hash differs"):
        DIAG.diagnose(root, root, tmp_path / "changed", compiler=tool.path)


def test_cli_preserves_failed_diagnostic_and_hash_index(tmp_path):
    output = tmp_path / "failed"
    status = DIAG.main(["--results", str(tmp_path / "missing"), "--out", str(output), "--prepare-only"])
    assert status == 1
    assert json.loads((output / "execution.json").read_text())["returncode"] == 1
    report = json.loads((output / "evidence/diagnostic-report.json").read_text())
    assert report["state"] == "failed" and not report["coverage_certified"]
    for entry in json.loads((output / "evidence/artifacts.json").read_text())["files"]:
        assert sha((output / "evidence" / entry["path"]).read_bytes()) == entry["sha256"]


def test_cli_never_writes_into_source_evidence(tmp_path):
    source = tmp_path / "evidence"
    source.mkdir()
    result = subprocess.run(
        [sys.executable, str(Path(DIAG.__file__)),
         "--results", str(source), "--out", str(source / "new")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "must not modify the archived results or submission" in result.stderr
    assert not list(source.iterdir())


def test_hidden_help_is_probed_and_preserved_without_compiling(tmp_path):
    path = tmp_path / "bishengir-compile"
    path.write_text(f"""#!{sys.executable}
import sys
if sys.argv[1:] == ['--help']:
    print('  --help-hidden  Show more options')
elif sys.argv[1:] == ['--help-hidden']:
    print('  --bishengir-print-ir-before=<pass>  Synthetic option')
else:
    sys.exit(99)
""")
    path.chmod(0o755)
    output = tmp_path / "out"
    text, probe = DIAG.compiler_help(path, output)
    assert "--bishengir-print-ir-before=" in text
    assert [query["option"] for query in probe["queries"]] == ["--help", "--help-hidden"]
    assert all(query["returncode"] == 0 for query in probe["queries"])
    for artifact in probe["artifacts"]:
        DIAG.read_blob(output / "tool", artifact)
    assert (output / "tool/probe.json").is_file()


@pytest.mark.parametrize("failure", ["returncode", "truncated"])
def test_failed_or_truncated_help_remains_available_for_diagnosis(tmp_path, failure):
    path = tmp_path / "bishengir-compile"
    content = (
        'import sys\nprint("x" * (1024*1024+1))\n'
        if failure == "truncated"
        else 'import sys\nprint("tool error", file=sys.stderr)\nsys.exit(2)\n'
    )
    path.write_text(f"#!{sys.executable}\n" + content)
    path.chmod(0o755)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="help failed or exceeded"):
        DIAG.compiler_help(path, output)
    probe = json.loads((output / "tool/probe.json").read_text())
    assert probe["queries"][0].get("truncated") or probe["queries"][0]["returncode"] == 2
    assert all(DIAG.read_blob(output / "tool", entry) is not None for entry in probe["artifacts"])


def test_replay_reproduces_v15_dump_environment_without_mutating_parent(tool, monkeypatch):
    monkeypatch.setenv("MLIR_ENABLE_DUMP", "0")
    tool.path.write_text(tool.path.read_text() + '\nimport os\nassert os.environ["MLIR_ENABLE_DUMP"] == "1"\n')
    tool.item["tool_sha256"] = DIAG.file_digest(tool.path)
    result = DIAG.replay(tool.item, tool.path, tool.help, "vendor-whitelist", tool.output, 10)
    assert result["ready_for_review"] and result["environment_overrides"] == {"MLIR_ENABLE_DUMP": "1"}
    assert os.environ["MLIR_ENABLE_DUMP"] == "0"


@pytest.mark.parametrize(
    ("control_changes", "target_changes", "expected"),
    [
        ({}, {}, "backend_dump_still_missing"),
        ({}, {"dump_observed": True, "ready_for_review": True}, "vendor_filter_restores_dump_in_this_replay"),
        (
            {"dump_observed": True, "ready_for_review": True},
            {"dump_observed": True, "ready_for_review": True},
            "both_variants_produce_dump",
        ),
        ({"dump_observed": True, "ready_for_review": True}, {}, "only_control_produces_dump"),
        ({}, {"dump_observed": True}, "dump_present_but_capture_needs_review"),
        ({"returncode": 124}, {}, "compiler_replay_failed"),
        ({}, {"binary_matches_archived": False, "dump_observed": True}, "archived_binary_not_reproduced"),
        ({}, {"tool_identity_unchanged": False}, "compiler_identity_changed"),
    ],
)
def test_replay_comparisons_are_explicitly_diagnostic_only(control_changes, target_changes, expected):
    base = dict(
        returncode=0,
        tool_identity_unchanged=True,
        binary_matches_archived=True,
        dump_observed=False,
        ready_for_review=False,
    )
    result = DIAG.compare_replays(
        {"v15-control": dict(base, **control_changes), "vendor-whitelist": dict(base, **target_changes)}
    )
    assert result["conclusion"] == expected
    assert result["diagnostic_only"] and not result["coverage_certified"]
    assert not result["execution_validation_performed"]


def test_prepared_provenance_survives_compiler_preflight_failure(source, monkeypatch, tmp_path):
    monkeypatch.setattr(DIAG, "installed_compiler", lambda *_: (_ for _ in ()).throw(ValueError("no compiler")))
    output = tmp_path / "failed"
    status = DIAG.main(["--results", str(source.root), "--out", str(output)])
    report = json.loads((output / "evidence/diagnostic-report.json").read_text())
    assert status == 1 and report["state"] == "failed"
    assert report["source_manifest_hash"] == DIAG.MANIFEST and set(report["operators"]) == set(DIAG.TARGETS)
    assert report["diagnostic_script_sha256"] == DIAG.file_digest(Path(DIAG.__file__))


def test_shell_entry_rejects_additional_operators_and_missing_explicit_evidence(tmp_path):
    script = Path(DIAG.__file__).with_name("backend-dump-diagnose.sh")
    result = subprocess.run([str(script), "--operators", "Abs"], capture_output=True, text=True)
    assert result.returncode == 2 and "only" in result.stderr
    result = subprocess.run(
        [str(script)],
        capture_output=True,
        text=True,
        env=dict(os.environ, TRITON_EVAL_BACKEND_SOURCE=str(tmp_path / "absent")),
    )
    assert result.returncode == 1 and "explicitly selected evidence directory is missing" in result.stderr


def write_shell_entry(script):
    script.write_bytes(Path(DIAG.__file__).with_name(script.name).read_bytes())
    script.chmod(0o755)


@pytest.mark.parametrize("source_mode", ["local", "archive", "expired"])
def test_shell_entry_uses_pinned_archive_or_original_local_evidence_without_running_a_suite(tmp_path, source_mode):
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "backend-dump-diagnose.sh"
    write_shell_entry(script)
    (scripts / "env-bootstrap.sh").write_text("export BACKEND_DIAGNOSTIC_BOOTSTRAPPED=1\n")
    (scripts / "diagnose_backend_dump.py").write_text("""import json, os, pathlib, sys
assert os.environ['BACKEND_DIAGNOSTIC_BOOTSTRAPPED'] == '1'
assert sys.argv[-1] == '--publish'
assert pathlib.Path(sys.argv[2], 'coverage-validation.json').is_file()
pathlib.Path('invocation.json').write_text(json.dumps(sys.argv[1:]))
""")
    record = "20261026-120532-223940"
    archive = tmp_path / "archive"
    evidence = (
        root / "runs/ascend-suite" / record / "evidence"
        if source_mode == "local"
        else archive / "records" / record / "evidence"
    )
    evidence.mkdir(parents=True)
    (evidence / "coverage-validation.json").write_text("{}")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python").symlink_to(sys.executable)
    fake_git = binaries / "git"
    _write_archive_git_stub(fake_git, record)
    fake_git.chmod(0o755)
    result = subprocess.run(
        [str(script)],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            PATH=str(binaries) + os.pathsep + os.environ["PATH"],
            SOURCE_MODE=source_mode,
            FAKE_ARCHIVE=str(archive),
            TRITON_EVAL_BACKEND_SOURCE="",
        ),
    )
    _assert_archive_invocation(result, root, source_mode, record)


def _assert_archive_invocation(result, root, source_mode, record):
    if source_mode == "expired":
        assert result.returncode == 1 and "expired from the rolling branch" in result.stderr
        assert not (root / "invocation.json").exists()
    else:
        assert result.returncode == 0, result.stderr
        arguments = json.loads((root / "invocation.json").read_text())
        assert arguments[0] == "--results" and arguments[-1] == "--publish"
        assert record in arguments[1]
        assert (
            str(root / "runs/ascend-suite") in arguments[1]
            if source_mode == "local"
            else "backend-diagnostic-source." in arguments[1]
        )


def _write_archive_git_stub(fake_git, record):
    fake_git.write_text(f"""#!{sys.executable}
import os, pathlib, subprocess, sys
arguments = sys.argv[1:]
assert os.environ['SOURCE_MODE'] != 'local', 'local source must not fetch'
if arguments == ['fetch','origin','+refs/heads/npu-results:refs/remotes/origin/npu-results']:
    sys.exit(0)
if arguments == ['rev-parse','refs/remotes/origin/npu-results']:
    print('a'*40)
elif arguments == ['cat-file','-e','a'*40+':records/{record}/evidence/coverage-validation.json']:
    sys.exit(1 if os.environ['SOURCE_MODE']=='expired' else 0)
elif arguments == ['archive','a'*40,'records/{record}']:
    sys.exit(subprocess.run(['tar','-c','-C',os.environ['FAKE_ARCHIVE'],arguments[2]]).returncode)
else:
    sys.exit(99)
""")
