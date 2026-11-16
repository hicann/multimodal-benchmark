# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""冻结参考、契约与判定资产，补充仅覆盖用例参数的 manifest 指纹。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from triton_eval import framework_path
from triton_eval.migration_metadata import decode_metadata, metadata_path

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_FILES = (
    "src/triton_eval/corpus_assets.py",
    "src/triton_eval/migration_metadata.py",
    "src/triton_eval/__init__.py",
    "src/triton_eval/console.py",
    "src/triton_eval/reviewed_kernels.py",
    "assets/reviewed-kernels.json",
    "spec/spec.yaml",
    "spec/tolerance.yaml",
    "spec/feature_atoms.yaml",
    "src/triton_eval/ascend_contracts.py",
    "src/triton_eval/comparison.py",
    "src/triton_eval/judge.py",
    "src/triton_eval/inputs.py",
    "src/triton_eval/operator_registry.py",
    "src/triton_eval/operator_registry_ascend_ops.py",
    "src/triton_eval/refcheck.py",
    "src/triton_eval/harness/npu_exec.py",
    "src/triton_eval/oracle.py",
    "src/triton_eval/runner.py",
    "src/triton_eval/integrity.py",
    "src/triton_eval/evalchild.py",
    "src/triton_eval/validator.py",
    "src/triton_eval/submission_preflight.py",
    "scripts/check_submission_readiness.py",
    "scripts/build_corpus_v06_ascend_ops.py",
    "src/triton_eval/coverage.py",
    "src/triton_eval/official_catalog.py",
    "src/triton_eval/official_reference.py",
    "src/triton_eval/official_reference_cpu.py",
    "scripts/build_official_corpus.py",
    "reference/ascend-official/sources-v08.json",
    "scripts/qualify_official_results.py",
    "src/triton_eval/official_generation.py",
    "src/triton_eval/operatorspec.py",
    "src/triton_eval/casegen.py",
    "reference/ascend-official/cann-correspondence-v08.json",
    "corpus/v0.9-official-candidates/manifest.json",
    "src/triton_eval/buildlog.py",
    "scripts/templates/coverage_kernels.py",
    "src/triton_eval/coverage_generation.py",
    "src/triton_eval/coverage_reference.py",
    "src/triton_eval/coverage_gate.py",
    "src/triton_eval/compilation_evidence.py",
    "src/triton_eval/implementation_evidence.py",
    "scripts/templates/implementation_witness_kernels.py",
    "scripts/templates/official_kernels.py",
    "src/triton_eval/coverage_evidence.py",
    "scripts/build_coverage_corpus.py",
    "scripts/check_coverage_readiness.py",
    "scripts/audit_coverage_results.py",
    "scripts/run_ascend_suite.py",
    "scripts/coverage-suite-verify.sh",
    "scripts/templates/compatibility_kernels.py",
    "scripts/coverage-regression-verify.sh",
    "scripts/probe_ascend_capabilities.py",
    "scripts/ascend_capability_kernels.py",
    "src/triton_eval/reference_execution.py",
    "src/triton_eval/ascend_lowering.py",
    "scripts/templates/stability_kernels.py",
    "scripts/coverage-gap-verify.sh",
    "src/triton_eval/coverage_overlay.py",
    "scripts/analyze_coverage_overlay.py",
    "src/triton_eval/backend_capture.py",
    "src/triton_eval/backend_evidence.py",
    "scripts/coverage-backend-verify.sh",
    "src/triton_eval/pattern_generation.py",
    "src/triton_eval/pattern_evidence.py",
    "src/triton_eval/pattern_contracts.py",
    "src/triton_eval/pattern_release.py",
    "scripts/templates/target_pattern_kernels.py",
    "scripts/build_pattern_corpus.py",
    "scripts/coverage-patterns-verify.sh",
    "scripts/build_scan_certification_corpus.py",
    "scripts/coverage-scan-verify.sh",
)


def revision_history_paths(corpus_dir: Path, metadata: dict, manifest_hash: str) -> list[Path]:
    revisions = metadata.get("previous_revisions", [])
    if not isinstance(revisions, list):
        raise ValueError("invalid asset revision history")
    seen = {metadata["origin_assets_hash"]}
    paths = []
    for revision in revisions:
        if not isinstance(revision, dict) or set(revision) != {"assets_hash", "migration_metadata"}:
            raise ValueError("invalid previous asset revision metadata")
        identity = revision["assets_hash"]
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise ValueError("invalid previous asset identity")
        if identity in seen:
            raise ValueError("invalid previous asset identity")
        seen.add(identity)
        path = corpus_dir / "evaluation-assets-history" / (identity + ".json")
        archived = json.loads(path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in archived.items() if key != "assets_hash"}
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if (
            hashlib.sha256(content).hexdigest() != identity
            or archived.get("assets_hash") != identity
            or archived.get("manifest_hash") != manifest_hash
        ):
            raise ValueError("previous asset identity or case manifest differs")
        stored_metadata = revision["migration_metadata"]
        previous_metadata = decode_metadata(stored_metadata, corpus_dir)
        expected = archived["files"].get("@corpus/evaluation-assets-migration.json")
        if hashlib.sha256(previous_metadata).hexdigest() != expected:
            raise ValueError("previous asset migration metadata differs")
        paths.append(path)
        if (
            isinstance(stored_metadata, dict)
            and stored_metadata.get("encoding") == "zlib-base64-file"
        ):
            paths.append(metadata_path(stored_metadata, corpus_dir))
    return paths


def reference_layout(corpus_dir: Path) -> dict:
    """Read an explicit, hash-bound mapping of historical reference paths."""
    index = corpus_dir / "reference-layout.json"
    if not index.exists():
        return {}
    layout = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(layout, dict) or set(layout) != {"format", "files"} or layout["format"] != 1:
        raise ValueError("invalid reference layout")
    if not isinstance(layout["files"], dict) or not layout["files"]:
        raise ValueError("empty or invalid reference layout")
    for name, record in layout["files"].items():
        _validate_reference_name(name)
        _validate_reference_record(record)
    return layout["files"]


def _unsafe_reference_parts(path, *, historical):
    invalid_parts = any(part in ("", ".", "..") for part in path.parts)
    invalid_filename = (
        path.name not in ("ref.py", "ref_cpu.py") if historical else path.suffix != ".py"
    )
    return invalid_parts or invalid_filename


def _validate_reference_name(name):
    if not isinstance(name, str):
        raise ValueError("invalid historical reference name")
    logical = Path(name)
    if (
        len(logical.parts) != 3
        or logical.parts[0] != "operators"
        or _unsafe_reference_parts(logical, historical=True)
    ):
        raise ValueError("invalid historical reference path")


def _validate_reference_record(record):
    if not isinstance(record, dict) or set(record) != {"source", "sha256"}:
        raise ValueError("invalid reference source record")
    if not isinstance(record["source"], str):
        raise ValueError("invalid shared reference name")
    source = Path(record["source"])
    if (
        source.is_absolute()
        or len(source.parts) != 3
        or _unsafe_reference_parts(source, historical=False)
    ):
        raise ValueError("invalid shared reference path")
    digest = record["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("invalid reference source digest")


def resolve_reference_path(corpus_dir: Path, relative: str) -> Path:
    """Resolve old paths to authenticated sources, or a byte-exact materialized copy."""
    records = reference_layout(corpus_dir)
    return _resolve_reference_source(corpus_dir, relative, records.get(relative))


def _resolve_reference_source(corpus_dir: Path, relative: str, record: dict | None) -> Path:
    local = corpus_dir / relative
    if record is None:
        return local
    source_root = ROOT / "corpus/reference_sources"
    materialized = local.exists() or local.is_symlink()
    candidate = local if materialized else source_root / record["source"]
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("missing or symbolic reference source")
    allowed_root = corpus_dir if materialized else source_root
    if not candidate.resolve().is_relative_to(allowed_root.resolve()):
        raise ValueError("reference source escapes its root")
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("reference source digest differs")
    return candidate


def reference_sources(corpus_dir: Path) -> dict[str, Path]:
    """Enumerate both local references and authenticated shared references."""
    sources = {
        path.relative_to(corpus_dir).as_posix(): path
        for path in sorted((corpus_dir / "operators").glob("*/ref*.py"))
    }
    for relative, record in reference_layout(corpus_dir).items():
        sources[relative] = _resolve_reference_source(corpus_dir, relative, record)
    return sources


def copy_reference_tree(corpus_dir: Path, destination: Path) -> None:
    """Rebuild historical operator paths without changing any reference bytes."""
    sources = reference_sources(corpus_dir)
    if (corpus_dir / "operators").is_dir():
        shutil.copytree(corpus_dir / "operators", destination / "operators", dirs_exist_ok=True)
    for relative, source in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def verify_corpus_asset_paths(corpus_dir: Path, files: dict, label: str) -> None:
    """Verify logical corpus assets without permitting arbitrary external paths."""
    references = reference_layout(corpus_dir)
    for name, expected in files.items():
        if not name.startswith("@corpus/"):
            continue
        relative = name.removeprefix("@corpus/")
        path = corpus_dir / relative
        if relative in references:
            path = _resolve_reference_source(corpus_dir, relative, references[relative])
        elif not path.resolve().is_relative_to(corpus_dir.resolve()):
            raise ValueError(label + ": " + name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(label + ": " + name)


def snapshot(corpus_dir: Path, manifest_hash: str) -> dict:
    corpus_dir = corpus_dir.resolve()
    from triton_eval.fixture_sources import LAYOUT, fixture_files

    framework_files = _snapshot_framework_files(corpus_dir)
    outputs = fixture_files(framework_files, ROOT)
    paths = [
        outputs[name] if name in outputs else framework_path(name, ROOT) for name in framework_files
    ]
    if (ROOT / LAYOUT).is_file():
        paths.extend([ROOT / "src/triton_eval/fixture_sources.py", ROOT / LAYOUT])
    framework_names = dict(zip(paths, framework_files))
    references = reference_sources(corpus_dir)
    indexed_references = (corpus_dir / "reference-layout.json").is_file()
    if indexed_references:
        paths.append(corpus_dir / "reference-layout.json")
    if (corpus_dir / "official-catalog.json").exists():
        paths.append(corpus_dir / "official-catalog.json")
    if (corpus_dir / "coverage-plan.json").is_file():
        paths.extend([corpus_dir / "coverage-plan.json", corpus_dir / "generation-log.yaml"])
    migration = corpus_dir / "evaluation-assets-migration.json"
    paths.extend(_snapshot_history(corpus_dir, manifest_hash))
    portable = (corpus_dir / "official-catalog.json").is_file() or (
        corpus_dir / "coverage-plan.json"
    ).is_file()
    files = _snapshot_files(paths, corpus_dir, portable or migration.is_file(), framework_names)
    for relative, source in references.items():
        key = (
            f"@corpus/{relative}"
            if indexed_references or portable or migration.is_file()
            else source.relative_to(ROOT).as_posix()
        )
        files[key] = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {
        "format": 3 if migration.is_file() else 2 if portable else 1,
        "manifest_hash": manifest_hash,
        "files": files,
    }
    payload["assets_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def freeze_assets(corpus_dir: Path, manifest_hash: str) -> dict:
    current = snapshot(corpus_dir, manifest_hash)
    target = corpus_dir / "evaluation-assets.json"
    if target.exists():
        verify_assets(corpus_dir, manifest_hash)
    else:
        target.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return current


def verify_assets(corpus_dir: Path, manifest_hash: str, *, required: bool = True) -> None:
    target = corpus_dir / "evaluation-assets.json"
    if not target.exists() and not required:
        return
    frozen = json.loads(target.read_text(encoding="utf-8"))
    try:
        current = snapshot(corpus_dir, manifest_hash)
    except (ValueError, OSError) as error:
        raise ValueError("评测资产与冻结指纹不一致：参考源码布局或指纹不一致") from error
    if current != frozen:
        changed = sorted(
            name
            for name in set(current["files"]) | set(frozen.get("files", {}))
            if current["files"].get(name) != frozen.get("files", {}).get(name)
        )
        raise ValueError(
            f"评测资产与冻结指纹不一致：{changed or ['manifest/assets_hash']}；请发布新版本"
        )


def verify_origin_assets(corpus_dir: Path, expected_origin: str) -> dict:
    """Authenticate an old release plus its explicit, byte-checked migration.

    Existing original indexes stay in history. Current files must still match
    their complete frozen snapshot; a predecessor is not a bypass for changes.
    """
    corpus_dir = Path(corpus_dir)
    active = json.loads((corpus_dir / "evaluation-assets.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in active.items() if key != "assets_hash"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != active.get("assets_hash"):
        raise ValueError("active asset index digest differs")
    if active.get("assets_hash") == expected_origin:
        return active
    migration_path = corpus_dir / "evaluation-assets-migration.json"
    if not migration_path.is_file():
        raise ValueError("asset identity differs without an authenticated migration")
    metadata = json.loads(migration_path.read_text(encoding="utf-8"))
    if metadata.get("origin_assets_hash") != expected_origin:
        raise ValueError("asset migration predecessor differs")
    archive = corpus_dir / "evaluation-assets-history" / (expected_origin + ".json")
    historical = json.loads(archive.read_text(encoding="utf-8"))
    payload = {key: value for key, value in historical.items() if key != "assets_hash"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        digest != expected_origin
        or historical.get("assets_hash") != expected_origin
        or historical.get("manifest_hash") != active.get("manifest_hash")
    ):
        raise ValueError("archived asset identity or case manifest differs")
    verify_assets(corpus_dir, active["manifest_hash"])
    return active


def _snapshot_framework_files(corpus_dir):
    framework_files = FRAMEWORK_FILES
    plan_path = corpus_dir / "coverage-plan.json"
    if (
        plan_path.is_file()
        and json.loads(plan_path.read_text()).get("version")
        == "v0.19-workload-completion-candidates"
    ):
        framework_files = (
            *FRAMEWORK_FILES,
            "src/triton_eval/workload_generation.py",
            "src/triton_eval/workload_contracts.py",
            "src/triton_eval/workload_release.py",
            "scripts/build_workload_corpus.py",
        )
    return framework_files


def _snapshot_history(corpus_dir, manifest_hash):
    paths = []
    migration = corpus_dir / "evaluation-assets-migration.json"
    if migration.is_file():
        metadata = json.loads(migration.read_text(encoding="utf-8"))
        origin = metadata.get("origin_assets_hash", "")
        if len(origin) != 64 or any(character not in "0123456789abcdef" for character in origin):
            raise ValueError("invalid archived asset identity")
        paths.extend([migration, corpus_dir / "evaluation-assets-history" / (origin + ".json")])
        paths.extend(revision_history_paths(corpus_dir, metadata, manifest_hash))
    return paths


def _snapshot_files(paths, corpus_dir, portable, framework_names):
    files = {
        (
            f"@corpus/{path.relative_to(corpus_dir).as_posix()}"
            if portable and path.is_relative_to(corpus_dir)
            else (
                framework_names[path]
                if path in framework_names
                else path.relative_to(ROOT).as_posix()
            )
        ): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return files
