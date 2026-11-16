#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build a redundant official-source candidate corpus, never a claimed NPU qualification."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import logging

import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from triton_eval.buildlog import GenerationLog, LOG_FILENAME
from triton_eval.corpusmod import build_manifest, load_manifest, write_manifest
from triton_eval.models import CaseSource, OperatorSpec
from triton_eval.official_catalog import CANDIDATES, COMPILER_ONLY, VERSION, specification, submission_entry_name
from triton_eval.operator_registry import by_name
from triton_eval.specload import load_feature_atoms, load_spec


from triton_eval.official_generation import SHAPES, cases_for, coverage_gaps
from triton_eval.official_generation import select_cases as _select_cases
from triton_eval.console import emit

from triton_eval.fixture_sources import fixture_path, require_source_dependencies


def select_cases():
    return _select_cases(CANDIDATES)


def _submission_function(candidate, spec, entry_name):
    tensor_names = [item.name for item in spec.inputs]
    parameters = tensor_names + [f"{item.name}={item.default!r}" for item in spec.attrs]
    attribute_values = "{" + ", ".join(f"{item.name!r}: {item.name}" for item in spec.attrs) + "}"
    tensors = "(" + ", ".join(tensor_names) + ("," if len(tensor_names) == 1 else "") + ")"
    return (
        f"\n\ndef {entry_name}({', '.join(parameters)}):\n"
        f"    return dispatch({candidate.name!r}, {tensors}, {attribute_values})\n"
    )


def prepare_assets(corpus, submission):
    lock = json.loads((ROOT / "reference/ascend-official/sources-v08.json").read_text(encoding="utf-8"))
    source_map = {record["name"]: record for record in lock["operators"]}
    catalogue = {
        "format": 1,
        "version": VERSION,
        "status": "CANDIDATE_NOT_NPU_QUALIFIED",
        "generation": {"generator": "official-domain-v2", "selection_policy": "bounded-marginal-v2"},
        "target_operators": 50,
        "cases_per_operator": 20,
        "sources": lock,
        "operators": [],
    }
    template = (fixture_path(ROOT / "scripts/templates/official_kernels.py")).read_text(encoding="utf-8")
    functions, entries = [], []
    for candidate in CANDIDATES:
        spec = by_name(candidate.name)
        source = source_map[candidate.name]
        if source["path"] != candidate.source or source["expression"] != candidate.expression:
            raise ValueError(f"source mapping drift: {candidate.name}")
        catalogue["operators"].append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "counts_toward_operator_target": candidate.name not in COMPILER_ONLY,
                "canonical_semantics": candidate.expression,
                "qualification_spec": specification(candidate).model_dump(mode="json"),
                "source_sha256": source["sha256"],
            }
        )
        _write_candidate_references(corpus, spec, candidate)
        entry_name = submission_entry_name(candidate.name, spec.entry)
        functions.append(_submission_function(candidate, spec, entry_name))
        entries.append({"name": candidate.name, "entry": entry_name, "path": "operators/kernels.py"})
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "official-catalog.json").write_text(
        json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (submission / "operators").mkdir(parents=True, exist_ok=True)
    families = {candidate.name: candidate.family for candidate in CANDIDATES}
    generated_source = template + "\n\nFAMILIES = " + repr(families) + "\n" + "".join(functions)
    require_source_dependencies(generated_source, "operators/kernels.py")
    (submission / "operators/kernels.py").write_text(generated_source, encoding="utf-8")
    import yaml

    (submission / "manifest.yaml").write_text(
        yaml.safe_dump({"submit_id": VERSION, "operators": entries}, sort_keys=False), encoding="utf-8"
    )


def _write_candidate_references(corpus, spec, candidate):
    directory = corpus / "operators" / spec.entry
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ref.py").write_text(
        f"from triton_eval.official_reference import bind\n\nreference, make_inputs = bind({candidate.name!r})\n",
        encoding="utf-8",
    )
    (directory / "ref_cpu.py").write_text(
        f"from triton_eval.official_reference_cpu import bind\n\nreference = bind({candidate.name!r})\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus" / VERSION)
    parser.add_argument("--submission", type=Path, default=ROOT / "examples/official-ascend-submission")
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    if (corpus / "manifest.json").exists():
        _verify_existing_corpus(corpus)
        return 0
    prepare_assets(corpus, args.submission.resolve())
    cases, reports = select_cases()
    log = GenerationLog(corpus / LOG_FILENAME)
    log.step(
        "official_source_selection",
        generator="official-domain-v2",
        candidates=len(CANDIDATES),
        case_count=len(cases),
        reports=reports,
    )
    from triton_eval.refcheck import check_all

    failures = check_all(cases, "cpu", corpus_dir=corpus)
    log.step("independent_cpu_reference_gate", failures=failures)
    if failures:
        emit(json.dumps(failures, ensure_ascii=False, indent=2))
        return 1
    manifest = _build_selected_manifest(corpus, cases)
    write_manifest(manifest, corpus / "manifest.json")
    from triton_eval.corpus_assets import freeze_assets

    freeze_assets(corpus, manifest.manifest_hash)
    pool = [case for candidate in CANDIDATES for case in cases_for(candidate)]
    coverage = coverage_gaps(cases, pool, load_feature_atoms())
    (corpus / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    log.step("coverage_gap_analysis", **coverage)
    emit(
        json.dumps(
            {
                "candidate_operators": len(CANDIDATES),
                "cases": len(cases),
                "per_operator": dict(Counter(case.operator for case in cases)),
                "manifest_hash": manifest.manifest_hash,
                "npu_qualified": False,
            },
            indent=2,
        )
    )
    return 0


def _verify_existing_corpus(corpus):
    from triton_eval.corpusmod import verify_corpus_files

    existing = load_manifest(corpus / "manifest.json")
    verify_corpus_files(corpus, existing)
    from triton_eval.corpus_assets import verify_assets

    verify_assets(corpus, existing.manifest_hash)
    emit(f"Frozen candidate assets verified: {existing.version}, {len(existing.cases)} cases")


def _build_selected_manifest(corpus, cases):
    operators = [
        OperatorSpec(
            name=candidate.name,
            source=CaseSource.VENDOR_SELFTEST,
            weight=1.0,
            ref_path=f"operators/{by_name(candidate.name).entry}/ref.py",
            desc=f"Official source candidate / {candidate.category.value}",
        )
        for candidate in CANDIDATES
    ]
    manifest = build_manifest(
        corpus,
        VERSION,
        load_spec().spec_version,
        operators,
        cases,
        feature_atoms_version=str(load_feature_atoms()["version"]),
    )
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
