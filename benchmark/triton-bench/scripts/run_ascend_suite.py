#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""复用正式 CLI，按算子隔离执行、逐分区审计并保留每轮原始身份。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import argparse


from check_run_completeness import audit_results
from triton_eval.console import emit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def audit_partition(manifest, operator, lines):
    selected = dict(manifest, cases=[case for case in manifest["cases"] if case["operator"] == operator])
    return audit_results(selected, lines)


def run_logged(command, path, timeout=None):
    emit(" ".join(map(str, command)), flush=True)
    with path.open("w", encoding="utf-8") as stream:
        try:
            return subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124


def backend_pattern_gate(output):
    path = output / 'coverage-validation.json'
    proof = json.loads(path.read_text()) if path.is_file() else {}
    atoms = ('FA-PAT-VEC-LOAD', 'FA-PAT-TILE-GEMM')
    checks = {}
    for atom in atoms:
        checks[atom] = {}
        for metric in ('compilation', 'execution_validation'):
            checks[atom][metric] = atom in proof.get(metric, {}).get('covered', [])
    passed = (proof.get('selected_operator_validation_passed') is True
              and all(all(check.values()) for check in checks.values()))
    report = dict(scope='two_backend_patterns', passed=passed, checks=checks,
                  coverage_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
                  note='functionality PASS does not imply backend pattern certification')
    (output / 'backend-pattern-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    emit(f'[backend] 两项模式认证门禁：{"PASS" if passed else "FAIL"}；功能 PASS 不替代此门禁', flush=True)
    return int(not passed)


def main():
    parser, args = _suite_arguments()
    from triton_eval.fixture_sources import fixture_path

    args.submission = fixture_path(args.submission)

    corpus, manifest, coverage_mode, operators = _suite_inputs(args, parser)

    output = _suite_output(args, corpus, coverage_mode)
    if output is None:
        return 1
    submission = args.submission.resolve()
    summary = _suite_summary(manifest, args, operators, coverage_mode)

    summary_path = output / "suite-summary.json"

    def save():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")

    save()
    if _suite_health_gate(args, corpus, output, summary, save, coverage_mode):
        return 1
    database = output / "eval.sqlite"
    prefix = [sys.executable, "-m", "triton_eval", "--db", database]
    payload = manifest.model_dump()
    for operator in operators:
        _run_suite_operator(operator, args, corpus, submission, output, prefix, payload, summary, save)
    _summarize_suite(summary, operators, save, corpus, manifest, submission, output)

    backend_code = _audit_suite_coverage(coverage_mode, corpus, output, submission, summary, manifest, operators)

    emit(f"结果目录：{output}\n完整性：{summary['complete']}；全 PASS：{summary['all_pass']}")
    return int(not (summary["all_pass"] and summary["commands_ok"]
                    and summary.get("qualification_returncode", 0) == 0
                    and summary.get("coverage_audit_returncode", 0) == 0 and backend_code == 0))




def _suite_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus/v0.9-official-candidates")
    parser.add_argument("--submission", type=Path, default=ROOT / "examples/official-ascend-submission")
    parser.add_argument("--profile", choices=("smoke", "standard", "official"), default="smoke")
    parser.add_argument("--operators", help="逗号分隔的注册算子名；缺省执行全部")
    parser.add_argument("--isolate-reference-gates", action="store_true",
                        help="gate each operator separately and continue after reference failures")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    return parser, args


def _suite_inputs(args, parser):
    from triton_eval.corpus_assets import verify_assets
    from triton_eval.corpusmod import load_manifest, verify_corpus_files

    corpus = args.corpus.resolve()
    if (corpus / "official-catalog.json").is_file():
        args.isolate_reference_gates = True
    manifest = load_manifest(corpus / "manifest.json")
    verify_corpus_files(corpus, manifest)
    verify_assets(corpus, manifest.manifest_hash)
    coverage_mode = (corpus / "coverage-plan.json").is_file()
    if coverage_mode:
        from triton_eval.coverage_gate import static_readiness
        static_readiness(corpus)
        args.isolate_reference_gates = True
    operators = sorted({case.operator for case in manifest.cases})
    if args.operators:
        requested = {name.strip() for name in args.operators.split(",") if name.strip()}
        if not requested or requested - set(operators):
            parser.error(f"未知或空算子集合：{sorted(requested - set(operators))}")
        operators = [name for name in operators if name in requested]
    return corpus, manifest, coverage_mode, operators


def _suite_output(args, corpus, coverage_mode):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = (args.out or ROOT / f"verify-records/ascend-contract-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    if coverage_mode:
        os.environ.setdefault('TRITON_EVAL_CAPTURE_PASSES', '1')
        os.environ.setdefault('TRITON_EVAL_CAPTURE_BACKEND', '1')
        os.environ.setdefault('MLIR_ENABLE_DUMP', '1')
        # A fresh suite cache prevents reuse of artifacts from an older compiler
        # session; repeated specializations within this suite may still hit it.
        os.environ['TRITON_CACHE_DIR'] = str(output / '.compiler-cache')
    submission = args.submission.resolve()
    snapshot = {}
    for path in sorted(submission.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
            snapshot[str(path.relative_to(submission))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "submission-sha256.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    (output / "evaluation-assets.json").write_bytes((corpus / "evaluation-assets.json").read_bytes())
    if coverage_mode:
        # Validate the entire submission once before any reference/device work.
        # Keep imports isolated from the coordinator and preserve their report.
        preflight_code = run_logged([
            sys.executable, ROOT / 'scripts/check_submission_readiness.py',
            '--corpus', corpus, '--submission', submission, '--load',
            '--out', output / 'submission-preflight.json',
        ], output / 'submission-preflight.log', timeout=120)
        if preflight_code:
            emit(f'[suite] submission preflight failed ({preflight_code}); '
                         f'see {output / "submission-preflight.log"}', flush=True)
            return None

    (output / "manifest.json").write_bytes((corpus / "manifest.json").read_bytes())
    if coverage_mode:
        (output / "coverage-plan.json").write_bytes((corpus / "coverage-plan.json").read_bytes())
    return output


def _suite_summary(manifest, args, operators, coverage_mode):
    summary = {"manifest_hash": manifest.manifest_hash, "profile": args.profile,
               "selected_operators": operators, "corpus_cases": len(manifest.cases),
               "expected_cases": sum(case.operator in operators for case in manifest.cases),
               "operators": {}, "complete": False, "all_pass": False}
    if coverage_mode:
        from triton_eval.reference_execution import POLICY
        summary['reference_execution_policy'] = POLICY
        summary['compiler_pass_capture_requested'] = os.environ.get('TRITON_EVAL_CAPTURE_PASSES') == '1'
        summary['backend_capture_requested'] = os.environ.get('TRITON_EVAL_CAPTURE_BACKEND') == '1'
        summary['backend_pattern_gate_required'] = os.environ.get('TRITON_EVAL_REQUIRE_BACKEND_PATTERNS') == '1'
        summary['target_pattern_gate_required'] = os.environ.get('TRITON_EVAL_REQUIRE_TARGET_PATTERNS') == '1'
        summary['validation_scope'] = ('full_corpus' if len(operators) == len(manifest.operators)
                                       else 'operator_regression')
        summary['coverage_target_gate_applied'] = summary['validation_scope'] == 'full_corpus'
        summary['compiler_cache_scope'] = 'fresh per suite; specialization reuse within suite allowed'
    return summary


def _suite_health_gate(args, corpus, output, summary, save, coverage_mode):
    for device in (() if args.isolate_reference_gates else ("cpu", "npu:0")):
        code = run_logged([sys.executable, ROOT / "scripts/verify_refs.py",
                           "--corpus", corpus, "--device", device],
                          output / f"reference-{device.replace(':', '-')}.log", timeout=1800)
        if code:
            summary["gate_failure"] = {"device": device, "returncode": code}
            save()
            return 1
    capability_command = [sys.executable, ROOT / "scripts/probe_ascend_capabilities.py",
                          "--out", output / "capabilities"]
    if coverage_mode and summary['validation_scope'] == 'operator_regression':
        capability_command.append('--health-only')
    capability_code = run_logged(capability_command, output / "capabilities.log", timeout=600)
    if capability_code:
        summary["gate_failure"] = {"stage": "health_probe", "returncode": capability_code}
        save()
        return 1
    return 0


def _run_suite_operator(operator, args, corpus, submission, output, prefix, payload, summary, save):
    partition = output / operator
    partition.mkdir()
    gate_codes = _suite_operator_reference_gate(args, corpus, partition, payload, operator, summary, save)
    if args.isolate_reference_gates and gate_codes is None:
        return
    run_code = run_logged(prefix + ["run", "--corpus", corpus, "--submission", submission,
                                    "--profile", args.profile, "--operators", operator,
                                    "--runs", partition], partition / "run.log")
    result_files = list(partition.glob("run-*/results.jsonl"))
    lines = result_files[0].read_text(encoding="utf-8").splitlines() if len(result_files) == 1 else []
    audit = audit_partition(payload, operator, lines)
    if args.isolate_reference_gates:
        audit["reference_gate_returncodes"] = gate_codes
    audit["run_returncode"] = run_code
    if len(result_files) != 1:
        audit["errors"].append(f"expected one result file, found {len(result_files)}")
        audit["complete"] = audit["all_pass"] = False
    if len(audit["run_ids"]) == 1:
        run_id = str(audit["run_ids"][0])
        audit["score_returncode"] = run_logged(prefix + ["score", "--run", run_id],
                                               partition / "score.log")
        audit["report_returncode"] = run_logged(prefix + ["report", "--run", run_id,
                                                          "--out", partition], partition / "report.log")
    else:
        audit["score_returncode"] = audit["report_returncode"] = 1
    summary["operators"][operator] = audit
    emit(f"{operator}: complete={audit['complete']}, all_pass={audit['all_pass']}", flush=True)
    save()


def _suite_operator_reference_gate(args, corpus, partition, payload, operator, summary, save):
    gate_codes = {}
    if args.isolate_reference_gates:
        gate_failure = None
        gate_codes = {}
        for device in ("cpu", "npu:0"):
            code = run_logged([sys.executable, ROOT / "scripts/verify_refs.py", "--corpus", corpus,
                               "--device", device, "--operators", operator],
                              partition / f"reference-{device.replace(':', '-')}.log", timeout=900)
            gate_codes[device] = code
            if code:
                gate_failure = {"device": device, "returncode": code, "attribution": "REFERENCE_GATE_FAILURE"}
                break
        if gate_failure:
            audit = audit_partition(payload, operator, [])
            audit.update(reference_gate_failure=gate_failure, run_returncode=1,
                         score_returncode=1, report_returncode=1)
            summary["operators"][operator] = audit
            save()
            emit(f"{operator}: reference gate failed; continuing", flush=True)
            return None
    return gate_codes


def _summarize_suite(summary, operators, save, corpus, manifest, submission, output):
    audits = list(summary["operators"].values())
    summary["complete"] = len(audits) == len(operators) and all(item["complete"] for item in audits)
    summary["all_pass"] = summary["complete"] and all(item["all_pass"] for item in audits)
    command_keys = ("run_returncode", "score_returncode", "report_returncode")
    summary["commands_ok"] = all(not any(item[key] for key in command_keys) for item in audits)
    save()
    if (corpus / "official-catalog.json").is_file() and len(operators) == len(manifest.operators):
        qualification_code = run_logged([
            sys.executable, ROOT / "scripts/qualify_official_results.py", "--corpus", corpus,
            "--submission", submission, "--results", output], output / "qualification.log")
        summary["qualification_returncode"] = qualification_code
        save()
    elif (corpus / "official-catalog.json").is_file():
        summary["qualification_skipped"] = ("partial operator run; a full same-version suite is required for 50/1000 "
                                            "qualification")
        save()


def _audit_suite_coverage(coverage_mode, corpus, output, submission, summary, manifest, operators):
    backend_code = 0
    if coverage_mode:
        coverage_command = [sys.executable, ROOT / "scripts/audit_coverage_results.py",
                            "--corpus", corpus, "--results", output, "--submission", submission]
        if summary['validation_scope'] == 'operator_regression':
            coverage_command.append('--operator-regression')
        coverage_code = run_logged(coverage_command, output / "coverage-audit.log")
        # The auditor hashes the finalized summary. Keep it immutable after
        # capture; coverage-validation.json records the audited coverage result.
        summary["coverage_audit_returncode"] = coverage_code
        if os.environ.get('TRITON_EVAL_CAPTURE_BACKEND') == '1' and (output / 'coverage-validation.json').is_file():
            proof = json.loads((output / 'coverage-validation.json').read_text())
            atoms = ('FA-PAT-VEC-LOAD', 'FA-PAT-TILE-GEMM')
            prefix = '[backend]'
            if summary['target_pattern_gate_required']:
                from triton_eval.pattern_release import required_pattern_targets
                atoms = required_pattern_targets(manifest.version, operators)
                prefix = '[patterns]'
            emit(f'{prefix} 以下是模式认证状态，不等同于用例正确性 PASS：', flush=True)
            for atom in atoms:
                compiled = atom in proof['compilation']['covered']
                executed = atom in proof['execution_validation']['covered']
                emit(f'{prefix} {atom}: compilation={compiled}, execution_validation={executed}', flush=True)
        if summary['backend_pattern_gate_required']:
            backend_code = backend_pattern_gate(output)
        if summary['target_pattern_gate_required']:
            from triton_eval.pattern_release import pattern_gate
            target_code = pattern_gate(output)
            emit(f'[patterns] 目标模式认证门禁：{"PASS" if target_code == 0 else "FAIL"}', flush=True)
            backend_code = backend_code or target_code
    return backend_code


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
