# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""评测子进程（L1）——被测实现唯一的执行环境。

`python -m triton_eval.evalchild` 由 `cli.run` 作为监督者启动。
分工边界：

    父进程（cli.cmd_run）   只做安全的事：解包、静态校验、落库、超时、
                           崩溃归因——**绝不 import 提交代码**。
    本进程                  与不可信代码同室：加载、执行、判定、把结果
                           逐条写盘（events.jsonl / results.jsonl，
                           每条 flush——被 SIGKILL 也已写的不丢）。

事件协议（run_dir/events.jsonl，每行一个 JSON 对象，append + flush）：

    meta       提交校验结论（含篡改校验结果）
    env_check  子进程实际加载的模块身份 vs 父进程环境指纹
    probe      健康探针结论
    case       一条完整的 CaseResultRecord
    fatal      不可恢复的失败（stage: env / corpus / child）
    done       正常收尾

为什么本模块顶层只 import 标准库：它以 `__main__` 方式运行，
相对导入不可用；且顶层保持零依赖能让「子进程起不来」的故障
不会被一层 import 错误掩盖。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _capture(path: Path):
    """把本进程的 stdout/stderr（fd 级）重定向到日志文件。

    fd 级而非 sys.stdout 级：C 扩展（triton / torch_npu / 提交方可能
    用到的任何东西）的输出不经过 Python 的 sys.stdout，只有 dup2
    拦得住。事件协议走独立的 events.jsonl，与本重定向互不影响。
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(fd)
            os.close(saved_out)
            os.close(saved_err)


def _emit(fh, kind: str, **payload) -> None:
    fh.write(json.dumps({"event": kind, **payload}, ensure_ascii=False) + "\n")
    fh.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triton-eval-evalchild")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--operators", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"

    with open(events_path, "a", encoding="utf-8") as events_fh:
        try:
            _run(args, run_dir, logs, events_fh)
        except KeyboardInterrupt:
            raise
        except SystemExit as exc:
            # 提交代码的逃逸路径到这里仍可能存在（guard_call 拦截的是
            # 校验/加载边界）；进程退出码必须可区分，不允许静默成功
            _emit(events_fh, "fatal", stage="child",
                  detail=f"子进程收到 SystemExit({exc.code})")
            return 1
        except BaseException as exc:  # noqa: BLE001
            _emit(events_fh, "fatal", stage="child",
                  detail=f"{type(exc).__name__}: {exc}"[:2000])
            return 1
    return 0


def _run(args, run_dir: Path, logs: Path, events_fh) -> None:
    from triton_eval.corpusmod import load_manifest, select_cases, verify_corpus_files
    from triton_eval.harness.npu_exec import health_probe
    from triton_eval.integrity import ApiGuard
    from triton_eval.specload import load_spec, load_tolerance

    # ---------- 冻结资产校验（与父进程同一份，不一致即 fatal） ----------
    corpus_dir = Path(args.corpus)
    manifest = load_manifest(corpus_dir / "manifest.json")
    try:
        verify_corpus_files(corpus_dir, manifest)
    except (ValueError, FileNotFoundError) as exc:
        _emit(events_fh, "fatal", stage="corpus", detail=str(exc)[:2000])
        return
    # 参考实现是随语料版本冻结的资产——必须在评测启动时指向本语料目录，
    # 否则会从旧版本静默加载（oracle 的历史缺陷，2026-10-13 修复）
    from triton_eval.oracle import configure_corpus

    configure_corpus(corpus_dir)

    # ---------- 篡改校验基线：必须先于任何提交代码的 import ----------
    guard = ApiGuard.snapshot()

    # ---------- 校验与加载（import 期的输出全部进 load.log） ----------
    sub_root = Path(args.submission)
    validation, discovered, declared = _validate_child_submission(sub_root, logs, guard)
    sub, load_failures = _load_child_submission(discovered, validation, logs)

    _emit_child_submission(events_fh, sub, sub_root, validation, declared, load_failures, guard)

    if not validation.ok:
        _emit(events_fh, "done", reason="提交校验未通过")
        return

    if not declared:
        # 合法但几乎一定是失误：显式说明，而不是安静地全 SUBMISSION_INVALID
        _emit(events_fh, "warn", detail="提交清单未声明任何算子")

    if sub is None:
        _emit(events_fh, "fatal", stage="child", detail="提交对象不可用")
        return

    # ---------- 环境一致性：子进程实际加载 vs 父进程指纹 ----------
    if not _verify_child_environment(run_dir, events_fh):
        return

    # ---------- 健康探针 ----------
    ok, reason = health_probe()
    _emit(events_fh, "probe", ok=ok, reason=reason)
    if not ok:
        _emit(events_fh, "done", reason=f"环境健康探针失败：{reason}")
        return

    # ---------- 逐用例执行 ----------
    spec = load_spec()
    profile = spec.profile(args.profile)
    table = load_tolerance()
    # 过滤逻辑必须与监督者同源（select_cases）：预算按父进程口径算
    cases = select_cases(manifest.cases, args.operators)
    if args.limit:
        cases = cases[: args.limit]
    jsonl_path = run_dir / "results.jsonl"

    # 超时覆盖必须与监督者同源（TRITON_EVAL_CASE_TIMEOUT_S）：
    # 只覆盖监督者会让两边口径分裂——子进程按语料原始 60s 设 SIGALRM、
    # 监督者按覆盖值算预算，用例恰好卡在预算边缘被误杀
    # （真机实测：exp 用例 65s vs 预算 65s，第一个擦线、第二个被杀）。
    timeout_override = os.environ.get("TRITON_EVAL_CASE_TIMEOUT_S")
    if timeout_override:
        cases = [c.model_copy(update={"timeout_s": int(timeout_override)}) for c in cases]

    for case in cases:
        m, log_path = _execute_child_case(case, args, sub, logs, profile, guard, run_dir, manifest, corpus_dir)
        if not _record_child_case(m, case, args, table, jsonl_path, log_path, events_fh):
            return

    _emit(events_fh, "done", reason="ok", cases=len(cases))




def _validate_child_submission(sub_root, logs, guard):
    from triton_eval.submission import discover
    from triton_eval.validator import ValidationError_, ValidationResult, validate_static, validate_load

    validation = ValidationResult(sub_root.name)
    with _capture(logs / "load.log"):
        from triton_eval.safeexec import guard_call

        try:
            guard_call(validate_static, sub_root, validation)
        except Exception as exc:  # noqa: BLE001
            validation.errors.append(
                ValidationError_("SUB-LOD-001", "load", "-",
                                 f"提交无法校验：{type(exc).__name__}: {exc}")
            )
        try:
            if not any(e.stage == "static" for e in validation.errors):
                discovered = discover(sub_root)
                validate_load(discovered, validation)
        except Exception as exc:  # noqa: BLE001
            validation.errors.append(
                ValidationError_("SUB-LOD-001", "load", "-",
                                 f"提交无法载入：{type(exc).__name__}: {exc}")
            )

    # import 期篡改：整个提交的判定权已不可信，整份拒绝
    tampered = guard.verify()
    if tampered:
        guard.restore()
        validation.errors.append(
            ValidationError_(
                "SUB-EXE-005", "exec", "-",
                f"提交在 import 期替换了关键接口：{', '.join(tampered[:8])}"
                "——oracle 与计时链路的判定权不再可信",
            )
        )

    try:
        discovered = discover(sub_root)
        declared = discovered.operator_names()
    except Exception as exc:  # noqa: BLE001
        declared = []
        validation.errors.append(
            ValidationError_("SUB-LOD-001", "load", "-",
                             f"提交清单不可用：{type(exc).__name__}: {exc}")
        )
        discovered = None

    return validation, discovered, declared


def _load_child_submission(discovered, validation, logs):
    from triton_eval.safeexec import guard_call
    from triton_eval.submission import load_all
    from triton_eval.validator import ValidationError_

    sub = None
    load_failures: dict[str, str] = {}
    if discovered is not None and validation.ok:
        with _capture(logs / "load.log"):
            try:
                sub = guard_call(load_all, discovered)
                load_failures = dict(sub.failures)
            except Exception as exc:  # noqa: BLE001
                validation.errors.append(
                    ValidationError_("SUB-LOD-001", "load", "-",
                                     f"提交无法载入：{type(exc).__name__}: {exc}")
                )

    return sub, load_failures


def _emit_child_submission(events_fh, sub, sub_root, validation, declared, load_failures, guard):
    _emit(
        events_fh, "meta",
        submit_id=sub.submit_id if sub else sub_root.name,
        validation_ok=validation.ok,
        errors=[{
            "rule_id": e.rule_id, "stage": e.stage, "operator": e.operator,
            "detail": e.detail, "lineno": e.lineno,
        } for e in validation.errors],
        declared=declared,
        operators=sorted(sub.loaded) if sub else [],
        load_failures=load_failures,
        api_guard_covered=len(guard.covered),
    )



def _verify_child_environment(run_dir, events_fh):
    from triton_eval.integrity import check_identity, module_identity

    env_path = run_dir / "env.json"
    expected_env = {}
    if env_path.is_file():
        try:
            expected_env = json.loads(env_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    mismatches = check_identity(expected_env)
    _emit(events_fh, "env_check", ok=not mismatches,
          mismatches=mismatches, identity=module_identity())
    if mismatches:
        _emit(events_fh, "fatal", stage="env",
              detail="子进程加载的模块与环境指纹不一致：" + "; ".join(mismatches))
        return False

    return True


def _execute_child_case(case, args, sub, logs, profile, guard, run_dir, manifest, corpus_dir):
    from triton_eval.harness.npu_exec import CaseMeasurement, execute_case
    from triton_eval.runner import execute_kwargs

    loaded = sub.loaded.get(case.operator)
    impl = loaded.func if loaded else None
    log_path = logs / f"{case.code}.log"

    # 单用例的评测侧异常必须只废掉本用例，不能杀死子进程。
    # 2026-10-13 真机实测：MoE 的参考把 row_idx 建在 CPU 上（缺
    # device=），与 NPU 实现输出比对时抛设备不匹配——异常从比对段
    # 逃出 execute_case，子进程直接死亡，后续 15 个用例全部被补记
    # RUNTIME_FAIL。缺陷半径必须是单用例：本用例记 ORACLE_FAIL，
    # 其余照跑。（参考侧缺陷已修；这是结构性的兜底，同类问题不再
    # 有整轮陪葬的爆炸半径。）
    try:
        with _capture(log_path):
            m = execute_case(
                case,
                # 参数组装统一走 runner，避免与 CLI/测试各写一份而分叉
                **execute_kwargs(case, impl),
                measure_perf=profile.measure_perf,
                repeats=profile.repeats,
                warmup_ms=profile.warmup_ms,
                rep_ms=profile.rep_ms,
                log_path=str(log_path),
                # 实现返回后、比对开始前查一次篡改：
                # 比对链路（compare 用的张量方法）不允许在污染状态下运行
                tamper_check=guard.verify,
                **(dict(compilation_dir=str(run_dir / 'compilation' / case.code),
                        evidence_identity=dict(manifest_hash=manifest.manifest_hash, run_id=args.run_id))
                   if (corpus_dir / 'coverage-plan.json').is_file() else {}),
            )
    except Exception as exc:  # noqa: BLE001
        m = CaseMeasurement(case_code=case.code, operator=case.operator,
                            log_path=str(log_path))
        m.oracle_ok = False
        m.duration_ms = 0.0
        m.compile_err = f"评测侧异常: {type(exc).__name__}: {exc}"[:2000]
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n[eval] {case.code} -> 评测侧异常被逐用例兜底："
                     f"{type(exc).__name__}\n")

    # 用例级篡改核查（含性能计时段）：检出即恢复并作废本用例
    tampered = guard.verify()
    if tampered:
        guard.restore()
        m.output_ok = False
        m.output_err = (
            f"SUB-EXE-005: 关键接口在用例执行期间被替换："
            f"{', '.join(tampered[:8])}"
        )
        m.compile_err = m.output_err

    return m, log_path


def _record_child_case(m, case, args, table, jsonl_path, log_path, events_fh):
    from triton_eval.bench.stats import summarize
    from triton_eval.judge import build_record, decide, tolerance_for_case

    status = decide(m, tolerance_for_case(table, case, m))
    rec = build_record(args.run_id, m, status,
                       summarize(m.samples_ms) if m.samples_ms else None)

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        jf.write(rec.model_dump_json() + "\n")
    # 用例日志的结构化尾注：不打开每个 log 也能从文件里 grep 出终态
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(
            f"\n[eval] {case.code} -> {status.value}"
            f" duration={m.duration_ms:.0f}ms"
            + (f" err={m.compile_err[:200]}" if m.compile_err else "")
            + "\n"
        )
    _emit(events_fh, "case", record=json.loads(rec.model_dump_json()))
    if not m.executed or not m.oracle_ok:
        try:
            import torch

            torch.npu.synchronize()
        except Exception as exc:
            _emit(events_fh, "fatal", stage="env",
                  detail=f"设备上下文在 {case.code} 后不可用：{type(exc).__name__}: {exc}"[:2000])
            return False

    return True


if __name__ == "__main__":
    raise SystemExit(main())
