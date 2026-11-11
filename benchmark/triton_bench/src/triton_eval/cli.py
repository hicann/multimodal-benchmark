# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""命令行入口。

子命令：
    import-corpus   把冻结清单导入数据库
    ingest          解包一个 zip 提交并校验（不执行 kernel）
    run             执行一轮评测（需要昇腾主机；提交代码在子进程里执行）
    score           对一次运行打分，支持重算
    report          生成报告
    import          从 results.jsonl 重建数据库记录
    compare         两轮对比：识别功能回归/修复与性能回归/提升

所有不需要 NPU 的子命令（import-corpus / ingest / score / report /
import / compare）都可以直接在本地执行——这是 results.jsonl 契约带来的能力。
"""

from __future__ import annotations

from collections import namedtuple
import json
import os
import shlex
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any
import logging

import argparse


from triton_eval import __version__
from triton_eval.corpusmod import load_manifest, read_result_lines, select_cases, verify_corpus_files
from triton_eval.models import CaseResultRecord, CaseStatus, is_counted
from triton_eval.report import build_context, dump_env, write_report
from triton_eval.scorer import compute_run_score
from triton_eval.specload import load_spec
from triton_eval.store import Store
from triton_eval.console import emit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "corpus" / "v0.5-cov20-frozen"  # 2026-10-13 起现行语料（25 算子，含融合）
DEFAULT_DB = PROJECT_ROOT / "runs" / "triton-eval.db"
DEFAULT_RUNS = PROJECT_ROOT / "runs"
DEFAULT_INGEST = DEFAULT_RUNS / "ingest"


def _comparable_speedups(previous, current):
    if not (previous.speedup):
        return False
    if not (current.speedup):
        return False
    if not (previous.speedup > 0):
        return False
    return bool(current.speedup > 0)


def _store(args: argparse.Namespace) -> Store:
    s = Store(args.db)
    s.create_schema()
    return s


def _verify_corpus_or_fail(corpus_dir: Path) -> Any:
    """校验冻结完整性。失败时给出清晰原因并返回非 0，而不是抛出调用栈。"""
    manifest = load_manifest(corpus_dir / "manifest.json")
    verify_corpus_files(corpus_dir, manifest)
    return manifest


# ---------- import-corpus ----------


def cmd_import_corpus(args: argparse.Namespace) -> int:
    corpus_dir = Path(args.corpus)
    try:
        manifest = _verify_corpus_or_fail(corpus_dir)
    except (ValueError, FileNotFoundError) as exc:
        emit(f"测试集校验失败：{exc}", file=sys.stderr)
        return 1
    with _store(args) as s:
        corpus_id = s.import_corpus(manifest)
    emit(f"已导入测试集 {manifest.version}：算子 {len(manifest.operators)} 个，"
                 f"用例 {len(manifest.cases)} 个（corpus_version_id={corpus_id}）")
    return 0


# ---------- run ----------


def cmd_run(args: argparse.Namespace) -> int:
    """执行一轮评测（需要昇腾主机）。

    父进程（本函数）只做安全的事：解包、静态校验、落库、监督。
    提交代码的加载与执行全部在子进程 `evalchild` 里——不可信代码
    与数据库、评测主进程物理隔离。崩溃与超时（SIGTERM → 宽限 →
    SIGKILL）也由本进程兜底：SIGALRM 是可被吞掉的软超时。
    """
    from types import SimpleNamespace

    spec = load_spec()
    spec.profile(args.profile)
    corpus_dir = Path(args.corpus)
    try:
        manifest = _verify_corpus_or_fail(corpus_dir)
    except (ValueError, FileNotFoundError) as exc:
        emit(f"测试集校验失败：{exc}", file=sys.stderr)
        return 1
    run_dir = Path(args.runs) / f"run-{_next_run_hint(Path(args.runs))}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ingested = _ingest_run_submission(args)
    if ingested is None:
        return 1
    sub_root, package = ingested
    with _store(args) as s:
        context = SimpleNamespace(args=args, s=s, manifest=manifest, corpus_dir=corpus_dir,
                                  run_dir=run_dir, sub_root=sub_root, package=package)
        _prepare_parent_run(context)
        if _reject_parent_validation(context):
            return 1
        _prepare_run_supervisor(context)

        def on_event(evt: dict) -> None:
            _handle_parent_event(context, evt)

        return _supervise_parent_run(context, on_event)


def _merge_validation(static: Any, meta_evt: dict) -> list[Any]:
    """静态结论 + 子进程校验结论合并去重（同一规则同一算子只报一次）。"""
    seen: set[tuple] = set()
    out: list[Any] = []
    for e in static.errors:
        key = (e.rule_id, e.operator)
        if key not in seen:
            seen.add(key)
            out.append(e)
    for e in meta_evt.get("errors") or []:
        key = (e.get("rule_id"), e.get("operator"))
        if key not in seen:
            seen.add(key)
            from triton_eval.validator import ValidationError_

            out.append(ValidationError_(
                e.get("rule_id", "SUB-LOD-001"), e.get("stage", "load"),
                e.get("operator", "-"), e.get("detail", ""),
                e.get("lineno"),
            ))
    return out


def _child_cmd() -> list[str]:
    """评测子进程命令。TRITON_EVAL_CHILD_CMD 可覆盖（测试/调试用）。"""
    override = os.environ.get("TRITON_EVAL_CHILD_CMD")
    if override:
        return shlex.split(override)
    return [sys.executable, "-m", "triton_eval.evalchild"]


def _finalize_run(
    *,
    s: Store,
    run_id: int,
    run_dir: Path,
    cases: list[Any],
    completed: list[str],
    fatal: dict[str, Any] | None,
    outcome: Any,
    done_reason: str | None,
    child_validation_ok: bool,
    probe_ok: bool,
    record_inserted: bool,
    discovered: Any,
    before_snapshot: dict[Path, bytes],
    env_check: dict[str, Any],
    record_kwargs_factory: Any,
) -> None:
    """收尾四件事：提交记录兜底、缺失用例补记、终态落库、完整性档案。

    任何路径都必须走到产物落盘（integrity.json / artifacts.json）——
    一轮被作废的评测同样需要留痕：探针失败的运行当初就是因为
    只留了一句 INVALID、什么都查不到。
    """
    # 子进程没到 meta 事件就死了：提交记录仍要落（身份留痕原则）
    if not record_inserted:
        detail = (f"子进程未产出校验结论：exit={outcome.exit_code}"
                  + (f" 信号 {outcome.signal_name}" if outcome.signal_name else ""))
        kwargs = record_kwargs_factory(
            validation_ok=False,
            validation=[{
                "rule_id": "SUB-LOD-001", "stage": "load", "operator": "-",
                "detail": detail, "lineno": None,
            }],
        )
        record_id = s.insert_submission_record(**kwargs)
        s.set_run_submission_record(run_id, record_id)

    status = "OK"
    reason: str | None = None

    if not child_validation_ok:
        status, reason = "INVALID", done_reason or "提交校验未通过（子进程）"
    elif not probe_ok:
        status, reason = "INVALID", done_reason or "环境健康探针失败"
    else:
        status, reason = _complete_child_run(
            s, run_id, run_dir, cases, completed, fatal, outcome)

    s.finish_run(run_id, status, reason)

    # ---------- 提交不可变核查（评测期间目录不得变化） ----------
    changed = _changed_submission(discovered, before_snapshot)

    # ---------- 完整性档案 + 产物清单 ----------
    _write_run_artifacts(run_dir, run_id, status, env_check, changed, outcome)


def _complete_child_run(s, run_id, run_dir, cases, completed, fatal, outcome):
    status, reason = "OK", None
    child_failed = (
        fatal is not None
        or outcome.exit_code not in (0, None)
        or outcome.killed_by_us
    )
    missing = [c for c in cases if c.code not in completed]

    if child_failed and missing:
        _fill_missing_results(s, run_id, run_dir, missing, fatal, outcome)

    if fatal is not None:
        status = "PARTIAL"
        reason = f"子进程致命错误（{fatal.get('stage')}）：{fatal.get('detail')}"[:2000]
    elif child_failed:
        # 崩溃 / 被杀的运行一律 PARTIAL：用例结果已被补记完整，
        # 但其中含合成记录——「非自然完成」这一事实必须可辨
        status = "PARTIAL"
        reason = ("评测子进程被监督者终止（硬超时：SIGTERM→SIGKILL）"
                  if outcome.killed_by_us
                  else f"评测子进程异常退出：exit={outcome.exit_code}"
                  + (f" 信号 {outcome.signal_name}" if outcome.signal_name else ""))
    elif missing:
        # 子进程正常退出却没跑完（防御分支：不该发生）
        status = "PARTIAL"
        reason = f"子进程正常结束但有 {len(missing)} 个用例无结果"

    return status, reason


def _fill_missing_results(s, run_id, run_dir, missing, fatal, outcome):
    # 补记缺失用例：没有结果的用例不能静默消失。
    # 归因：环境类 fatal 记 ENV_ABORT（不怪被测方）；超时 kill /
    # 崩溃信号记 RUNTIME_FAIL——SIGALRM 被吞后，监督者杀进程是
    # 唯一可靠的超时手段（ADV-F4 的教训）。
    if outcome.killed_by_us:
        diag = "评测子进程被硬超时终止（SIGTERM→SIGKILL），用例未完成"
    elif fatal is not None:
        diag = f"子进程致命错误（{fatal.get('stage')}）：{fatal.get('detail')}"[:2000]
    elif outcome.signal_name:
        diag = f"子进程被信号终止：{outcome.signal_name}"
    else:
        diag = f"子进程异常退出：exit={outcome.exit_code}"
    fill = CaseStatus.ENV_ABORT if (fatal or {}).get("stage") == "env" \
        else CaseStatus.RUNTIME_FAIL
    for case in missing:
        rec = CaseResultRecord(
            run_id=run_id, case_code=case.code, operator=case.operator,
            status=fill, compile_err=diag,
            log_path=str(run_dir / "logs" / f"{case.code}.log"),
        )
        try:
            s.insert_case_result(rec)
            with open(run_dir / "results.jsonl", "a", encoding="utf-8") as fh:
                fh.write(rec.model_dump_json() + "\n")
            emit(f"  {case.code}: {fill.value}（补记：子进程未产出）")
        except Exception as exc:  # noqa: BLE001
            emit(f"  {case.code}: 补记失败 {exc}", file=sys.stderr)


def _changed_submission(discovered, before_snapshot):
    changed: list[str] = []
    if discovered is not None and before_snapshot:
        from triton_eval.submission import verify_unchanged

        try:
            changed = verify_unchanged(before_snapshot, discovered.snapshot())
        except Exception:  # noqa: BLE001
            changed = ["<快照比对失败>"]
        if changed:
            emit(f"警告：提交目录在评测期间被改动（{len(changed)} 个文件）：\n  "
                         + "\n  ".join(changed[:10]), file=sys.stderr)

    return changed


def _write_run_artifacts(run_dir, run_id, status, env_check, changed, outcome):
    from datetime import datetime, timezone

    integrity = {
        "run_status": status,
        "env_identity": env_check.get("identity"),
        "env_mismatches": env_check.get("mismatches"),
        "submission_changed_during_run": changed,
        "child": {
            "exit_code": outcome.exit_code,
            "signal": outcome.signal_name,
            "killed_by_us": outcome.killed_by_us,
            "stderr_tail": outcome.stderr_tail[-10:],
            "unparsed_lines": outcome.unparsed_lines[:20],
        },
    }
    (run_dir / "integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from triton_eval.package import sha256_file

    files = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "artifacts.json":
            files.append({
                "path": str(p.relative_to(run_dir)),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    (run_dir / "artifacts.json").write_text(
        json.dumps({
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _snapshot_fp(snapshot: dict[Path, bytes]) -> str:
    """提交内容的指纹。用于事后确认「评测的就是当时那一份」。

    快照比目录名可靠：目录名不变、内容被偷偷改过的情况无法从名字看出。
    """
    import hashlib

    h = hashlib.sha256()
    for path in sorted(snapshot):
        h.update(str(path).encode("utf-8"))
        h.update(snapshot[path])
    return h.hexdigest()


def _load_and_validate(sub_root: Path):
    """载入并校验一份提交，返回 (sub, validation, declared, fp, load_failures)。

    任何一步失败都不抛出：提交有问题不是「评测崩溃」的理由，
    必须转成结构化的校验结论落库并披露。
    """
    from triton_eval.submission import Submission, discover, load_all
    from triton_eval.validator import ValidationError_, validate_submission

    # 校验本身也要走 guard：SystemExit 穿透时，下面那个 try 已经来不及了
    from triton_eval.safeexec import guard_call, is_escape

    try:
        validation = guard_call(validate_submission, sub_root)
    except Exception as exc:  # noqa: BLE001
        from triton_eval.validator import ValidationError_, ValidationResult

        validation = ValidationResult("")
        validation.errors.append(
            ValidationError_("SUB-LOD-001", "load", "-",
                             f"提交无法校验：{type(exc).__name__}: {exc}")
        )

    declared: list[str] = []
    snapshot_fp = ""
    load_failures: dict[str, str] = {}
    sub = None

    try:
        discovered = discover(sub_root)
        declared = discovered.operator_names()
        snapshot_fp = _snapshot_fp(discovered.snapshot())
        # 静态/加载已失败的提交不再重复加载：短路，省一次无谓的导入
        if validation.ok:
            sub = load_all(discovered)
            load_failures = dict(sub.failures)
        else:
            sub = Submission(sub_root, discovered.manifest)
    except Exception as exc:  # noqa: BLE001
        validation.errors.append(
            ValidationError_("SUB-LOD-001", "load", "-",
                             f"提交无法载入：{type(exc).__name__}: {exc}")
        )
    # SystemExit 不是 Exception，上面的兜底拦不住——
    # 必须补一层，否则提交代码一个 sys.exit() 就能让进程静默消失（退出码 0）。
    # 转成校验错误而不是抛出：这不是评测崩溃，是「这份提交越界了」，
    # 应当像其它违规一样被记录并披露。
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise  # 操作者的中断意图，不是提交方行为
        validation.errors.append(
            ValidationError_("SUB-LOD-001", "load", "-",
                             f"提交试图终止进程：{type(exc).__name__}: {exc}")
        )
    return sub, validation, declared, snapshot_fp, load_failures


@lru_cache(maxsize=64)
def _op_by_name(name: str):
    from triton_eval.operator_registry import by_name

    return by_name(name)


def _next_run_hint(runs_dir: Path) -> int:
    if not runs_dir.exists():
        return 1
    existing = [p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    return len(existing) + 1


def _resolve_run_id(s: Store, value: str) -> int | None:
    """解析 run 标识。支持 `latest`，免得手动数 id。

    注意：`latest` 取的是数据库里编号最大的运行，不保证它有结果——
    探针失败的运行也会被计入。
    """
    if value == "latest":
        row = s.conn.execute(
            "SELECT id FROM eval_run ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return int(row["id"]) if row else None
    return int(value)


def _resolve_run_id_with_results(s: Store, value: str) -> int | None:
    """取最近一次**有结果**的运行。评分与报告真正想要的是这个。"""
    if value == "latest":
        row = s.conn.execute(
            "SELECT r.id FROM eval_run r "
            "JOIN case_result c ON c.run_id = r.id "
            "ORDER BY r.id DESC LIMIT 1"
        ).fetchone()
        return int(row["id"]) if row else None
    return int(value)


# ---------- validate ----------


def cmd_validate(args: argparse.Namespace) -> int:
    """独立校验一份提交，不执行任何 kernel。

    存在的理由：提交方需要**在跑评测之前**就知道自己合不合规。
    曾经只能靠 `run` 整轮跑一遍才看得到结论，而一轮要几十分钟，
    且失败信息埋在评测输出里。校验与执行必须能分开做。
    """
    sub_root = Path(args.submission)
    _sub, validation, declared, _fp, load_failures = _load_and_validate(sub_root)

    emit(validation.render())
    if declared:
        emit(f"\n声明算子（{len(declared)}）：{', '.join(sorted(declared))}")
    if load_failures:
        emit("\n加载失败（不影响其余算子）：")
        for name in sorted(load_failures):
            emit(f"  {name}: {load_failures[name]}")

    if args.rules:
        emit("\n规则清单：")
        from triton_eval.validator import list_rules

        for rule_id, desc in list_rules():
            emit(f"  {rule_id:<12} {desc}")

    return 0 if validation.ok else 1


# ---------- score ----------


def cmd_score(args: argparse.Namespace) -> int:
    spec = load_spec()
    with _store(args) as s:
        run_id = _resolve_run_id_with_results(s, args.run)
        if run_id is None:
            emit(f"运行 {args.run} 没有用例结果（可能该轮被判 INVALID）", file=sys.stderr)
            return 1
        results = s.case_results(run_id)
        if not results:
            emit(f"运行 {run_id} 没有用例结果", file=sys.stderr)
            return 1
        weights = _load_weights(s, run_id)
        # 按算子锚点的性能分：全对才出分（见 anchors.py 的死锁论证）
        op_scores, op_reasons = _perf_op_view(s, run_id, results, weights)

        score = compute_run_score(
            run_id=run_id, results=results, weights=weights,
            thresholds=spec.fast_thresholds,
            # 用解析后的 run_id，不能用 args.run（可能是 "latest"）
            cv_gate=spec.profile(_profile_of(s, run_id)).cv_gate,
            scorer_version=args.scorer_version or spec.scorer_version,
            op_scores=op_scores,
        )
        _set_score_groups(score, results, weights)
        s.insert_run_score(score)

        # 覆盖率必须与分数同时出现：
        # 分母只含「提交了的算子」，只交 2 个也能拿 100 分——
        # 不披露覆盖率，100 分就会被误读成「全部 14 个算子都通过了」。
        ops = manifest_operators(s, run_id)
        covered = sorted({r.operator for r in results if is_counted(r.status)})
        counted_cases = sum(1 for r in results if is_counted(r.status))

        # 语言特性覆盖率：分母 = 本轮题库可及的原子清单（yaml active），
        # 两个口径：题库静态覆盖（出了多少种考法）与本轮达成
        # （计入 PASS 的用例标签并集——没跑过/没跑对的考法不算达成）。
        fa = _feature_atoms()
        fa_total = len(fa)
        if fa_total:
            corpus_atoms, pass_atoms = _fa_atoms(s, run_id)
            fa_corpus = len(corpus_atoms & fa)
            fa_run = len(pass_atoms & fa)

    emit(f"功能主得分 OpScore = {score.coverage_ops_core:.2f}")
    emit(f"严格通过率        = {score.coverage_strict:.2f}")
    emit(f"静默错误数        = {score.severe_count}")
    emit(f"提交覆盖          = {len(covered)}/{len(ops)} 个算子"
                 f"（{counted_cases}/{len(results)} 个用例计入分母）")
    if fa_total:
        emit(f"语言特性（题库）  = {fa_corpus}/{fa_total}"
                     f"（{fa_corpus / fa_total:.0%}，FA 清单见 spec/feature_atoms.yaml）")
        emit(f"语言特性（本轮）  = {fa_run}/{fa_total}（计入 PASS 的用例标签并集）")
    if covered and len(covered) < len(ops):
        emit("  未覆盖：" + ", ".join(sorted(set(ops) - set(covered))))
    _emit_performance_and_groups(score, op_scores, op_reasons)
    return 0


def _set_score_groups(score, results, weights):
    from triton_eval.operator_registry import by_name as _op_by_name
    from triton_eval.scorer import coverage_by_category, coverage_by_level

    def _cat(name: str) -> str | None:
        op = _op_by_name(name)
        return op.category.value if op else None

    def _lvl(name: str) -> str | None:
        op = _op_by_name(name)
        return op.level.value if op else None

    score.by_category = coverage_by_category(results, weights, _cat)
    score.by_level = coverage_by_level(results, weights, _lvl)


def _emit_performance_and_groups(score, op_scores, op_reasons):
    if score.perf_score is not None:
        scored_ops = [op for op, v in op_scores.items() if v is not None]
        emit(f"性能得分（按算子加权）= {score.perf_score:.2f}"
                     f"（{len(scored_ops)}/{len(op_scores)} 个算子出分）")
    elif score.perf_geo is not None:
        emit(f"性能得分          = 不可用（无算子达到出分条件，"
                     f"P_geo={score.perf_geo:.4f} 仅报裸值）")
    else:
        emit("性能得分          = 不可用（本轮无性能数据；冒烟档不测性能）")
    if op_scores:
        emit("按算子性能分：")
        for op in sorted(op_scores):
            value = op_scores[op]
            shown = f"{value:.2f}" if value is not None else "—"
            emit(f"  {op:<18} {shown:>8}  {op_reasons[op]}")
    for dim, title in (("by_category", "按类别"), ("by_level", "按难度")):
        groups = getattr(score, dim, None)
        if not groups:
            continue
        emit(f"\n{title}：")
        for key in sorted(groups):
            value = groups[key]
            emit(f"  {key:<20} {'无数据' if value is None else f'{value:.2f}'}")


def manifest_operators(s: Store, run_id: int) -> list[str]:
    """该轮所用 corpus 声明的全部算子名，用于披露提交覆盖率。"""
    rows = s.conn.execute(
        "SELECT o.name FROM operator o "
        "JOIN corpus_version c ON c.id = o.corpus_version_id "
        "JOIN eval_run r ON r.corpus_version_id = c.id WHERE r.id = ?",
        (run_id,),
    ).fetchall()
    return [r["name"] for r in rows]


def _feature_atoms() -> set[str]:
    """FA 清单的 active 原子集合（reserved 不进分母——见 yaml 的诚实性原则）。"""
    from triton_eval.specload import load_feature_atoms

    fa = load_feature_atoms()
    return {a["id"] for a in fa.get("atoms", [])}


def _fa_atoms(s: Store, run_id: int) -> tuple[set[str], set[str]]:
    """(题库原子并集, 计入 PASS 的用例原子并集)。标签存于 eval_case 表。"""
    base = (
        "SELECT c.feature_atoms_json FROM eval_case c "
        "JOIN operator o ON o.id = c.operator_id "
        "JOIN corpus_version cv ON cv.id = o.corpus_version_id "
        "JOIN eval_run r ON r.corpus_version_id = cv.id WHERE r.id = ?"
    )
    corpus_atoms: set[str] = set()
    for row in s.conn.execute(base, (run_id,)).fetchall():
        corpus_atoms.update(json.loads(row["feature_atoms_json"] or "[]"))
    pass_atoms: set[str] = set()
    for row in s.conn.execute(
        base + " AND c.id IN (SELECT case_id FROM case_result "
               "WHERE run_id = ? AND status = 'PASS')",
        (run_id, run_id),
    ).fetchall():
        pass_atoms.update(json.loads(row["feature_atoms_json"] or "[]"))
    return corpus_atoms, pass_atoms


def _load_weights(s: Store, run_id: int) -> dict[str, float]:
    rows = s.conn.execute(
        "SELECT o.name, o.weight FROM operator o "
        "JOIN corpus_version c ON c.id = o.corpus_version_id "
        "JOIN eval_run r ON r.corpus_version_id = c.id WHERE r.id = ?",
        (run_id,),
    ).fetchall()
    return {r["name"]: r["weight"] for r in rows}


def _profile_of(s: Store, run_id: int) -> str:
    run = s.get_run(run_id)
    return run["profile"] if run else "smoke"


def _perf_op_view(
    s: Store, run_id: int, results: list[Any], weights: dict[str, float]
) -> tuple[dict[str, float | None], dict[str, str]]:
    """按算子的性能分视图（打分与报告共用，保证两处口径一致）。

    尺子按算子持有（死锁论证见 anchors.py 模块注释）：某算子本轮
    未全对 → 该算子无分，不影响其他算子。
    """
    from triton_eval.anchors import op_corpus_cases, resolve_op_anchors, score_per_operator

    run = s.get_run(run_id)
    corpus_id = run["corpus_version_id"] if run else None
    if corpus_id is None:
        return {}, {}
    op_cases = op_corpus_cases(s, corpus_id)
    anchors = resolve_op_anchors(s, corpus_id)
    return score_per_operator(results, anchors, op_cases, weights)


# ---------- report ----------


def cmd_report(args: argparse.Namespace) -> int:
    spec = load_spec()
    with _store(args) as s:
        run_id = _resolve_run_id_with_results(s, args.run)
        if run_id is None:
            emit(f"运行 {args.run} 没有用例结果（可能该轮被判 INVALID）", file=sys.stderr)
            return 1
        run = s.get_run(run_id)
        if run is None:
            emit(f"未找到运行 {run_id}", file=sys.stderr)
            return 1
        env = json.loads(run["env_json"] or "{}")
        sub = s.conn.execute(
            "SELECT * FROM compiler_submission WHERE id=?", (run["submission_id"],)
        ).fetchone()
        corpus = s.conn.execute(
            "SELECT * FROM corpus_version WHERE id=?", (run["corpus_version_id"],)
        ).fetchone()
        record_id = run["submission_record_id"] if "submission_record_id" in run.keys() else None
        sub_record = s.submission_record(record_id) if record_id else None
        results = s.case_results(run_id)
        scores = s.run_scores(run_id)
        score = scores[-1] if scores else None

        # 按算子性能分视图：与 score 命令同一口径重算，报告口径不漂移
        weights = _load_weights(s, run_id)
        op_scores, op_reasons = _perf_op_view(s, run_id, results, weights)
        op_perf = {
            op: {"score": op_scores[op], "reason": op_reasons[op]}
            for op in sorted(op_scores)
        }

        ctx = build_context(
            run=dict(run), env=env, submission=dict(sub) if sub else {},
            results=results, score=score,
            submission_record=sub_record,
            total_operators=len(manifest_operators(s, run_id)),
            corpus_version=corpus["version"] if corpus else "unknown",
            spec_version=spec.spec_version,
            scorer_version=score["scorer_version"] if score else spec.scorer_version,
            run_dir=run.get("run_dir") or str(DEFAULT_RUNS / f"run-{run_id}"),
            op_perf=op_perf,
        )
        out = write_report(ctx, Path(args.out), f"report-run-{run_id}")

    emit(f"报告已生成：{out['markdown']}")
    emit(f"            {out['html']}")
    return 0


# ---------- import ----------


def cmd_import(args: argparse.Namespace) -> int:
    with _store(args) as s:
        if s.get_run(args.run) is None:
            emit(f"未找到运行 {args.run}，请先创建运行记录", file=sys.stderr)
            return 1
        records = read_result_lines(Path(args.jsonl))
        count = 0
        for rec in records:
            rec.run_id = args.run
            s.insert_case_result(rec)
            count += 1
    emit(f"已从 {args.jsonl} 导入 {count} 条结果到运行 {args.run}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """列出一次运行中所有非 PASS 的用例及其误差量级。

    用于区分「容差过严」与「实现真的错了」——两者都表现为非 PASS，
    但误差量级差好几个数量级。
    """
    with _store(args) as s:
        run_id = _resolve_run_id(s, args.run)
        if run_id is None:
            emit("没有找到任何运行", file=sys.stderr)
            return 1
        rows = s.conn.execute(
            "SELECT c.code, r.status, r.max_rel_err, r.max_abs_err, r.compile_err "
            "FROM case_result r JOIN eval_case c ON c.id = r.case_id "
            "WHERE r.run_id = ? AND r.status <> 'PASS' ORDER BY c.code",
            (run_id,),
        ).fetchall()

    if not rows:
        emit(f"运行 {run_id}：全部 PASS")
        return 0

    emit(f"运行 {run_id} 非 PASS 用例共 {len(rows)} 个：")
    emit(f"{'用例':<22}{'终态':<16}{'rel 误差':>14}{'abs 误差':>14}  线索")
    emit("-" * 90)
    for r in rows:
        rel = f"{r['max_rel_err']:.3e}" if r["max_rel_err"] is not None else "-"
        ab = f"{r['max_abs_err']:.3e}" if r["max_abs_err"] is not None else "-"
        hint = (r["compile_err"] or "")[:40]
        emit(f"{r['code']:<22}{r['status']:<16}{rel:>14}{ab:>14}  {hint}")
    return 0


def cmd_rescore_from_jsonl(args: argparse.Namespace) -> int:
    """只凭 jsonl 完成打分与报告，全程不执行任何 kernel。

    离线重算拿不到题库的历史轮次数据（只有这一份 jsonl），解析不出
    按算子锚点 → 性能分记不可用（None），只报 P_geo 裸值。
    需要锚点口径时用 score 命令（数据库里有历史轮次）。
    """
    spec = load_spec()
    results = read_result_lines(Path(args.jsonl))
    for r in results:
        r.run_id = args.run
    score = compute_run_score(
        run_id=args.run, results=results, weights={},
        thresholds=spec.fast_thresholds, cv_gate=0.05,
        scorer_version=spec.scorer_version,
    )
    emit(json.dumps(json.loads(score.model_dump_json()), ensure_ascii=False, indent=2))
    return 0


# ---------- compare ----------


def _run_identity(s: Store, run_id: int) -> dict[str, Any]:
    """一轮的身份信息（编译器、环境、题库），供对比报告的头部使用。"""
    run = s.get_run(run_id)
    sub = s.conn.execute(
        "SELECT * FROM compiler_submission WHERE id=?", (run["submission_id"],)
    ).fetchone()
    corpus = s.conn.execute(
        "SELECT * FROM corpus_version WHERE id=?", (run["corpus_version_id"],)
    ).fetchone()
    env = json.loads(run["env_json"] or "{}")
    return {
        "run_id": run_id,
        "profile": run["profile"],
        "status": run["status"],
        "started_at": run["started_at"],
        "compiler": f"{sub['name']} {sub['version']}" if sub else "?",
        "git_commit": (sub["git_commit"] or "") if sub else "",
        "env_fp": run.get("env_fp") or "",
        "npu_model": env.get("npu_model") or "?",
        "clock_locked": bool(env.get("clock_locked")),
        "corpus_version": corpus["version"] if corpus else "?",
    }


def _fmt(value: Any, spec: str = "") -> str:
    """分数表单元格：None 显示 —（不可用），绝不显示 0 充数。"""
    if value is None:
        return "—"
    return format(value, spec) if spec else str(value)


def cmd_compare(args: argparse.Namespace) -> int:
    """两轮对比：识别功能回归/修复与性能回归/提升。

    口径：题库必须同版本（题不一样没法比）；环境指纹不同只警告不阻止
    ——结论是否可信由读者结合指纹判断；加速比变化超过规范门槛
    （compare.speedup_change_gate，默认 10%）才算显著变化。
    """
    spec = load_spec()
    gate = spec.speedup_change_gate
    with _store(args) as s:
        new_id = _resolve_run_id_with_results(s, args.run)
        old_id = _resolve_run_id_with_results(s, args.against)
        if new_id is None or old_id is None:
            emit("两轮都必须有用例结果才能对比", file=sys.stderr)
            return 1
        if new_id == old_id:
            emit("--run 与 --against 不能是同一轮", file=sys.stderr)
            return 1

        new_ident = _run_identity(s, new_id)
        old_ident = _run_identity(s, old_id)
        if new_ident["corpus_version"] != old_ident["corpus_version"]:
            emit(f"题库版本不同（run {new_id} 用 {new_ident['corpus_version']}，"
                         f"run {old_id} 用 {old_ident['corpus_version']}），"
                         "用例不可比；请分别出报告。", file=sys.stderr)
            return 1

        env_warning = ""
        if new_ident["env_fp"] and old_ident["env_fp"] \
                and new_ident["env_fp"] != old_ident["env_fp"]:
            env_warning = ("⚠ 两轮环境指纹不同（跨环境对比，加速比的绝对差"
                           "可能来自环境而非实现变化），结论需谨慎。")

        new_results = {r.case_code: r for r in s.case_results(new_id)}
        old_results = {r.case_code: r for r in s.case_results(old_id)}
        new_score = (s.run_scores(new_id) or [None])[-1]
        old_score = (s.run_scores(old_id) or [None])[-1]

        # ---------- 用例级变化 ----------
        changes = _comparison_changes(old_results, new_results, gate)
        func_regress, func_fix, perf_regress, perf_gain, _only_old, _only_new = changes

        # ---------- Markdown 报告 ----------
        lines = _comparison_intro(new_id, old_id, new_ident, old_ident, gate, env_warning)
        _append_comparison_scores(lines, old_score, new_score)
        _append_comparison_tables(lines, changes, gate)
        _append_comparison_summary(lines, func_regress, func_fix, perf_regress, perf_gain)
        out_path = _write_comparison(args.out, new_id, old_id, lines)

    emit(f"两轮对比：run {new_id} vs run {old_id}")
    emit(f"  功能回归 {len(func_regress)}、修复 {len(func_fix)}；"
                 f"性能显著回归 {len(perf_regress)}、提升 {len(perf_gain)}")
    if env_warning:
        emit(f"  {env_warning}", file=sys.stderr)
    emit(f"  详情：{out_path}")
    return 0


def _write_comparison(destination, new_id, old_id, lines):
    text = "\n".join(lines)

    out_dir = Path(destination)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare-run{new_id}-vs-run{old_id}.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path


ComparisonChanges = namedtuple('ComparisonChanges', 'func_regress func_fix perf_regress perf_gain only_old only_new')


def _comparison_changes(old_results, new_results, gate):
    func_regress: list[tuple[str, str, str, str]] = []
    func_fix: list[tuple[str, str, str, str]] = []
    perf_regress: list[tuple[str, str, float, float, float]] = []
    perf_gain: list[tuple[str, str, float, float, float]] = []
    only_old: list[tuple[str, str]] = []
    only_new: list[tuple[str, str]] = []
    for code in sorted(set(old_results) | set(new_results)):
        o = old_results.get(code)
        n = new_results.get(code)
        if o is None:
            only_new.append((code, n.operator if n else "?"))
            continue
        if n is None:
            only_old.append((code, o.operator))
            continue
        if o.status.value == "PASS" and n.status.value != "PASS":
            hint = (n.compile_err or "")[:60]
            func_regress.append((code, o.operator, n.status.value, hint))
        elif o.status.value != "PASS" and n.status.value == "PASS":
            hint = (o.compile_err or "")[:60]
            func_fix.append((code, o.operator, o.status.value, hint))
        if _comparable_speedups(o, n):
            delta = (n.speedup - o.speedup) / o.speedup
            row = (code, o.operator, o.speedup, n.speedup, f"{delta:+.0%}")
            if delta <= -gate:
                perf_regress.append(row)
            elif delta >= gate:
                perf_gain.append(row)

    return ComparisonChanges(func_regress, func_fix, perf_regress, perf_gain, only_old, only_new)


def _comparison_intro(new_id, old_id, new_ident, old_ident, gate, env_warning):
    from datetime import datetime, timezone

    lines: list[str] = []
    lines.append(f"# 两轮对比：run {new_id} vs run {old_id}（参照）")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- 题库版本：{new_ident['corpus_version']}（两轮一致）")
    lines.append(f"- 加速比显著变化门槛：±{gate:.0%}"
                 f"（低于该幅度视为噪声，不列入变化清单）")
    if env_warning:
        lines.append(f"- {env_warning}")
    lines.append("")
    lines.append("## 1. 两轮身份")
    lines.append("")
    lines.append("| 项 | 参照轮（against） | 本轮（run） |")
    lines.append("|---|---|---|")
    for key, title in (("run_id", "运行编号"), ("profile", "档位"),
                       ("status", "状态"), ("started_at", "开始时间"),
                       ("compiler", "编译器"), ("git_commit", "代码提交"),
                       ("env_fp", "环境指纹"), ("npu_model", "加速卡型号"),
                       ("clock_locked", "时钟锁定")):
        lines.append(f"| {title} | {old_ident[key]} | {new_ident[key]} |")
    lines.append("")
    lines.append("## 2. 分数对照")
    lines.append("")
    lines.append("| 指标 | 参照轮 | 本轮 | Δ |")
    lines.append("|---|---|---|---|")

    return lines


def _append_comparison_scores(lines, old_score, new_score):
    def _row(title: str, key: str, fmt: str = ".2f") -> None:
        o = old_score.get(key) if old_score else None
        n = new_score.get(key) if new_score else None
        delta: Any = None
        if o is not None and n is not None:
            delta = n - o
        lines.append(
            f"| {title} | {_fmt(o, fmt)} | {_fmt(n, fmt)} | "
            f"{_fmt(delta, fmt)} |"
        )

    if old_score and new_score:
        _row("功能主得分 OpScore", "coverage_ops_core")
        _row("严格通过率 OpStrictPass", "coverage_strict")
        _row("静默错误数", "severe_count", "d")
        _row("几何平均加速比 P_geo", "perf_geo", ".4f")
        _row("性能得分（按算子加权）", "perf_score")
        _row("不稳定比例", "unstable_rate")
    else:
        lines.append("| （有轮次尚未打分，请先对两轮分别执行 score） | | | |")
    lines.append("")
    lines.append("## 3. 用例级变化")
    lines.append("")


def _append_comparison_tables(lines, changes, gate):
    func_regress, func_fix, perf_regress, perf_gain, only_old, only_new = changes

    def _table(title: str, rows: list[Any], header: str) -> None:
        lines.append(f"### {title}（{len(rows)} 个）")
        lines.append("")
        if not rows:
            lines.append("无。")
        else:
            ncols = header.count("|") - 1
            lines.append(header)
            lines.append("|" + "---|" * ncols)
            for row in rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    _table("3.1 功能回归（参照轮通过 → 本轮未通过）", func_regress,
           "| 用例 | 算子 | 本轮终态 | 线索 |")
    _table("3.2 功能修复（参照轮未通过 → 本轮通过）", func_fix,
           "| 用例 | 算子 | 参照轮终态 | 当时的线索 |")
    _table(f"3.3 性能显著回归（加速比下降 ≥ {gate:.0%}）", perf_regress,
           "| 用例 | 算子 | 参照轮加速比 | 本轮加速比 | 变化 |")
    _table(f"3.4 性能显著提升（加速比上升 ≥ {gate:.0%}）", perf_gain,
           "| 用例 | 算子 | 参照轮加速比 | 本轮加速比 | 变化 |")
    _table("3.5 仅在一轮出现的用例（另一轮没跑或没结果）",
           [(c, op, "仅参照轮") for c, op in only_old]
           + [(c, op, "仅本轮") for c, op in only_new],
           "| 用例 | 算子 | 出现在 |")


def _append_comparison_summary(lines, func_regress, func_fix, perf_regress, perf_gain):
    lines.append("## 4. 结论摘要")
    lines.append("")
    if func_regress or perf_regress:
        lines.append(f"相对参照轮：功能回归 {len(func_regress)} 个、"
                     f"修复 {len(func_fix)} 个；性能显著回归 "
                     f"{len(perf_regress)} 个、提升 {len(perf_gain)} 个。"
                     "回归清单见上节，请优先核查功能回归与静默错误。")
    else:
        lines.append(f"相对参照轮：功能回归 0 个、修复 {len(func_fix)} 个；"
                     f"性能显著回归 0 个、提升 {len(perf_gain)} 个。"
                     "未发现回归。")
    lines.append("")


# ---------- ingest ----------


def cmd_ingest(args: argparse.Namespace) -> int:
    """解包一个 zip 提交并做完整校验，不执行任何 kernel。

    与 validate 的分工：validate 吃目录，ingest 吃包。
    包级规则（SUB-PKG-*）在这一步全部过一遍；解包产物是内容寻址目录，
    同一个包解两次落在同一处。
    """
    from triton_eval.package import PackageError, ingest_zip

    zip_path = Path(args.package)
    try:
        info = ingest_zip(zip_path, Path(args.dest))
    except PackageError as exc:
        # 摄取失败也要留痕：这是「包」这一层的校验结论，与内容级结论同级
        with _store(args) as s:
            s.insert_submission_record(
                submit_id=zip_path.stem, root_path="",
                operators=[], snapshot_fp="",
                validation_ok=False,
                validation=[{"rule_id": exc.rule_id, "stage": "package",
                             "operator": "-", "detail": exc.detail, "lineno": None}],
                package_sha256=exc.sha256 or None,
                package_name=zip_path.name,
            )
        emit(exc.render(), file=sys.stderr)
        return 1

    emit(f"解包 {zip_path.name} → {info.dest}")
    emit(f"sha256 {info.sha256}（内容寻址目录 {info.short_sha}）")
    if info.stripped_prefix:
        emit(f"已剥离外层目录：{info.stripped_prefix!r}")
    emit(f"文件 {len(info.files)} 个，共 {info.total_bytes} 字节")

    sub, validation, declared, fp, load_failures = _load_and_validate(info.dest)
    emit(validation.render())
    if declared:
        emit(f"\n声明算子（{len(declared)}）：{', '.join(sorted(declared))}")
    if load_failures:
        emit("\n加载失败（不影响其余算子）：")
        for name in sorted(load_failures):
            emit(f"  {name}: {load_failures[name]}")

    with _store(args) as s:
        s.insert_submission_record(
            submit_id=sub.submit_id if sub else info.dest.name,
            root_path=str(info.dest),
            operators=declared,
            snapshot_fp=fp,
            validation_ok=validation.ok,
            validation=[{
                "rule_id": e.rule_id, "stage": e.stage, "operator": e.operator,
                "detail": e.detail, "lineno": e.lineno,
            } for e in validation.errors],
            load_failures=load_failures,
            package_sha256=info.sha256,
            package_name=zip_path.name,
        )
    emit(f"\n下一步：python -m triton_eval run --submission {info.dest}")
    return 0 if validation.ok else 1


# ---------- 解析 ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="triton-eval", description="Triton 编译器评测体系（MVP）")
    p.add_argument("--version", action="version", version=f"triton-eval {__version__}")
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 数据库路径")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("import-corpus", help="把冻结清单导入数据库")
    c.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    c.set_defaults(func=cmd_import_corpus)

    _add_run_parser(sub)

    ig = sub.add_parser("ingest", help="解包一个 zip 提交并校验（不执行 kernel）")
    ig.add_argument("--package", required=True, help="提交 zip 包路径")
    ig.add_argument("--dest", default=str(DEFAULT_INGEST),
                    help="解包目标目录（内容寻址，幂等）")
    ig.set_defaults(func=cmd_ingest)

    va = sub.add_parser("validate", help="只校验提交是否合规，不执行 kernel")
    va.add_argument("--submission", required=True, help="提交目录")
    va.add_argument("--rules", action="store_true", help="同时列出规则清单")
    va.set_defaults(func=cmd_validate)

    sc = sub.add_parser("score", help="对一次运行打分（--run 可填 latest）")
    sc.add_argument("--run", required=True, help="运行编号，或 latest")
    sc.add_argument("--scorer-version", default=None)
    sc.set_defaults(func=cmd_score)

    rp = sub.add_parser("report", help="生成报告（--run 可填 latest）")
    rp.add_argument("--run", required=True, help="运行编号，或 latest")
    rp.add_argument("--out", default=str(DEFAULT_RUNS / "reports"))
    rp.set_defaults(func=cmd_report)

    im = sub.add_parser("import", help="从 results.jsonl 重建数据库记录")
    im.add_argument("--run", required=True, help="运行编号，或 latest")
    im.add_argument("--jsonl", required=True)
    im.set_defaults(func=cmd_import)

    rs = sub.add_parser("rescore", help="只凭 jsonl 重算分数（不执行 kernel）")
    rs.add_argument("--run", required=True, help="运行编号，或 latest")
    rs.add_argument("--jsonl", required=True)
    rs.set_defaults(func=cmd_rescore_from_jsonl)

    dg = sub.add_parser("diagnose", help="列出非 PASS 的用例及其误差（定位用）")
    dg.add_argument("--run", required=True, help="运行编号，或 latest")
    dg.set_defaults(func=cmd_diagnose)

    cp = sub.add_parser("compare",
                        help="两轮对比：识别功能回归/修复与性能回归/提升")
    cp.add_argument("--run", required=True, help="本轮编号，或 latest")
    cp.add_argument("--against", required=True, help="参照轮编号，或 latest")
    cp.add_argument("--out", default=str(DEFAULT_RUNS / "reports"),
                    help="对比报告的输出目录")
    cp.set_defaults(func=cmd_compare)

    return p


def _add_run_parser(subparsers):
    r = subparsers.add_parser("run", help="执行一轮评测（需要昇腾主机）")
    r.add_argument("--profile", default="smoke", choices=["smoke", "standard", "official"])
    r.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    r.add_argument("--submission", required=True,
                   help="提交目录或 zip 包（含 manifest.yaml 与 operators/）")
    r.add_argument("--ingest-dir", default=str(DEFAULT_INGEST),
                   help="zip 包的解包目录（内容寻址，幂等）")
    r.add_argument("--runs", default=str(DEFAULT_RUNS))
    r.add_argument("--compiler-name", default="triton-ascend")
    r.add_argument("--compiler-version", default=None)
    r.add_argument("--compiler-commit", default=None)
    r.add_argument("--clock-locked", action="store_true")
    r.add_argument("--limit", type=int, default=None, help="只跑前 N 个用例")
    r.add_argument("--operators", default=None,
                   help="只跑指定算子（逗号分隔，可用用例编号前缀或类名，"
                        "如 cross_entropy_loss,transpose）——环境窗口跑不完"
                        "全量测速时按算子分批用")
    r.set_defaults(func=cmd_run)


def main(argv: list[str] | None = None) -> int:
    """顶层入口。必须兜住 BaseException——见 safeexec 模块的说明。

    最后一道防线：即使某个边界漏改了，也不允许进程「无输出、退出码 0」地
    消失。那会让 CI 把「评测被攻击掉了」当成评测成功。
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        emit("\n被用户中断", file=sys.stderr)
        return 130
    except BaseException as exc:  # noqa: BLE001
        emit(f"评测异常终止：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _ingest_run_submission(args):
    sub_path = Path(args.submission)
    package = None
    if sub_path.is_file():
        from triton_eval.package import PackageError, ingest_zip

        try:
            package = ingest_zip(sub_path, Path(args.ingest_dir))
        except PackageError as exc:
            emit(f"提交包摄取失败：\n{exc.render()}", file=sys.stderr)
            return None
        sub_root = package.dest
        note = f"，剥离外层目录 {package.stripped_prefix!r}" if package.stripped_prefix else ""
        emit(f"已解包 {sub_path.name} → {sub_root}"
                     f"（sha256 {package.short_sha}，{len(package.files)} 个文件{note}）")
    else:
        sub_root = sub_path

    return sub_root, package


def _parent_submission(sub_root):
    from triton_eval.submission import discover
    from triton_eval.validator import (
        ValidationError_,
        ValidationResult,
        validate_static,
    )

    validation = ValidationResult(sub_root.name)
    declared: list[str] = []
    before_snapshot: dict[Path, bytes] = {}
    discovered = None
    try:
        validate_static(sub_root, validation)
        discovered = discover(sub_root)
        declared = discovered.operator_names()
        before_snapshot = discovered.snapshot()
    except Exception as exc:  # noqa: BLE001
        validation.errors.append(
            ValidationError_("SUB-LOD-001", "load", "-",
                             f"提交清单不可用：{type(exc).__name__}: {exc}")
        )

    return validation, declared, before_snapshot, discovered


def _record_parent_environment(context):
    from triton_eval.envprobe.collect import probe_env

    context.env, context.env_fp = probe_env(
        compiler_name=context.args.compiler_name,
        compiler_version=context.compiler_version,
        compiler_commit=context.args.compiler_commit,
        compiler_install_mode="source",
        clock_locked=context.args.clock_locked,
    )
    dump_env(context.env, context.run_dir)  # 子进程的篡改校验/一致性核对以此为基准

    context.run_id = context.s.create_run(
        context.corpus_id, context.submission_id, context.args.profile, context.env, context.env_fp,
                          "PENDING", run_dir=str(context.run_dir))
    emit(f"运行 {context.run_id}：{context.run_dir}")


def _prepare_parent_run(context):
    from triton_eval.envprobe.collect import triton_import_version

    context.corpus_id = context.s.import_corpus(context.manifest)
    # 被测编译器身份：优先用命令行给定，其次从环境实测。
    # 安装方式固定为 source：目录式提交没有打包环节。
    context.compiler_version = context.args.compiler_version or triton_import_version() or "unknown"
    context.submission_id = context.s.upsert_submission(
        context.args.compiler_name, context.compiler_version, context.args.compiler_commit, "source"
    )

    # ---------- 父进程侧：静态校验 + 快照（绝不 import 提交代码） ----------
    parent = _parent_submission(context.sub_root)
    context.validation, context.declared, context.before_snapshot, context.discovered = parent

    # 空清单不违规（它是合法 YAML），但此刻「0 个算子」几乎一定是
    # 提交方的失误。不报错会让它安静地跑完 60 个用例、全部
    # SUBMISSION_INVALID（ADV-E2 实测）。
    if not context.declared:
        emit("警告：提交清单未声明任何算子，本轮不会有算子被评测",
                     file=sys.stderr)

    _record_parent_environment(context)


def _parent_record_kwargs(context, **extra):
    kwargs = dict(
        submit_id=context.sub_root.name,
        root_path=str(context.sub_root),
        operators=context.declared,
        snapshot_fp=_snapshot_fp(context.before_snapshot) if context.before_snapshot else "",
        validation_ok=context.validation.ok,
        validation=[{
            "rule_id": e.rule_id, "stage": e.stage, "operator": e.operator,
            "detail": e.detail, "lineno": e.lineno,
        } for e in context.validation.errors],
        package_sha256=context.package.sha256 if context.package else None,
        package_name=context.package.package_name if context.package else None,
    )
    # extra 优先：子进程的完整结论覆盖父进程的静态初值
    kwargs.update(extra)
    return kwargs


def _reject_parent_validation(context):
    if not context.validation.ok:
        # 静态失败不需要子进程：加载阶段必然短路，直接作废
        record_id = context.s.insert_submission_record(**_parent_record_kwargs(context))
        context.s.set_run_submission_record(context.run_id, record_id)
        context.s.set_run_probe(context.run_id, "SKIP")
        context.s.finish_run(context.run_id, "INVALID",
                     f"提交静态校验未通过：{len(context.validation.errors)} 项")
        emit(context.validation.render(), file=sys.stderr)
        return True

    emit(f"提交目录 {context.sub_root.name}：声明 {len(context.declared)} 个算子"
                 f"（corpus 共 {len(context.manifest.operators)} 个）")

    return False


def _prepare_run_supervisor(context):
    context.cases = select_cases(context.manifest.cases, context.args.operators)
    if context.args.limit:
        context.cases = context.cases[: context.args.limit]
    if context.args.operators:
        emit(f"按算子过滤（{context.args.operators}）：本轮跑 {len(context.cases)} / "
                     f"{len(context.manifest.cases)} 个用例")
    (context.run_dir / "results.jsonl").write_text("", encoding="utf-8")
    context.events_path = context.run_dir / "events.jsonl"
    context.events_path.touch()

    # ---------- 监督子进程 ----------
    # 事件预算 = 用例 timeout + 余量（盖住冷编译与性能计时）。
    # 环境变量可覆盖预算参数（测试 / 现场调试用；正常运行不设）。
    from triton_eval.supervisor import GRACE_S, IDLE_MARGIN_S

    _override = os.environ.get("TRITON_EVAL_CASE_TIMEOUT_S")
    context.case_timeouts = (
        [int(_override)] * len(context.cases) if _override
        else [c.timeout_s for c in context.cases]
    )
    context.idle_margin_s = float(os.environ.get("TRITON_EVAL_IDLE_MARGIN_S", IDLE_MARGIN_S))
    context.grace_s = float(os.environ.get("TRITON_EVAL_GRACE_S", GRACE_S))

    context.child_cmd = _child_cmd() + [
        "--run-dir", str(context.run_dir), "--run-id", str(context.run_id),
        "--corpus", str(context.corpus_dir), "--submission", str(context.sub_root),
        "--profile", context.args.profile,
    ]
    if context.args.limit:
        context.child_cmd += ["--limit", str(context.args.limit)]
    if context.args.operators:
        context.child_cmd += ["--operators", context.args.operators]

    context.completed: list[str] = []
    context.fatal: dict[str, Any] | None = None
    context.done_reason: str | None = None
    context.child_validation_ok = True
    context.probe_ok = True
    context.env_check: dict[str, Any] = {}
    context.record_inserted = False


def _handle_parent_event(context, evt):
    kind = evt.get("event")
    if kind == "meta":
        _record_parent_meta(context, evt)
        return
    if kind == "env_check":
        context.env_check.update(evt)
        if not evt.get("ok"):
            emit(f"环境一致性校验失败：{evt.get('mismatches')}",
                         file=sys.stderr)
        return
    if kind == "probe":
        context.probe_ok = bool(evt.get("ok"))
        context.s.set_run_probe(context.run_id, "PASS" if context.probe_ok else "FAIL")
        if not context.probe_ok:
            emit(f"健康探针失败：{evt.get('reason')}", file=sys.stderr)
        return
    if kind == "case":
        _record_parent_case(context, evt)
        return
    if kind == "warn":
        emit(f"警告：{evt.get('detail')}", file=sys.stderr)
        return
    if kind == "fatal":
        context.fatal = evt
        emit(f"子进程致命错误（{evt.get('stage')}）：{evt.get('detail')}",
                     file=sys.stderr)
        return
    if kind == "done":
        context.done_reason = evt.get("reason")
        return


def _supervise_parent_run(context, on_event):
    from triton_eval.supervisor import run_child

    try:
        outcome = run_child(
            context.child_cmd,
            events_path=context.events_path,
            case_timeouts=context.case_timeouts,
            on_event=on_event,
            stdout_file=context.run_dir / "logs" / "child-stdout.log",
            stderr_file=context.run_dir / "logs" / "child-stderr.log",
            idle_margin_s=context.idle_margin_s,
            grace_s=context.grace_s,
        )
    except OSError as exc:
        context.s.finish_run(context.run_id, "INVALID", f"评测子进程无法启动：{exc}")
        emit(f"评测子进程无法启动：{exc}", file=sys.stderr)
        return 1

    return _finish_parent_run(context, outcome)


def _finish_parent_run(context, outcome):
    from functools import partial

    # ---------- 收尾：补记、核查、归档 ----------
    _finalize_run(
        s=context.s, run_id=context.run_id, run_dir=context.run_dir, cases=context.cases, completed=context.completed,
        fatal=context.fatal, outcome=outcome, done_reason=context.done_reason,
        child_validation_ok=context.child_validation_ok, probe_ok=context.probe_ok,
        record_inserted=context.record_inserted, discovered=context.discovered,
        before_snapshot=context.before_snapshot, env_check=context.env_check,
        record_kwargs_factory=partial(_parent_record_kwargs, context),
    )
    run = context.s.get_run(context.run_id)
    jsonl = context.run_dir / "results.jsonl"
    if run and run["status"] == "OK":
        emit(f"运行 {context.run_id} 完成，结果: {jsonl}")
        return 0
    reason = (run or {}).get("invalid_reason") if run else None
    emit(f"运行 {context.run_id} 未正常完成（{run['status'] if run else '?'}）："
                 f"{reason}", file=sys.stderr)
    emit(f"已产出的结果: {jsonl}（{len(context.completed)} 个用例）", file=sys.stderr)
    return 1


def _record_parent_meta(context, evt):
    context.child_validation_ok = bool(evt.get("validation_ok"))
    failures = evt.get("load_failures") or {}
    if failures:
        emit(f"提交中有 {len(failures)} 个算子加载失败：", file=sys.stderr)
        for name, why in sorted(failures.items()):
            emit(f"  {name}: {why}", file=sys.stderr)
    ops = evt.get("operators") or []
    emit(f"子进程载入 {len(ops)} 个算子"
                 f"（篡改校验守护 {evt.get('api_guard_covered', '?')} 个关键接口）")
    # 校验失败必须把结论打到终端：错误只进数据库的话，
    # 「为什么被拒」要靠开 SQL 才知道——提交方无从整改，
    # 验证脚本也只能盲猜（真机首轮实测踩到）。
    if not context.child_validation_ok:
        emit("提交校验未通过（子进程完整结论）：", file=sys.stderr)
        for e in evt.get("errors") or []:
            loc = f"（第 {e.get('lineno')} 行）" if e.get("lineno") else ""
            emit(f"  [{e.get('rule_id')}] {e.get('detail')}{loc}",
                         file=sys.stderr)
    # 提交记录以子进程的完整校验结论为准（含加载与篡改校验）
    kwargs = _parent_record_kwargs(context, 
        validation_ok=context.child_validation_ok,
        load_failures=failures,
        validation=[{
            "rule_id": e.rule_id, "stage": e.stage, "operator": e.operator,
            "detail": e.detail, "lineno": e.lineno,
        } for e in _merge_validation(context.validation, evt)],
    )
    record_id = context.s.insert_submission_record(**kwargs)
    context.s.set_run_submission_record(context.run_id, record_id)
    context.record_inserted = True


def _record_parent_case(context, evt):
    rec = CaseResultRecord.model_validate(evt["record"])
    try:
        context.s.insert_case_result(rec)
    except Exception as exc:  # noqa: BLE001
        # UNIQUE(run_id, case_id) 冲突等：记录而不让监督崩溃
        emit(f"  {rec.case_code}: 结果落库失败 {exc}", file=sys.stderr)
    context.completed.append(rec.case_code)
    emit(f"  {rec.case_code}: {rec.status.value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
