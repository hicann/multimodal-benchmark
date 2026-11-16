# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""报告生成（L0，无需 NPU）。

报告只需要数据库里的记录，因此可以在任何机器上重算并重新渲染，
不必重新执行评测——这是 results.jsonl 契约的直接收益。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import CaseStatus, is_counted, is_severe

TEMPLATE_DIR = Path(__file__).parent / "templates"


def build_context(
    run: dict[str, Any],
    env: dict[str, Any],
    submission: dict[str, Any],
    results: list[Any],
    score: dict[str, Any] | None,
    *,
    corpus_version: str,
    spec_version: str,
    scorer_version: str,
    run_dir: str,
    submission_record: dict[str, Any] | None = None,
    total_operators: int = 0,
    op_perf: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status.value] = by_status.get(r.status.value, 0) + 1

    perf_count = sum(1 for r in results if r.speedup)

    # 提交覆盖率：只交一小部分算子也能在「已交的」上面拿满分，
    # 不披露覆盖率就会被读成「全部通过」。两者必须同时出现才有意义。
    covered = sorted({r.operator for r in results if is_counted(r.status)})
    declared: list[str] = []
    if submission_record:
        try:
            declared = json.loads(submission_record.get("operators_json") or "[]")
        except (TypeError, ValueError):
            declared = []

    return {
        "run": run,
        "env": env,
        "env_fp": run.get("env_fp", ""),
        "submission": submission,
        "submission_record": submission_record,
        "coverage": {
            "covered": covered,
            "covered_count": len(covered),
            "declared": declared,
            "declared_count": len(declared),
            "total_operators": total_operators,
            "missing": sorted(set(declared) - set(covered)),
        },
        "corpus_version": corpus_version,
        "spec_version": spec_version,
        "scorer_version": scorer_version,
        "score": score,
        # 按算子性能分（score_per_operator 的输出）：分数 + 为什么没分。
        # 「为什么没分」必须与分数同屏，否则加权分会被误读成全量尺子的分。
        "op_perf": op_perf or {},
        "counts": {"total": len(results), "by_status": by_status, "perf": perf_count},
        "severe_cases": [r for r in results if is_severe(r.status)],
        # 未锁频 → 性能数据不可信，必须标注
        "unverified_perf": not env.get("clock_locked", False),
        "artifacts": {
            "results_jsonl": str(Path(run_dir) / "results.jsonl"),
            "env_json": str(Path(run_dir) / "env.json"),
            "run_dir": str(run_dir),
        },
    }


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_markdown(ctx: dict[str, Any]) -> str:
    return _environment().get_template("report.md.j2").render(**ctx)


def render_html(ctx: dict[str, Any]) -> str:
    return _environment().get_template("report.html.j2").render(**ctx)


def write_report(ctx: dict[str, Any], out_dir: str | Path, basename: str) -> dict[str, Path]:
    """渲染并写出 Markdown 与 HTML 两份报告，返回路径。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    md_path = d / f"{basename}.md"
    html_path = d / f"{basename}.html"
    md_path.write_text(render_markdown(ctx), encoding="utf-8")
    html_path.write_text(render_html(ctx), encoding="utf-8")
    return {"markdown": md_path, "html": html_path}


def dump_env(env: dict[str, Any], run_dir: str | Path) -> Path:
    """把环境指纹落盘，供事后追溯。"""
    p = Path(run_dir) / "env.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(env, ensure_ascii=False, indent=2, sort_keys=True),
                 encoding="utf-8")
    return p


def status_breakdown(results: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r.status.value] = out.get(r.status.value, 0) + 1
    return out


def pass_count(results: list[Any]) -> int:
    return sum(1 for r in results if r.status is CaseStatus.PASS)
