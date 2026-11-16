#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""演示「无 NPU 重算路径」。

场景：昇腾主机上跑完一轮，产出了 runs/<id>/results.jsonl；
现在换到一台没有 NPU 的机器（或数据库丢了），只凭这份 jsonl
就要能恢复记录、重算分数、重新生成报告，全程不执行任何 kernel。

用法：
    python scripts/demo_offline_rescore.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
import logging

import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from triton_eval.cli import main  # noqa: E402
from triton_eval.corpusmod import load_manifest  # noqa: E402
from triton_eval.models import CaseResultRecord, CaseStatus  # noqa: E402
from triton_eval.store import Store  # noqa: E402
from triton_eval.console import emit

DEMO = Path("/tmp/triton-eval-demo")
CORPUS = PROJECT_ROOT / "corpus" / "v0.3-multi-input-frozen"

ENV = {
    "arch": "aarch64", "os": "Linux 5.10.0-hce2", "kernel": "5.10.0-hce2",
    "npu_model": "Ascend910", "npu_count": 2, "npu_id": "0",
    "ascend_visible_devices": "0", "driver_version": "25.5.5",
    "cann_version": "9.0.0", "python_version": "3.11.4",
    "torch_version": "2.7.1+cpu", "torch_npu_version": "2.7.1.post4",
    "compiler_name": "triton-ascend", "compiler_version": "3.2.1",
    "triton_import_version": "3.2.0",
    "venv_path": "/home/developer/venv/te-compiler", "venv_lock_hash": "abc",
    "clock_locked": False, "env_health": "ok", "env_health_notes": "",
}


def setup() -> tuple[str, int, Path]:
    """模拟一次已在昇腾主机完成的运行：只留下 results.jsonl。"""
    if DEMO.exists():
        shutil.rmtree(DEMO)
    run_dir = DEMO / "run-1"
    run_dir.mkdir(parents=True)

    store = Store(DEMO / "demo.db")
    store.create_schema()
    manifest = load_manifest(CORPUS / "manifest.json")
    corpus_id = store.import_corpus(manifest)
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(corpus_id, sub_id, "standard", ENV, "demo-fp", "PASS",
                              run_dir=str(run_dir))
    store.finish_run(run_id, "OK")

    lines = []
    for i, case in enumerate(manifest.cases):
        if i == 2:
            status = CaseStatus.WRONG_RESULT
        elif i == 12:
            status = CaseStatus.COMPILE_FAIL
        else:
            status = CaseStatus.PASS
        rec = CaseResultRecord(
            run_id=run_id, case_code=case.code, operator=case.operator, status=status,
            max_rel_err=(1e-2 if status is CaseStatus.WRONG_RESULT else 1e-7),
            max_abs_err=(1e-3 if status is CaseStatus.WRONG_RESULT else 1e-9),
            speedup=(2.0 if status is CaseStatus.PASS else None),
            perf_cv=(0.01 if status is CaseStatus.PASS else None),
            perf_median_ms=(0.4 if status is CaseStatus.PASS else None),
            samples_ms=([0.39, 0.41, 0.40] if status is CaseStatus.PASS else []),
            compile_err=("triton.CompilationError"
                         if status is CaseStatus.COMPILE_FAIL else None),
        )
        lines.append(rec.model_dump_json())

    jsonl = run_dir / "results.jsonl"
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "env.json").write_text(
        json.dumps(ENV, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

    # 清空数据库结果，只留真相源
    store.conn.execute("DELETE FROM case_result WHERE run_id=?", (run_id,))
    store.conn.commit()
    remaining = len(store.case_results(run_id))
    store.close()
    emit(f"[setup] run_id={run_id}，jsonl 行数={len(lines)}，"
                 f"数据库中剩余结果={remaining}（已清空，只剩 jsonl）")
    return str(DEMO / "demo.db"), run_id, jsonl


def main() -> int:
    db, run_id, jsonl = setup()
    emit()

    emit("[1/3] 从 results.jsonl 重建数据库记录")
    if main_cli(["--db", db, "import", "--run", str(run_id), "--jsonl", str(jsonl)]) != 0:
        return 1

    emit("\n[2/3] 离线打分（首轮基线即锚点）")
    if main_cli(["--db", db, "score", "--run", str(run_id)]) != 0:
        return 1

    emit("\n[3/3] 生成报告")
    out = str(DEMO / "reports")
    if main_cli(["--db", db, "report", "--run", str(run_id), "--out", out]) != 0:
        return 1

    md = Path(out) / f"report-run-{run_id}.md"
    emit(f"\n报告片段：")
    emit("-" * 60)
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith(("| **主得分", "| 严格通过率", "| 静默错误数",
                            "| 几何平均", "| 性能得分")):
            emit(line)
    emit("-" * 60)
    emit(f"\n全程未执行任何 kernel —— 这就是 results.jsonl 契约的价值。")
    return 0


def main_cli(argv: list[str]) -> int:
    from triton_eval.cli import build_parser

    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
