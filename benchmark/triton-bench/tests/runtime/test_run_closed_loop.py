# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""「zip → 结果」闭环的本地集成测试（stub 子进程，不需要 NPU）。

真实闭环里评测子进程（evalchild）需要昇腾环境；这里用 TRITON_EVAL_CHILD_CMD
接缝换成假子进程，验证闭环的**监督侧**：zip 摄取、静态校验、事件流落库、
部分结果恢复、终态判定、评分、报告——除「真跑 kernel」之外的一切。

假子进程忠实模拟事件协议（meta / env_check / probe / case / done），
并按契约把 results.jsonl 逐条写盘。
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

from triton_eval.cli import main

DEMO = fixture_path(Path(__file__).resolve().parents[2] / "examples" / "demo-submission")
# 现行语料的用例数（随默认语料走，不写死版本数字）
_N_CASES = len(json.loads(
    (Path(__file__).resolve().parents[2] / "corpus" / "v0.5-cov20-frozen"
     / "manifest.json").read_text(encoding="utf-8"))["cases"])

# 假子进程：读 --corpus 的 manifest.json，逐用例发 PASS 事件。
# mode=hang 版本吞 SIGTERM 赖着不走，用于验证监督者的硬超时与补记。
STUB_CHILD = '''
import argparse, json, os, signal, sys, time

p = argparse.ArgumentParser()
p.add_argument("--run-dir"); p.add_argument("--run-id", type=int)
p.add_argument("--corpus"); p.add_argument("--submission")
p.add_argument("--profile"); p.add_argument("--limit", type=int)
p.add_argument("--operators")
a = p.parse_args()

events = os.path.join(a.run_dir, "events.jsonl")
jsonl = os.path.join(a.run_dir, "results.jsonl")

def emit(kind, **payload):
    with open(events, "a") as fh:
        fh.write(json.dumps({"event": kind, **payload}) + "\\n")
        fh.flush()

manifest = json.load(open(os.path.join(a.corpus, "manifest.json"), encoding="utf-8"))
cases = manifest["cases"][: a.limit] if a.limit else manifest["cases"]
if a.operators:  # 与 corpusmod.select_cases 同口径（监督者与子进程必须一致）
    wanted = {x.strip().lower() for x in a.operators.split(",") if x.strip()}
    cases = [c for c in cases
             if wanted & {c["code"].split("-", 1)[0].lower(), c["operator"].lower()}]

emit("meta", submit_id="stub", validation_ok=True, errors=[],
     declared=sorted({c["operator"] for c in cases}),
     operators=sorted({c["operator"] for c in cases}),
     load_failures={}, api_guard_covered=50)
emit("env_check", ok=True, mismatches=[], identity={})
emit("probe", ok=True, reason="")

if os.environ.get("STUB_MODE") == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(600)

for c in cases:
    rec = {"run_id": a.run_id, "case_code": c["code"], "operator": c["operator"],
           "status": "PASS"}
    line = json.dumps(rec)
    with open(jsonl, "a") as fh:
        fh.write(line + "\\n")
    emit("case", record=rec)

emit("done", reason="ok", cases=len(cases))
'''


def _demo_zip(tmp_path: Path) -> Path:
    z = tmp_path / "demo.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(DEMO.rglob("*")):
            if p.is_file() and p.suffix in {".yaml", ".yml", ".py"}:
                zf.writestr(str(p.relative_to(DEMO)), p.read_bytes())
    return z


def _run_cli(db: Path, *extra: str) -> int:
    return main(["--db", str(db), *extra])


def test_zip_run_score_report_closed_loop(tmp_path: Path, monkeypatch, capsys) -> None:
    stub = tmp_path / "stub_child.py"
    stub.write_text(STUB_CHILD, encoding="utf-8")
    monkeypatch.setenv("TRITON_EVAL_CHILD_CMD", f"{sys.executable} {stub}")
    monkeypatch.delenv("STUB_MODE", raising=False)

    z = _demo_zip(tmp_path)
    db = tmp_path / "t.db"
    runs = tmp_path / "runs"
    ingest = tmp_path / "ingest"

    # ---------- run：zip 直进 ----------
    rc = _run_cli(db, "run", "--profile", "smoke", "--submission", str(z),
                  "--runs", str(runs), "--ingest-dir", str(ingest))
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "已解包" in out.out

    run_dirs = [d for d in runs.iterdir() if d.is_dir() and d.name.startswith("run-")]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # 事件协议与产物契约
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert kinds.count("case") == _N_CASES, \
        "全部用例都要有事件（数量随现行语料）"

    for artifact in ("results.jsonl", "env.json", "integrity.json", "artifacts.json"):
        assert (run_dir / artifact).is_file(), f"缺产物 {artifact}"
    arts = json.loads((run_dir / "artifacts.json").read_text())
    listed = {f["path"] for f in arts["files"]}
    assert "results.jsonl" in listed and "env.json" in listed
    assert all(f["sha256"] for f in arts["files"])

    # 数据库侧
    import sqlite3

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM eval_run ORDER BY id DESC LIMIT 1").fetchone()
    assert row["status"] == "OK"
    n = conn.execute("SELECT COUNT(*) c FROM case_result WHERE run_id=?",
                     (row["id"],)).fetchone()["c"]
    assert n == _N_CASES
    # 提交记录携带包身份
    rec = conn.execute(
        "SELECT * FROM submission_record ORDER BY id DESC LIMIT 1").fetchone()
    assert rec["package_name"] == "demo.zip"
    assert rec["package_sha256"] and len(rec["package_sha256"]) == 64
    assert rec["validation_ok"] == 1
    conn.close()

    # ---------- score ----------
    rc = _run_cli(db, "score", "--run", "latest")
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "OpScore = 100.00" in out.out, "假子进程全 PASS，主分应为满分"

    # ---------- report ----------
    rc = _run_cli(db, "report", "--run", "latest", "--out", str(runs / "reports"))
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert (runs / "reports" / "report-run-1.md").is_file()


def test_operators_filter_runs_only_selected_ops(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """--operators：真机环境两小时自动清空（2026-10-12 实测），测速全量
    跑不完一个窗口——按算子分窗跑，窗口内全对的算子照样冻结性能锚点。
    断言过滤在父进程（事件预算/打印）与子进程（实际执行）两侧同源。
    """
    stub = tmp_path / "stub_child.py"
    stub.write_text(STUB_CHILD, encoding="utf-8")
    monkeypatch.setenv("TRITON_EVAL_CHILD_CMD", f"{sys.executable} {stub}")
    monkeypatch.delenv("STUB_MODE", raising=False)

    z = _demo_zip(tmp_path)
    db = tmp_path / "t.db"
    runs = tmp_path / "runs"

    rc = _run_cli(db, "run", "--profile", "smoke", "--submission", str(z),
                  "--operators", "cross_entropy_loss,transpose", "--runs", str(runs))
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "按算子过滤" in out.out

    run_dir = next(d for d in runs.iterdir() if d.name.startswith("run-"))
    results = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    ops = {r["operator"] for r in results}
    prefixes = {r["case_code"].split("-", 1)[0] for r in results}
    assert prefixes == {"cross_entropy_loss", "transpose"}, prefixes
    assert len(results) == 40, "CEL 20 + transpose 20（v0.4 清单实测）"

    # 类名写法也必须等价命中（select_cases 两种写法同源）
    rc = _run_cli(db, "run", "--profile", "smoke", "--submission", str(z),
                  "--operators", "Transpose", "--runs", str(runs))
    assert rc == 0, capsys.readouterr().err
    run_dir = sorted(d for d in runs.iterdir() if d.name.startswith("run-"))[-1]  # 最新一轮
    results = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    assert {r["case_code"].split("-", 1)[0] for r in results} == {"transpose"}

    # 过滤后没有用例必须显式报错，不能静默跑一个空轮
    with pytest.raises(ValueError, match="过滤后没有用例"):
        select_cases_checked()


def select_cases_checked():
    from triton_eval.corpusmod import load_manifest, select_cases
    manifest = load_manifest(Path(__file__).resolve().parents[2]
                             / "corpus" / "v0.4-cov20-frozen" / "manifest.json")
    return select_cases(manifest.cases, "no_such_operator")


def test_dir_submission_still_works(tmp_path: Path, monkeypatch, capsys) -> None:
    """目录式入口不回归（zip 只是新增形态，不是替换）。"""
    stub = tmp_path / "stub_child.py"
    stub.write_text(STUB_CHILD, encoding="utf-8")
    monkeypatch.setenv("TRITON_EVAL_CHILD_CMD", f"{sys.executable} {stub}")
    monkeypatch.delenv("STUB_MODE", raising=False)

    db = tmp_path / "t.db"
    rc = _run_cli(db, "run", "--profile", "smoke",
                  "--submission", str(DEMO),
                  "--runs", str(tmp_path / "runs"),
                  "--ingest-dir", str(tmp_path / "ingest"),
                  "--limit", "3")
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "已解包" not in out.out


def test_bad_zip_is_rejected_before_run(tmp_path: Path, capsys) -> None:
    z = tmp_path / "fake.zip"
    z.write_bytes(b"not a zip")
    rc = _run_cli(tmp_path / "t.db", "run", "--submission", str(z),
                  "--runs", str(tmp_path / "runs"))
    out = capsys.readouterr()
    assert rc == 1
    assert "SUB-PKG-001" in out.err


def test_hung_child_gets_partial_results_filled(tmp_path: Path, monkeypatch, capsys) -> None:
    """子进程挂死：已产出的事件落库，未完成的用例补记 RUNTIME_FAIL，run 判 PARTIAL。"""
    stub = tmp_path / "stub_child.py"
    stub.write_text(STUB_CHILD, encoding="utf-8")
    monkeypatch.setenv("TRITON_EVAL_CHILD_CMD", f"{sys.executable} {stub}")
    monkeypatch.setenv("STUB_MODE", "hang")
    # 测试接缝：把事件预算压到秒级（正常运行用例默认 60s + 180s 余量）
    monkeypatch.setenv("TRITON_EVAL_CASE_TIMEOUT_S", "1")
    monkeypatch.setenv("TRITON_EVAL_IDLE_MARGIN_S", "1")
    monkeypatch.setenv("TRITON_EVAL_GRACE_S", "0.5")

    db = tmp_path / "t.db"
    rc = _run_cli(db, "run", "--profile", "smoke",
                  "--submission", str(_demo_zip(tmp_path)),
                  "--runs", str(tmp_path / "runs"),
                  "--ingest-dir", str(tmp_path / "ingest"),
                  "--limit", "3")
    out = capsys.readouterr()
    assert rc == 1, "PARTIAL 运行必须返回非 0"

    import sqlite3

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM eval_run ORDER BY id DESC LIMIT 1").fetchone()
    assert row["status"] == "PARTIAL"
    assert row["invalid_reason"] and "硬超时" in row["invalid_reason"]
    statuses = [r["status"] for r in conn.execute("SELECT status FROM case_result WHERE run_id=?", (row["id"],))]
    # 3 个用例：挂死的子进程一个都没产出 → 全部补记 RUNTIME_FAIL
    assert statuses == ["RUNTIME_FAIL"] * 3, statuses
    errs = [r["compile_err"] for r in conn.execute("SELECT compile_err FROM case_result WHERE run_id=?", (row["id"],))]
    assert all("硬超时" in e for e in errs)
    conn.close()
