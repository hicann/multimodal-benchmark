# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""命令行端到端测试（L0，无需 NPU）。

覆盖「打分 / 报告 / 从 jsonl 重建」这条不依赖硬件的链路——
只要 results.jsonl 在，这些能力在任何机器上都能跑。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton_eval import cli
from triton_eval.corpusmod import load_manifest
from triton_eval.models import CaseResultRecord, CaseStatus
from triton_eval.store import Store, utcnow

# v0.1-mvp-frozen（5 个手写算子）与 v0.2-category-frozen 均已淘汰：
# 前者只是冒烟，后者缺 attrs 与多输入构造，多数算子根本跑不出来。
# 现在用 v0.3-multi-input-frozen。
CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus" / "v0.3-multi-input-frozen"

ENV = {
    "arch": "aarch64", "os": "Linux 5.10.0-hce2", "kernel": "5.10.0-hce2",
    "npu_model": "Ascend910", "npu_count": 2, "npu_id": "0",
    "ascend_visible_devices": "0", "driver_version": "25.5.5",
    "cann_version": "9.0.0", "python_version": "3.11.4",
    "torch_version": "2.7.1+cpu", "torch_npu_version": "2.7.1.post4",
    "compiler_name": "triton-ascend", "compiler_version": "3.2.1",
    "triton_import_version": "3.2.0", "venv_path": "/home/x/venv/te-compiler",
    "venv_lock_hash": "abc", "clock_locked": False,
    "env_health": "ok", "env_health_notes": "",
}


@pytest.fixture()
def seeded(tmp_path: Path):
    """构造一次已完成的运行：导入真实 corpus，写入构造的结果。"""
    db = tmp_path / "t.db"
    store = Store(db)
    store.create_schema()
    manifest = load_manifest(CORPUS_DIR / "manifest.json")
    corpus_id = store.import_corpus(manifest)
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(corpus_id, sub_id, "standard", ENV, "fp-x", "PASS",
                              run_dir=str(tmp_path / "run-1"))

    # 第 3 个用 WRONG_RESULT（静默错误）、第 13 个用 COMPILE_FAIL，其余 PASS
    statuses = [
        CaseStatus.WRONG_RESULT if i == 2
        else CaseStatus.COMPILE_FAIL if i == 12
        else CaseStatus.PASS
        for i in range(len(manifest.cases))
    ]
    for case, st in zip(manifest.cases, statuses):
        store.insert_case_result(
            CaseResultRecord(
                run_id=run_id, case_code=case.code, operator=case.operator, status=st,
                max_rel_err=(1e-2 if st is CaseStatus.WRONG_RESULT else 1e-7),
                max_abs_err=(1e-3 if st is CaseStatus.WRONG_RESULT else 1e-9),
                speedup=(2.0 if st is CaseStatus.PASS else None),
                perf_cv=(0.01 if st is CaseStatus.PASS else None),
                perf_median_ms=(0.4 if st is CaseStatus.PASS else None),
                samples_ms=([0.4] if st is CaseStatus.PASS else []),
                compile_err="boom" if st is CaseStatus.COMPILE_FAIL else None,
            )
        )
    store.finish_run(run_id, "OK")
    store.close()
    return {"db": str(db), "run": run_id, "tmp": tmp_path, "cases": manifest.cases}


# ---------- 任务 8.2 ----------


def test_score_command(seeded, capsys) -> None:
    rc = cli.main(["--db", seeded["db"], "score", "--run", str(seeded["run"])])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OpScore" in out
    assert "静默错误数        = 1" in out
    assert "首轮基线" not in out
    assert "性能得分" in out


def test_score_first_round_per_op_anchor(seeded) -> None:
    """按算子锚点：全对的算子首轮自锚得 100，有判错的算子无分但不拖累别人。"""
    rc = cli.main(["--db", seeded["db"], "score", "--run", str(seeded["run"])])
    assert rc == 0
    store = Store(seeded["db"])
    scores = store.run_scores(seeded["run"])
    assert scores[0]["baseline_anchor_ref"] == "PER-OP"
    # seeded 轮有两个算子含判错（WRONG_RESULT / COMPILE_FAIL）→ 无分；
    # 其余全对算子自锚 → 各 100 → 轮级加权分 = 100
    assert scores[0]["perf_score"] == 100.0
    store.close()


def test_score_reports_which_operators_lack_perf_score(seeded, capsys) -> None:
    """没分的算子必须点名并给原因——加权分不能被读成全量尺子的分。"""
    cli.main(["--db", seeded["db"], "score", "--run", str(seeded["run"])])
    out = capsys.readouterr().out
    assert "按算子性能分" in out
    assert "本轮未全对" in out, "seeded 轮含两个判错算子，原因必须可见"


def test_score_rescore_creates_new_record(seeded) -> None:
    cli.main(["--db", seeded["db"], "score", "--run", str(seeded["run"])])
    cli.main(["--db", seeded["db"], "score", "--run", str(seeded["run"]),
              "--scorer-version", "0.9.0-test"])
    store = Store(seeded["db"])
    versions = {s["scorer_version"] for s in store.run_scores(seeded["run"])}
    assert len(versions) == 2, "重算必须留痕为新记录"
    store.close()


# ---------- 任务 8.3 ----------


def test_import_command_rebuilds_db(seeded, tmp_path) -> None:
    """从 jsonl 重建数据库记录。"""
    store = Store(seeded["db"])
    results = store.case_results(seeded["run"])
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text("\n".join(r.model_dump_json() for r in results) + "\n",
                     encoding="utf-8")
    store.conn.execute("DELETE FROM case_result WHERE run_id=?", (seeded["run"],))
    store.conn.commit()
    assert store.case_results(seeded["run"]) == []
    store.close()

    rc = cli.main(["--db", seeded["db"], "import", "--run", str(seeded["run"]),
                   "--jsonl", str(jsonl)])
    assert rc == 0

    store = Store(seeded["db"])
    restored = store.case_results(seeded["run"])
    assert len(restored) == len(seeded["cases"])
    store.close()


def test_import_rejects_unknown_run(seeded, tmp_path) -> None:
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("", encoding="utf-8")
    rc = cli.main(["--db", seeded["db"], "import", "--run", "999", "--jsonl", str(jsonl)])
    assert rc == 1


# ---------- 任务 8.4 ----------


def test_report_command(seeded, capsys) -> None:
    cli.main(["--db", seeded["db"], "score", "--run", str(seeded["run"])])
    out_dir = seeded["tmp"] / "reports"
    rc = cli.main(["--db", seeded["db"], "report", "--run", str(seeded["run"]),
                   "--out", str(out_dir)])
    assert rc == 0
    md = out_dir / f"report-run-{seeded['run']}.md"
    html = out_dir / f"report-run-{seeded['run']}.html"
    assert md.is_file() and html.is_file()

    text = md.read_text(encoding="utf-8")
    assert "## 9. 结果解释边界" in text
    assert "UNVERIFIED" in text, "环境未锁频 → 性能数据必须标注"
    assert "静默错误清单" in text
    severe_code = seeded["cases"][2].code
    assert severe_code in text, (
        f"静默错误用例 {severe_code} 必须出现在报告里"
    )


def test_report_unknown_run(seeded) -> None:
    rc = cli.main(["--db", seeded["db"], "report", "--run", "999"])
    assert rc == 1


# ---------- 只凭 jsonl 重算（任务 9.4 的基础能力） ----------


def test_rescore_from_jsonl_only(seeded, tmp_path, capsys) -> None:
    """无需数据库、无需 NPU，只凭 jsonl 就能算出分数。"""
    store = Store(seeded["db"])
    results = store.case_results(seeded["run"])
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text("\n".join(r.model_dump_json() for r in results) + "\n",
                     encoding="utf-8")
    store.close()

    rc = cli.main(["--db", seeded["db"], "rescore", "--run", str(seeded["run"]),
                   "--jsonl", str(jsonl)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == seeded["run"]
    assert payload["severe_count"] == 1
    assert "total" not in payload, "不得产出跨赛道总分"


# ---------- compare（两轮对比） ----------


def _run_row(seeded: dict) -> dict:
    store = Store(seeded["db"])
    row = store.conn.execute(
        "SELECT corpus_version_id, submission_id FROM eval_run WHERE id = ?",
        (seeded["run"],),
    ).fetchone()
    out = dict(row)
    store.close()
    return out


def test_compare_command(seeded, tmp_path, capsys) -> None:
    """两轮对比：功能回归/修复、性能显著变化各就各位，报告落盘。"""
    row = _run_row(seeded)
    store = Store(seeded["db"])
    run2 = store.create_run(row["corpus_version_id"], row["submission_id"],
                            "standard", ENV, "fp-x", "PASS",
                            run_dir=str(seeded["tmp"] / "run-2"))
    for i, case in enumerate(seeded["cases"]):
        if i == 0:
            st, sp = CaseStatus.WRONG_RESULT, 2.0    # 功能回归（原 PASS）
        elif i in (2, 12):
            st, sp = CaseStatus.PASS, 2.0            # 功能修复 ×2（原判错）
        elif i == 1:
            st, sp = CaseStatus.PASS, 1.0            # 性能显著回归（-50%）
        else:
            st, sp = CaseStatus.PASS, 2.0
        store.insert_case_result(CaseResultRecord(
            run_id=run2, case_code=case.code, operator=case.operator,
            status=st, speedup=sp, perf_cv=0.01,
            perf_median_ms=0.4, samples_ms=[0.4],
        ))
    store.finish_run(run2, "OK")
    store.close()

    # 两轮都打分，对照表才有 Δ
    cli.main(["--db", seeded["db"], "score", "--run", str(seeded["run"])])
    cli.main(["--db", seeded["db"], "score", "--run", str(run2)])

    out_dir = tmp_path / "reports"
    rc = cli.main(["--db", seeded["db"], "compare", "--run", str(run2),
                   "--against", str(seeded["run"]), "--out", str(out_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "功能回归 1、修复 2" in out
    assert "性能显著回归 1、提升 0" in out

    md = (out_dir / f"compare-run{run2}-vs-run{seeded['run']}.md"
          ).read_text(encoding="utf-8")
    assert "## 1. 两轮身份" in md
    assert "## 2. 分数对照" in md and "Δ" in md
    assert seeded["cases"][0].code in md, "功能回归用例必须点名"
    assert seeded["cases"][2].code in md and seeded["cases"][12].code in md, \
        "功能修复用例必须点名"
    assert seeded["cases"][1].code in md, "性能显著回归用例必须点名"
    assert "-50%" in md, "加速比变化幅度必须给出"


def test_compare_warns_on_env_mismatch(seeded, tmp_path, capsys) -> None:
    """环境指纹不同的两轮对比：警告但不阻止，可信度留给读者判断。"""
    row = _run_row(seeded)
    store = Store(seeded["db"])
    run2 = store.create_run(row["corpus_version_id"], row["submission_id"],
                            "standard", ENV, "fp-different", "PASS",
                            run_dir=str(seeded["tmp"] / "run-2b"))
    for case in seeded["cases"]:
        store.insert_case_result(CaseResultRecord(
            run_id=run2, case_code=case.code, operator=case.operator,
            status=CaseStatus.PASS, speedup=2.0, perf_cv=0.01,
        ))
    store.finish_run(run2, "OK")
    store.close()

    rc = cli.main(["--db", seeded["db"], "compare", "--run", str(run2),
                   "--against", str(seeded["run"]), "--out", str(tmp_path)])
    assert rc == 0, "指纹不同只警告，不阻止"
    captured = capsys.readouterr()
    assert "环境指纹不同" in captured.err
    assert "环境指纹不同" in (tmp_path / f"compare-run{run2}-vs-run{seeded['run']}.md"
                        ).read_text(encoding="utf-8")


def test_compare_rejects_different_corpus(seeded, tmp_path, capsys) -> None:
    """题库版本不同的两轮不可比——题都不一样，直接拒绝。"""
    from triton_eval.models import (
        CaseSource, CaseSpec, CorpusManifest, OperatorSpec,
    )

    row = _run_row(seeded)
    store = Store(seeded["db"])
    ops = [OperatorSpec(name="solo", source=CaseSource.VENDOR_SELFTEST,
                        weight=1.0, ref_path="operators/solo/ref.py")]
    cases = [CaseSpec(code="solo-f32-0000", operator="solo",
                      source=CaseSource.VENDOR_SELFTEST, dtype="float32",
                      shape=[1024], seed=1, launch={}, feature_atoms=[])]
    manifest = CorpusManifest(version="v-other", spec_version="0.1-mvp",
                              frozen_at=utcnow(), operators=ops, cases=cases,
                              manifest_hash="y")
    corpus_id = store.import_corpus(manifest)
    run3 = store.create_run(corpus_id, row["submission_id"], "standard",
                            ENV, "fp-z", "PASS")
    store.insert_case_result(CaseResultRecord(
        run_id=run3, case_code="solo-f32-0000", operator="solo",
        status=CaseStatus.PASS, speedup=2.0,
    ))
    store.finish_run(run3, "OK")
    store.close()

    rc = cli.main(["--db", seeded["db"], "compare", "--run", str(run3),
                   "--against", str(seeded["run"]), "--out", str(tmp_path)])
    assert rc == 1
    assert "题库版本不同" in capsys.readouterr().err


def test_compare_rejects_same_run(seeded, tmp_path) -> None:
    rc = cli.main(["--db", seeded["db"], "compare",
                   "--run", str(seeded["run"]),
                   "--against", str(seeded["run"]), "--out", str(tmp_path)])
    assert rc == 1


# ---------- import-corpus ----------


def test_import_corpus_command(tmp_path, capsys) -> None:
    """导入必须与清单一致。

    用例数不能写死：冻结门禁会剔除自检不通过的用例，
    且剔除结果依设备而定（CPU 与昇腾上可能不同）。
    曾经写死 60，于是在昇腾上冻出 59 个用例后这条就红了——
    那不是导入坏了，是断言把「冻结结果」当成了常量。
    """

    manifest = load_manifest(CORPUS_DIR / "manifest.json")
    rc = cli.main(["--db", str(tmp_path / "c.db"), "import-corpus",
                   "--corpus", str(CORPUS_DIR)])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"算子 {len(manifest.operators)} 个" in out
    assert f"用例 {len(manifest.cases)} 个" in out


def test_import_corpus_validates_content_hash(tmp_path) -> None:
    """冻结后的用例被篡改，导入必须失败。

    契约变更后没有 kernel 可篡改，改为篡改用例字段（seed）。
    """

    bad = tmp_path / "bad-corpus"
    bad.mkdir()
    data = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    data["cases"][0]["seed"] = 987654321
    (bad / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    rc = cli.main(["--db", str(tmp_path / "c.db"), "import-corpus", "--corpus", str(bad)])
    assert rc != 0
