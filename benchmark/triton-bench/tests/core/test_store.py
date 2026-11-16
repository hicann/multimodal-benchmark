# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""存储层单元测试（L0，无需 NPU）。"""

from __future__ import annotations

import sqlite3

import pytest

from triton_eval.models import (
    CaseResultRecord,
    CaseSource,
    CaseSpec,
    CaseStatus,
    CorpusManifest,
    OperatorSpec,
    RunScoreRecord,
)
from triton_eval.store import Store, utcnow

CORPUS_VERSION = "v0.3-multi-input-frozen"


def _manifest() -> CorpusManifest:
    op = OperatorSpec(
        name="add", source=CaseSource.VENDOR_SELFTEST, weight=1.0,
        ref_path="operators/add/ref.py",
    )
    cases = []
    for case_index in range(3):
        cases.append(CaseSpec(
            code=f'add-f32-{case_index:04d}',
            operator='add',
            source=CaseSource.VENDOR_SELFTEST,
            dtype='float32',
            shape=[1024],
            seed=1000 + case_index,
            launch={'BLOCK': 256},
            feature_atoms=['FA-MATH-ADD'],
            content_hash='h',
        ))
    return CorpusManifest(
        version=CORPUS_VERSION, spec_version="0.1-mvp",
        frozen_at=utcnow(), operators=[op], cases=cases, manifest_hash="x",
    )


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    s.create_schema()
    yield s
    s.close()


def test_schema_created_on_empty_db(store) -> None:
    tables = {row["name"] for row in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {
        "corpus_version", "operator", "eval_case", "compiler_submission",
        "eval_run", "case_result", "run_score",
    }
    assert expected <= tables


def test_corpus_import_and_case_lookup(store) -> None:
    corpus_id = store.import_corpus(_manifest())
    assert corpus_id > 0
    assert store.case_id_by_code("add-f32-0000") is not None
    assert store.case_id_by_code("nonexistent") is None


def test_reimport_same_corpus_keeps_foreign_keys(store) -> None:
    """重复导入同一 corpus 版本不得破坏外键。

    曾经的 bug：用 INSERT OR REPLACE 写 corpus_version，
    REPLACE 会删除旧行，而 operator/eval_case 有外键指向它 →
    `FOREIGN KEY constraint failed`。第二次运行就会炸。
    """
    m1 = _manifest()
    first = store.import_corpus(m1)
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(first, sub_id, "smoke", {}, "fp", "PASS")
    store.insert_case_result(
        CaseResultRecord(run_id=run_id, case_code="add-f32-0000",
                         operator="add", status=CaseStatus.PASS)
    )

    # 重新生成清单（frozen_at 与 hash 都会变），再导入一次
    m2 = _manifest()
    for c in m2.cases:
        c.content_hash = "changed"
    m2.frozen_at = "2027-01-01T00:00:00+00:00"
    m2.manifest_hash = "different-hash"

    second = store.import_corpus(m2)
    assert second == first, "同一版本必须复用同一个 id，不能新建"

    # 已有结果记录的外键仍然有效
    assert len(store.case_results(run_id)) == 1
    assert store.case_id_by_code("add-f32-0000") is not None

    # 更新的字段确实生效
    row = store.conn.execute(
        "SELECT manifest_hash FROM corpus_version WHERE id=?", (first,)
    ).fetchone()
    assert row["manifest_hash"] == "different-hash"


def test_reimport_twice_in_a_row_is_idempotent(store) -> None:
    """连续导入两次，用例数与算子数不应翻倍。"""
    store.import_corpus(_manifest())
    store.import_corpus(_manifest())
    n_cases = store.conn.execute("SELECT COUNT(*) c FROM eval_case").fetchone()["c"]
    n_ops = store.conn.execute("SELECT COUNT(*) c FROM operator").fetchone()["c"]
    assert n_cases == 3
    assert n_ops == 1


def test_duplicate_case_result_rejected(store) -> None:
    """同一 run 的同一 case 只能有一条结果。"""
    corpus_id = store.import_corpus(_manifest())
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(corpus_id, sub_id, "smoke", {"arch": "aarch64"}, "fp", "PASS")

    rec = CaseResultRecord(run_id=run_id, case_code="add-f32-0000",
                           operator="add", status=CaseStatus.PASS)
    store.insert_case_result(rec)
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_case_result(rec)


def test_unknown_case_result_rejected(store) -> None:
    corpus_id = store.import_corpus(_manifest())
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(corpus_id, sub_id, "smoke", {}, "fp", "PASS")
    rec = CaseResultRecord(run_id=run_id, case_code="ghost", operator="add",
                           status=CaseStatus.PASS)
    with pytest.raises(KeyError):
        store.insert_case_result(rec)


def test_rebuild_from_jsonl_restores_results(store, tmp_path) -> None:
    """数据库可从 results.jsonl 完整重建。这是可迁移性的支点。"""
    corpus_id = store.import_corpus(_manifest())
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(corpus_id, sub_id, "smoke", {}, "fp", "PASS")

    jsonl = tmp_path / "results.jsonl"
    lines = []
    for i, status in enumerate([CaseStatus.PASS, CaseStatus.WRONG_RESULT,
                                CaseStatus.COMPILE_FAIL]):
        rec = CaseResultRecord(run_id=run_id, case_code=f"add-f32-{i:04d}",
                               operator="add", status=status, max_rel_err=1e-6)
        lines.append(rec.model_dump_json())
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 先写进去，再清空，验证能重建回来
    for line in lines:
        store.insert_case_result(CaseResultRecord.model_validate_json(line))
    store.conn.execute("DELETE FROM case_result WHERE run_id=?", (run_id,))
    store.conn.commit()
    assert store.case_results(run_id) == []

    restored = store.rebuild_from_jsonl(run_id, jsonl)
    assert restored == 3
    results = store.case_results(run_id)
    assert len(results) == 3
    assert results[1].severe is True
    assert results[1].operator == "add"


def test_run_score_allows_rescore_per_scorer_version(store) -> None:
    """重算必须新增记录，不能覆盖原有记录。"""
    corpus_id = store.import_corpus(_manifest())
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(corpus_id, sub_id, "smoke", {}, "fp", "PASS")

    for ver in ("0.1.0", "0.2.0"):
        store.insert_run_score(
            RunScoreRecord(
                run_id=run_id, coverage_ops_core=80.0, coverage_strict=60.0,
                severe_count=1, baseline_anchor_ref="PER-OP",
                scorer_version=ver, computed_at=utcnow(),
            )
        )
    scores = store.run_scores(run_id)
    assert len(scores) == 2
    assert {s["scorer_version"] for s in scores} == {"0.1.0", "0.2.0"}


def test_duplicate_score_for_same_scorer_version_rejected(store) -> None:
    corpus_id = store.import_corpus(_manifest())
    sub_id = store.upsert_submission("triton-ascend", "3.2.1", None, "wheel")
    run_id = store.create_run(corpus_id, sub_id, "smoke", {}, "fp", "PASS")
    score = RunScoreRecord(
        run_id=run_id, coverage_ops_core=80.0, coverage_strict=60.0, severe_count=0,
        baseline_anchor_ref="PER-OP", scorer_version="0.1.0", computed_at=utcnow(),
    )
    store.insert_run_score(score)
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_run_score(score)


def test_rejects_incompatible_old_db(tmp_path) -> None:
    """旧版 schema（eval_case.kernel_path NOT NULL）的库必须在建表时就拒用。

    曾经的表现：CREATE TABLE IF NOT EXISTS 静默放过旧表，
    import_corpus 插到一半才抛 IntegrityError，看起来像数据坏了。
    """

    old = sqlite3.connect(tmp_path / "old.db")
    old.executescript("""
        CREATE TABLE eval_case (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kernel_path TEXT NOT NULL,
            code TEXT NOT NULL
        );
    """)
    old.commit()
    old.close()

    s = Store(tmp_path / "old.db")
    with pytest.raises(RuntimeError, match="kernel_path"):
        s.create_schema()
    s.close()
