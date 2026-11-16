# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""测试集冻结与校验的单元测试（L0，无需 NPU）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from triton_eval.corpusmod import (
    build_manifest,
    compute_case_content_hash,
    manifest_digest,
    screen_determinism,
    verify_corpus_files,
    verify_manifest,
)
from triton_eval.models import CaseSource, CaseSpec, OperatorSpec
from triton_eval.store import utcnow

KERNEL_SRC = "import triton\nimport triton.language as tl\n"


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus" / "v0.1-mvp-frozen"
    (d / "operators" / "add").mkdir(parents=True)
    (d / "operators" / "add" / "kernel.py").write_text(KERNEL_SRC, encoding="utf-8")
    (d / "operators" / "add" / "ref.py").write_text("def ref(x, y):\n    return x + y\n",
                                                    encoding="utf-8")
    return d


def _cases(n: int = 3) -> list[CaseSpec]:
    return [
        CaseSpec(
            code=f"add-f32-{i:04d}", operator="add", source=CaseSource.VENDOR_SELFTEST,
            dtype="float32",
            shape=[1024], seed=1000 + i, launch={"BLOCK": 256},
            feature_atoms=["FA-MATH-ADD"],
        )
        for i in range(n)
    ]


def _operators() -> list[OperatorSpec]:
    return [OperatorSpec(name="add", source=CaseSource.VENDOR_SELFTEST, weight=1.0,
                         ref_path="operators/add/ref.py")]


# ---------- 清单生成（任务 3.4） ----------


def test_manifest_build_fills_hashes(corpus_dir: Path) -> None:
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(), _cases())
    assert m.manifest_hash
    assert all(c.content_hash for c in m.cases), "每个用例都必须有内容哈希"
    assert len(m.cases) == 3


def test_manifest_digest_is_stable(corpus_dir: Path) -> None:
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(), _cases())
    again = manifest_digest(m)
    assert again == m.manifest_hash, "清单哈希必须可重复计算"
    assert manifest_digest(m) == again


def test_manifest_digest_ignores_frozen_at() -> None:
    """同一份内容在不同时间生成，必须得到同一个哈希。

    曾经的 bug：frozen_at 参与哈希，导致每次 build_corpus 都产出
    不同的 manifest_hash —— 无法判断两次运行是否用了同一份 corpus，
    「冻结」语义失效。
    """
    import copy

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "operators" / "add").mkdir(parents=True)
        (d / "operators" / "add" / "kernel.py").write_text("x = 1\n", encoding="utf-8")
        ops = [OperatorSpec(name="add", source=CaseSource.VENDOR_SELFTEST,
                            weight=1.0, ref_path="operators/add/ref.py")]
        cases = [CaseSpec(code="add-f32-0000", operator="add",
                          source=CaseSource.VENDOR_SELFTEST,

                          dtype="float32", shape=[8], seed=1, launch={})]

        m1 = build_manifest(d, "v1", "0.1-mvp", ops, cases)
        m2 = copy.deepcopy(m1)
        m2.frozen_at = "2099-01-01T00:00:00+00:00"  # 改时间
        assert m1.frozen_at != m2.frozen_at
        assert m1.manifest_hash == manifest_digest(m2), (
            "改变 frozen_at 不应改变清单哈希"
        )


def test_manifest_digest_changes_with_content(corpus_dir: Path) -> None:
    """但内容变了，哈希必须变（否则篡改检测失效）。"""
    m = build_manifest(corpus_dir, "v1", "0.1-mvp", _operators(), _cases())
    original = m.manifest_hash
    m.cases[0].seed = 987654321
    assert manifest_digest(m) != original, "内容变化必须反映在哈希上"


def test_manifest_case_count_matches(corpus_dir: Path) -> None:
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(), _cases(5))
    assert len(m.cases) == 5


# ---------- 冻结校验（任务 3.5） ----------


def test_verify_manifest_passes_for_untouched(corpus_dir: Path) -> None:
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(), _cases())
    verify_manifest(m)


def test_verify_manifest_fails_when_tampered(corpus_dir: Path) -> None:
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(), _cases())
    m.cases[0].seed = 999999
    with pytest.raises(ValueError, match="清单哈希不匹配"):
        verify_manifest(m)


def test_verify_corpus_files_detects_case_tamper(corpus_dir: Path) -> None:
    """用例被篡改必须被检出。

    契约变更后没有 kernel 可篡改，改为验证**用例内容**本身：
    改 seed 后重算哈希会与清单记录不符。
    """
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(), _cases())
    verify_corpus_files(corpus_dir, m)

    m.cases[0].seed = 424242
    # 改用例字段会先触发清单哈希校验（它覆盖 cases），
    # 因此这里匹配的是清单级失败——同样是「冻结后被修改」的正确信号
    with pytest.raises(ValueError, match="哈希不匹配"):
        verify_corpus_files(corpus_dir, m)


def test_case_hash_depends_only_on_case_not_implementation(tmp_path: Path) -> None:
    """契约变更后，内容哈希只覆盖用例自身字段。

    这带来一个关键性质：哈希标识「要测什么」，与「谁来实现」无关
    —— 正是支持外部提交所需要的。
    """
    case = _cases(1)[0]
    h1 = compute_case_content_hash(tmp_path, case)
    # 换一个不存在的目录也不影响：用例不再引用 kernel 文件
    h2 = compute_case_content_hash(tmp_path / "nonexistent", case)
    assert h1 == h2, "用例哈希不得依赖任何实现文件"

    case.seed = 999
    assert compute_case_content_hash(tmp_path, case) != h1, "内容变化必须反映在哈希上"


# ---------- 确定性筛查（任务 3.3 的判定逻辑） ----------


def test_deterministic_cases_are_kept() -> None:
    cases = _cases(3)
    kept, excluded = screen_determinism(cases, lambda c: f"stable-{c.code}", repeats=5)
    assert len(kept) == 3
    assert excluded == []


def test_nondeterministic_cases_are_excluded() -> None:
    cases = _cases(3)
    calls = {"n": 0}

    def flaky(case: CaseSpec) -> str:
        # 只有第二个用例每次执行返回不同指纹
        if case.code == "add-f32-0001":
            calls["n"] += 1
            return f"varies-{calls['n']}"
        return f"ok-{case.code}"

    kept, excluded = screen_determinism(cases, flaky, repeats=5)
    assert [c.code for c in kept] == ["add-f32-0000", "add-f32-0002"]
    assert len(excluded) == 1
    assert excluded[0]["code"] == "add-f32-0001"
    assert "不确定" in excluded[0]["reason"]


def test_screen_result_feeds_manifest_excluded_list(corpus_dir: Path) -> None:
    """被剔除的用例要能记入清单，事后可查「这个 case 为什么不在里面」。"""
    cases = _cases(3)
    calls = {"n": 0}

    def flaky(case: CaseSpec) -> str:
        if case.code == "add-f32-0000":
            calls["n"] += 1
            return f"varies-{calls['n']}"
        return f"ok-{case.code}"

    kept, excluded = screen_determinism(cases, flaky)
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(), kept,
                       excluded=excluded)
    assert len(m.cases) == 2
    assert m.excluded[0]["code"] == "add-f32-0000"
    assert m.frozen_at


def test_excluded_records_survive_write_load_roundtrip(corpus_dir: Path) -> None:
    """剔除记录持久化：写盘 → 读回，逐条原因必须一字不丢。

    v0.4/v0.5 是零剔除，这条通路从未被真实数据走过——门禁一旦剔除
    而此处有缺陷，审计将查不到任何记录。用注入的剔除记录钉住通路。
    """
    from triton_eval.corpusmod import load_manifest, write_manifest

    excluded = [
        {"code": "add-f32-0000", "reason": "参考实现自检：重复执行 5 次得到 2 种不同结果"},
        {"code": "add-f32-0001", "reason": "参考实现自检：输出含 NaN"},
    ]
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(),
                       _cases(1), excluded=excluded)
    assert write_manifest(m, corpus_dir / "manifest.json")

    loaded = load_manifest(corpus_dir / "manifest.json")
    assert loaded.excluded == excluded, "剔除记录经写盘读回必须完整"
    verify_manifest(loaded)  # 完整清单（含剔除记录）自洽


def test_excluded_records_are_tamper_evident(corpus_dir: Path) -> None:
    """剔除记录不可被事后抹除或改写——它在清单指纹覆盖范围内。

    审计含义：有人想悄悄把一条剔除记录改成「正常通过」（或删掉），
    清单哈希立刻对不上，评测启动时即被检出。
    """
    from triton_eval.corpusmod import load_manifest, write_manifest

    excluded = [{"code": "add-f32-0000", "reason": "参考实现自检：输出含 NaN"}]
    m = build_manifest(corpus_dir, "v0.1-mvp-frozen", "0.1-mvp", _operators(),
                       _cases(1), excluded=excluded)
    assert write_manifest(m, corpus_dir / "manifest.json")

    loaded = load_manifest(corpus_dir / "manifest.json")
    loaded.excluded = []  # 事后抹除剔除记录
    with pytest.raises(ValueError, match="清单哈希不匹配"):
        verify_manifest(loaded)

    loaded = load_manifest(corpus_dir / "manifest.json")
    loaded.excluded[0]["reason"] = "无异常"  # 事后改写原因
    with pytest.raises(ValueError, match="清单哈希不匹配"):
        verify_manifest(loaded)


def test_all_frozen_corpora_load_and_verify() -> None:
    """schema 演进后，历史冻结清单必须仍然可加载可校验。

    实测教训：CaseSpec 增加 value_range/contiguous 字段后，
    pydantic 重新序列化多出这两个默认字段，旧清单的冻结哈希对不上、
    整个 v0.3 语料加载失败——冻结资产不能因为评测方改 schema 就报废。
    （v0.3 已做显式迁移：内容语义不变，重算指纹，保留 frozen_at。）
    """
    from triton_eval.corpusmod import load_manifest

    root = Path(__file__).resolve().parents[2]
    corpora = sorted((root / "corpus").glob("v*/manifest.json"))
    assert len(corpora) >= 2, "应至少有 v0.3 与 v0.4 两代冻结语料"
    for mf in corpora:
        m = load_manifest(mf)
        if m.version in ("v0.7-ascend-contract", "v0.8-official-candidates",
                         "v0.8.1-official-candidates", "v0.9-official-candidates",
                         "v0.10-coverage-candidates", "v0.11-implementation-candidates",
                         "v0.11.1-contract-candidates", "v0.12-compat-candidates",
                         "v0.13-evidence-candidates", "v0.14-backend-evidence-candidates",
                         "v0.15-backend-capture-fix-candidates",
                         "v0.16-backend-pattern-certification-candidates",
                         "v0.17-pattern-completion-candidates"):
            verify_manifest(m)
            assert all(compute_case_content_hash(mf.parent, case) == case.content_hash for case in m.cases)
            verify_corpus_files(mf.parent, m)
        else:
            verify_corpus_files(mf.parent, m)
        # 每个用例必须携带非空标签（v0.4 起；v0.3 迁移后允许为空——它没打标）
        if m.version.startswith("v0.4"):
            labeled = sum(1 for c in m.cases if c.feature_atoms)
            assert labeled == len(m.cases), f"{m.version} 有未打标用例"
