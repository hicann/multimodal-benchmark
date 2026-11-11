#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""生成并冻结测试集清单。

MVP 阶段的 5 个手写算子已淘汰——它们都是冒烟用的，
覆盖面只有 Elementwise 与 Reduction，无法反映编译器真实能力差异。

现在改为：
    - 算子规格来自 `operator_registry`（取自 cann-bench proto）
    - 用例由算子规格**派生**，不再绑定 kernel
    - 按 category 绑定固定数量（默认每类 5 个）

用法：
    python scripts/build_corpus.py                 # 生成清单
    python scripts/build_corpus.py --screen        # 先做确定性筛查（需昇腾）
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_eval.casegen import count_by_category, generate_all, manifest_operators
from triton_eval.corpusmod import build_manifest, write_manifest  # noqa: E402
from triton_eval.models import CaseSource, CaseSpec, OperatorSpec  # noqa: E402
from triton_eval.operator_registry import ALL_OPERATORS  # noqa: E402
from triton_eval.specload import load_feature_atoms, load_spec  # noqa: E402
from triton_eval.taxonomy import (  # noqa: E402
    DEFAULT_CASES_PER_CATEGORY,
    categories_missing_operators,
)
from triton_eval.console import emit

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPUS_DIR = PROJECT_ROOT / "corpus" / "v0.3-multi-input-frozen"
VERSION = "v0.3-multi-input-frozen"


def _operators() -> list[OperatorSpec]:
    """从算子规格映射为 corpus 的算子条目（含权重）。

    MVP 等权 1.0：权重标定（profiler 采集调用频次 × 耗时占比）属 P1，
    手工拍权重会污染 OpScore 的解释性。
    """
    return manifest_operators(ALL_OPERATORS)


def _cases(per_category: int) -> list[CaseSpec]:
    """派生用例并转换为 corpus 的 CaseSpec（不含 kernel 字段）。

    attrs 必须随用例一起冻结：曾经只在 DerivedCase 里生成、写 manifest 时
    丢掉，结果必填属性（dim / perm / group_list）传不到参考实现，
    那些算子的用例谁都跑不出来。
    """
    derived = generate_all(ALL_OPERATORS, per_category=per_category)
    return [
        CaseSpec(
            code=d.code,
            operator=d.operator,
            source=CaseSource.VENDOR_SELFTEST,
            dtype=d.dtype,
            shape=d.shape,
            attrs=d.attrs,
            seed=d.seed,
            launch={},
            timeout_s=d.timeout_s,
            feature_atoms=[],
        )
        for d in derived
    ]


def main() -> int:
    args = _build_arguments()

    ops = _operators()
    cases = _cases(args.per_category)
    excluded: list[dict[str, str]] = []

    # 缺参考实现的算子，其用例必然判 ORACLE_FAIL——必须说出来，
    # 否则「跑不通」会被误读成被测编译器不行。
    missing_refs = [
        op.name for op in ALL_OPERATORS
        if not (CORPUS_DIR / "operators" / op.entry / "ref.py").is_file()
    ]
    if missing_refs:
        emit(f"警告：{len(missing_refs)} 个算子缺少 ref.py，"
                     f"其用例将全部判 ORACLE_FAIL：{', '.join(missing_refs)}",
                     file=sys.stderr)

    # 参考实现自检是**冻结门禁**，不是可选的善后步骤。
    # 参考一旦不可执行或不确定，其用例必然判 ORACLE_FAIL，
    # 而那个失败会被误读成被测方的问题。宁可冻得少一点，也不要冻进噪声。
    cases = _reference_gate(args, cases, excluded)

    if args.screen:
        import torch_npu  # noqa: F401  注册 torch.npu

        from triton_eval.casegen import generate_all as _gen
        from triton_eval.harness.npu_exec import execute_case

        def runner(case: CaseSpec) -> str:
            m = execute_case(case)
            return f"{m.compiled}|{m.executed}|{m.submission_ok}|{m.max_rel_err}"

        kept, excluded = _screen(cases, runner, args.repeats)
        cases = kept
        emit(f"确定性筛查：保留 {len(cases)} 个，剔除 {len(excluded)} 个")

    spec = load_spec()
    manifest = build_manifest(
        CORPUS_DIR, VERSION, spec.spec_version, ops, cases,
        feature_atoms_version=str(load_feature_atoms()["version"]),
        excluded=excluded,
    )
    out = CORPUS_DIR / "manifest.json"
    written = write_manifest(manifest, out)

    emit(f"已冻结 {VERSION}：算子 {len(manifest.operators)} 个，"
                 f"用例 {len(manifest.cases)} 个")
    for cat, n in sorted(count_by_category(
        generate_all(ALL_OPERATORS, per_category=args.per_category)
    ).items()):
        emit(f"  {cat:<20} {n}")
    missing = categories_missing_operators(
        set(count_by_category(
            generate_all(ALL_OPERATORS, per_category=args.per_category)))
    )
    if missing:
        emit(f"  空缺类别：{', '.join(missing)}")
    emit(f"清单哈希 {manifest.manifest_hash}")
    emit(f"写入 {out}" if written else f"内容未变，未重写 {out}")
    return 0


def _screen(cases, runner, repeats: int):
    from triton_eval.corpusmod import screen_determinism

    return screen_determinism(cases, runner, repeats=repeats)




def _build_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true",
                    help="执行确定性筛查（需要昇腾主机）")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--per-category", type=int, default=DEFAULT_CASES_PER_CATEGORY)
    ap.add_argument("--verify-refs", dest="verify_refs", action="store_true",
                    default=True, help="冻结前自检参考实现（默认开启）")
    ap.add_argument("--no-verify-refs", dest="verify_refs", action="store_false",
                    help="跳过参考实现自检（不推荐）")
    ap.add_argument("--device", default="cpu",
                    help="自检所用设备；昇腾机上用 npu:0 可检出 CPU 上不显形的算子兼容问题")
    args = ap.parse_args()
    return args


def _reference_gate(args, cases, excluded):
    if not args.verify_refs:
        return cases
    from triton_eval.refcheck import check_all, torch_available

    if not torch_available():
        emit("（无 torch，跳过参考实现自检）", file=sys.stderr)
    else:
        if args.device.startswith("npu"):
            import torch_npu  # noqa: F401  注册 torch.npu
        failures = check_all(cases, args.device, corpus_dir=CORPUS_DIR)
        if failures:
            kept = [c for c in cases if c.code not in failures]
            for code, why in sorted(failures.items()):
                excluded.append({"code": code, "reason": f"参考实现自检：{why}"})
            # 输出到 stdout：门剔除是冻结结果的一部分，
            # 曾经只写 stderr，被 `| tail` 一截就完全看不出来。
            emit(f"参考自检（device={args.device}）：剔除 {len(failures)} 个用例")
            for code, why in sorted(failures.items()):
                emit(f"  {code}: {why}")
            cases = kept
        else:
            emit(f"参考自检（device={args.device}）：{len(cases)} 个用例全部通过")
    return cases


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
