#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证每个算子的参考实现可独立执行，并产出合理输出。

为什么需要它：
    参考实现是判定的锚。若某个 ref.py 本身有问题（参数数量不对、
    dtype 处理错、输入构造越界），失败会被记成 ORACLE_FAIL，
    而这个失败是**评测方自己的问题**，不该归因给被测编译器。
    本脚本在纯 CPU 上把 60 个用例的「造输入 → 算参考」走一遍，
    把这类问题在冻结之后、评测之前暴露出来。

曾经的错误形态（都真实发生过）：
    - 用统一的 [torch.randn(8)] 去验所有算子 → add/mul 需要两个输入，报 IndexError
    - 输入构造不读 attrs → gather 的 index 越界、grouped_matmul 的组数与行数对不上
    - 整型输入用 randn→int → 截断出大量 0，arg_max 出现并列最大值，用例不可判定

用法：
    python scripts/verify_refs.py                 # CPU（推荐，随时可跑）
    python scripts/verify_refs.py --device npu:0  # 昇腾机上复核算子兼容性
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from triton_eval.console import emit

from triton_eval.corpusmod import load_manifest, verify_corpus_files  # noqa: E402
from triton_eval.refcheck import check_all  # noqa: E402

CORPUS = PROJECT_ROOT / "corpus" / "v0.3-multi-input-frozen"


def main() -> int:
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError:
        emit(
            "本脚本需要 torch（用于执行参考实现）。\n"
            "它在昇腾执行机的 venv 中可用；本地可用 `pip install torch` 装 CPU 版。\n"
            "跳过即可，不影响其它检查。",
            file=sys.stderr,
        )
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--operators", help="comma-separated operator names")
    args = ap.parse_args()

    # 必须在任何张量创建之前注册 torch.npu：
    # 本脚本走的是 L0 的 inputs 模块（不含 torch_npu 的副作用导入），
    # 不导入的话 "npu:0" 会被当成未知设备类型（实测直接 RuntimeError）。
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    manifest = load_manifest(Path(args.corpus) / "manifest.json")
    verify_corpus_files(Path(args.corpus), manifest)
    cases = manifest.cases
    if args.operators:
        selected = set(args.operators.split(","))
        known = {case.operator for case in cases}
        if selected - known:
            ap.error(f"unknown operators: {sorted(selected - known)}")
        cases = [case for case in cases if case.operator in selected]
    failures = check_all(cases, args.device, corpus_dir=args.corpus)

    emit(f"参考实现自检：{len(cases)} 个用例，device={args.device}")
    if failures:
        emit(f"失败 {len(failures)} 个：")
        for code, why in sorted(failures.items()):
            emit(f"  {code:<26} {why}")
        return 1
    emit("全部通过：输入构造与参考实现均可执行、输出有限且可复现")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
