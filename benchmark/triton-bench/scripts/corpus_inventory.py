# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""题库清单统计工具——打印「每个算子搭载了哪些题」的原始数据。

docs/corpus-inventory.md 的第七节就是本工具的输出整理成表。
文档与数据的一致性靠重新运行对账：改了题库（或怀疑文档过期）时跑一遍，
逐算子核对形状 / 类型 / 参数 / 数值分布 / 跨步 / 规模。

用法：
    python scripts/corpus_inventory.py            # 统计默认 v0.4 题库
    python scripts/corpus_inventory.py <corpus 目录>
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from triton_eval.console import emit

CORPUS = Path(sys.argv[1]) if len(sys.argv) > 1 \
    else Path(__file__).resolve().parents[1] / "corpus" / "v0.4-cov20-frozen"

manifest = json.loads(
    (CORPUS / "manifest.json").read_text(encoding="utf-8"))
cases = manifest["cases"]
emit(f"题库 {manifest['version']}，总数: {len(cases)}")


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def size_tag(n):
    if n <= 4096:
        return "S"
    if n <= 262144:
        return "M"
    if n <= 2097152:
        return "L"
    return "X"


by_op = collections.defaultdict(list)
for c in cases:
    by_op[c["operator"]].append(c)

for op in sorted(by_op):
    cs = by_op[op]
    shapes = sorted({tuple(c["shape"]) for c in cs}, key=lambda s: (-len(s), s))
    dtypes = sorted({c["dtype"] for c in cs})
    kinds = collections.Counter(
        (c["value_range"] or {}).get("kind", "normal") for c in cs)
    strided = sum(1 for c in cs if not c["contiguous"])
    sizes = collections.Counter(size_tag(numel(c["shape"])) for c in cs)
    dimvals = collections.Counter(
        c["attrs"].get("dim") for c in cs if "dim" in c.get("attrs", {}))
    emit(f"\n== {op}（{len(cs)} 题）")
    emit(f"   形状({len(shapes)}): {[list(s) for s in shapes]}")
    emit(f"   类型: {dtypes}")
    emit(f"   数值: {dict(kinds)}  跨步: {strided}  规模: {dict(sizes)}")
    if dimvals:
        emit(f"   dim: {dict(dimvals)}")
    attrsets = sorted({json.dumps(c["attrs"], sort_keys=True) for c in cs})
    for a in attrsets:
        emit(f"     {a}")

# 全局统计
all_kinds = collections.Counter(
    (c["value_range"] or {}).get("kind", "normal") for c in cases)
all_sizes = collections.Counter(size_tag(numel(c["shape"])) for c in cases)
all_strided = sum(1 for c in cases if not c["contiguous"])
atoms = collections.Counter()
for c in cases:
    atoms.update(c.get("feature_atoms", []))
emit("\n== 全局")
emit(f"   数值: {dict(all_kinds)}  跨步: {all_strided}  规模: {dict(all_sizes)}")
emit(f"   标签并集: {len(atoms)}")
emit(f"   最多: {atoms.most_common(5)}")
emit(f"   最少: {atoms.most_common()[-5:]}")
