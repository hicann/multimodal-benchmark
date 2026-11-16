# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""生成过程日志（审计）：把构建流水线的每一步写入语料目录的 yaml。

任务书要求「生成过程保留日志以供审计」。此前构建只往终端打印——会话
结束即丢失，v0.3/v0.4/v0.5 的历史构建因此没有任何留痕可查。

设计：
    - **整体重写**：每次构建（包括内容未变的重跑）都写入完整事件列表，
      文件顶部为 license 头注释，其下为 YAML 列表，机器可解析；
      终端输出继续给人看，两者并行不互相替代。
    - **不进指纹**：清单哈希只覆盖 manifest 内容（corpusmod），
      本文件是目录内的旁路审计产物，不参与指纹、不影响评测。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import tempfile
import fcntl
from contextlib import contextmanager

from collections import Counter


import yaml

LOG_FILENAME = "generation-log.yaml"
LOG_FORMAT = 1


def report_frozen_corpus(cases, manifest, rate, feature_version, out, written, *, version, column_width):
    """Emit the historical builders' summary and return its size buckets."""
    from .console import emit
    from .coverage import size_class

    by_operator_size = Counter()
    for case in cases:
        elements = 1
        for size in case.shape:
            elements *= size
        by_operator_size.update((f"{case.operator}/{size_class(elements)}",))
    emit(f"\n已冻结 {version}：算子 {len(manifest.operators)} 个，"
         f"用例 {len(manifest.cases)} 个，FA 覆盖 {rate:.1%}（v{feature_version}）")
    for key, count in sorted(by_operator_size.items()):
        emit(f"  {key:<{column_width}} {count}")
    emit(f"清单哈希 {manifest.manifest_hash}")
    emit(f"写入 {out}" if written else f"内容未变，未重写 {out}")
    return by_operator_size

_LICENSE_HEADER = (
    "# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.\n"
    "# This program is free software, you can redistribute it and/or modify it under the terms and conditions of\n"
    '# CANN Open Software License Agreement Version 2.0 (the "License").\n'
    "# Please refer to the License for details. You may not use this file except in compliance with the License.\n"
    '# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,\n'
    "# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.\n"
    "# See LICENSE in the root of the software repository for the full text of the License.\n"
)


def _read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, list) or any(not isinstance(record, dict) for record in data):
        raise ValueError("generation log must contain a list of event mappings")
    return data


def read_log(path) -> list[dict]:
    """Read YAML logs and historical JSONL without rewriting either format."""
    return _read_records(Path(path))


def write_log(path, records: list[dict]) -> None:
    """Atomically replace YAML output while retaining every recorded event."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(records, sort_keys=False, allow_unicode=True, width=4096)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".generation-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_LICENSE_HEADER + "\n" + body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _log_lock(path):
    lock_path = path.with_name("." + path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class GenerationLog:
    """每步一事件：{"ts": ..., "step": ..., ...数据}。"""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock(self.path):
            self._records = _read_records(self.path)
            if not self.path.is_file():
                self._records.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "step": "log_header", "log_format": LOG_FORMAT,
                    "note": "License header followed by a YAML event list; timestamps are UTC ISO8601",
                })
                self._flush()

    def step(self, step: str, **data: object) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "step": step}
        record.update(data)
        with _log_lock(self.path):
            self._records = _read_records(self.path)
            self._records.append(record)
            self._flush()

    def _flush(self) -> None:
        write_log(self.path, self._records)


def builder_arguments():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=20,
                    help="每个 category 的用例数（规模旋钮）")
    ap.add_argument("--device", default="cpu",
                    help="参考自检设备；昇腾机上可用 npu:0")
    ap.add_argument("--no-verify-refs", dest="verify_refs", action="store_false",
                    default=True)
    args = ap.parse_args()
    return args


def run_reference_gate(cases, device, corpus_dir, excluded):
    """Filter failed references identically for the historical corpus builders."""
    import sys
    from importlib import import_module

    from .console import emit
    from .refcheck import check_all, torch_available

    if not torch_available():
        emit("（无 torch，跳过参考实现自检——不应在冻结机上发生）", file=sys.stderr)
        return cases, True
    if device.startswith("npu"):
        import_module("torch_npu")
    failures = check_all(cases, device, corpus_dir=corpus_dir)
    if not failures:
        return cases, False
    retained = [case for case in cases if case.code not in failures]
    for code, why in sorted(failures.items()):
        excluded.append({"code": code, "reason": f"参考实现自检：{why}"})
    emit(f"参考自检（device={device}）：剔除 {len(failures)} 个")
    for code, why in sorted(failures.items()):
        emit(f"  {code}: {why}")
    return retained, False


def select_manifest_candidates(candidates, per_category, atom_universe):
    """Select and report common coverage statistics without changing builder logs."""
    from .console import emit
    from .coverage import guided_selection
    from .operator_registry import by_name

    selected, report = guided_selection(
        candidates, per_category=per_category, atom_universe=atom_universe,
        category_of=lambda case: by_name(case.operator).category.value if by_name(case.operator) else "")
    emit(f"引导选择：{report['selected']} 个（池 {report['pool_size']}，每类配额 {per_category}）")
    stats = report["coverage"]
    emit(f"选择后覆盖 {len(stats['covered'])}/{len(atom_universe)}（{stats['rate']:.1%}）")
    if stats["missing"]:
        emit(f"  未覆盖：{', '.join(stats['missing'])}")
    return selected, report
