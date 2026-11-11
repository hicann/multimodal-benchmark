# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""测试集的冻结、校验与确定性筛查（L0，无需 NPU）。

「采集 / 解析分离」在本模块的体现：
    确定性筛查的**判定逻辑**是纯函数，接受一个执行器回调。
    在昇腾主机上传入真实执行器；在本地传入假执行器即可完整测试筛查逻辑。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from .models import CaseSpec, CorpusManifest


class CaseRunner(Protocol):
    """用例执行器协议。真实实现需要 NPU，测试时可传入假实现。"""

    def __call__(self, case: CaseSpec) -> str:
        """返回该次执行结果的稳定指纹（用于判断是否可复现）。"""


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_case_content_hash(corpus_dir: Path, case: CaseSpec) -> str:
    """用例内容哈希。

    契约变更后用例**不再绑定 kernel**，因此哈希只覆盖用例自身字段
    （code / operator / dtype / shape / seed / attrs 等）。

    这带来一个重要性质：哈希标识的是「要测什么」，
    与「谁来实现」无关——正是支持外部提交所需要的。
    """
    return hashlib.sha256(_canonical(case.model_dump(exclude={"content_hash"}))).hexdigest()


def manifest_digest(manifest: CorpusManifest) -> str:
    """清单哈希 —— 只覆盖「内容」，不覆盖「时间」。

    曾经把 frozen_at 也算进去，导致同一份 corpus 每次重新生成都得到
    不同的哈希（实测连续两次：f2cf68f9... 与 43e0b894...）。
    后果很严重：
      - 无法判断两次运行用的是否是同一份 corpus（内容相同却哈希不同）
      - 「冻结」失去意义——本该标识内容的指纹却在随时间变化
      - 每次 build_corpus 都会让 manifest.json 产生 diff，污染 git

    frozen_at 是元信息不是内容，故参与排除。
    """
    return hashlib.sha256(
        _canonical(
            manifest.model_dump(exclude={"manifest_hash", "frozen_at"})
        )
    ).hexdigest()


def build_manifest(
    corpus_dir: Path,
    version: str,
    spec_version: str,
    operators: list,
    cases: list[CaseSpec],
    feature_atoms_version: str = "",
    excluded: list[dict[str, str]] | None = None,
) -> CorpusManifest:
    """构建冻结清单：补全内容哈希与清单哈希。"""
    for case in cases:
        case.content_hash = compute_case_content_hash(corpus_dir, case)
    manifest = CorpusManifest(
        version=version,
        spec_version=spec_version,
        feature_atoms_version=feature_atoms_version,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        operators=operators,
        cases=cases,
        excluded=excluded or [],
    )
    manifest.manifest_hash = manifest_digest(manifest)
    return manifest


def verify_manifest(manifest: CorpusManifest) -> None:
    """校验清单自洽：哈希必须能重算出来。被篡改则抛错。"""
    expected = manifest_digest(manifest)
    if expected != manifest.manifest_hash:
        raise ValueError(
            f"清单哈希不匹配：期望 {manifest.manifest_hash}，重算得到 {expected}。"
            "corpus 在冻结后被修改过"
        )


def verify_corpus_files(corpus_dir: Path, manifest: CorpusManifest) -> None:
    """校验每个用例的 kernel 文件存在且内容哈希未变。"""
    verify_manifest(manifest)
    from .corpus_assets import verify_assets

    verify_assets(corpus_dir, manifest.manifest_hash, required=False)
    for case in manifest.cases:
        actual = compute_case_content_hash(corpus_dir, case)
        if actual != case.content_hash:
            raise ValueError(
                f"用例 {case.code} 内容哈希不匹配：清单记录 {case.content_hash}，"
                f"实际 {actual}"
            )


def screen_determinism(
    cases: list[CaseSpec],
    runner: CaseRunner,
    repeats: int = 5,
) -> tuple[list[CaseSpec], list[dict[str, str]]]:
    """确定性筛查：连续执行多次，结果指纹不一致的用例被剔除。

    不确定性是 corpus 的质量问题，在冻结前筛掉，
    而不是在评测时给它一个特殊状态（那会污染评分分母）。
    """
    kept: list[CaseSpec] = []
    excluded: list[dict[str, str]] = []
    for case in cases:
        seen: set[str] = set()
        for _ in range(repeats):
            seen.add(runner(case))
        if len(seen) == 1:
            kept.append(case)
        else:
            excluded.append(
                {
                    "code": case.code,
                    "reason": f"重复执行 {repeats} 次得到 {len(seen)} 种不同结果，判定为不确定",
                }
            )
    return kept, excluded


def load_manifest(path: Path) -> CorpusManifest:
    return CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def select_cases(cases: list[CaseSpec], operators_csv: str | None) -> list[CaseSpec]:
    """按算子过滤用例（run 命令的 --operators，逗号分隔）。

    名字两种写法都认：算子 ID（用例编号前缀，如 grouped_matmul）或注册表
    类名（如 GroupedMatmul）。父进程与评测子进程必须用同一份过滤逻辑，
    否则事件预算与实际执行的用例会对不上。

    为什么需要它（2026-10-12 实测约束）：真机环境**两小时自动清空**，
    冒烟全量 + standard 全量测速跑不完一个窗口。性能锚点本来就**按算子
    持有**（见 anchors.py），按算子分窗跑：窗口内全对的算子照样冻结自己
    的锚点、拿到真实的加速比与性能分，跨窗口逐步攒齐全部算子。
    """
    if not operators_csv:
        return cases
    wanted = {name.strip().lower() for name in operators_csv.split(",") if name.strip()}
    kept = [c for c in cases
            if wanted & {c.code.split("-", 1)[0].lower(), c.operator.lower()}]
    if not kept:
        raise ValueError(
            f"--operators 过滤后没有用例（给了 {sorted(wanted)}）；"
            "算子名以用例编号前缀为准（如 cross_entropy_loss,transpose）")
    return kept


def write_manifest(manifest: CorpusManifest, path: Path) -> bool:
    """写出清单。内容未变时不重写，返回是否实际写入。

    为什么要幂等：
        frozen_at 是生成时刻的时间戳，每次调用都不同。即使 manifest_hash
        已排除它（不再影响哈希），它仍会被写进文件，导致 manifest.json
        每次都产生 git diff —— 进而让 `git pull` 报
        「本地修改会被覆盖」，被迫手工 checkout。

    做法：若文件已存在且 manifest_hash 相同，则保留原 frozen_at 并跳过写入。
    这样「冻结时间」语义也正确——它是首次冻结的时刻，不是最后一次重生成的时刻。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        try:
            existing = CorpusManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001 - 文件损坏则直接覆盖重写
            existing = None
        if existing is not None and existing.manifest_hash == manifest.manifest_hash:
            manifest.frozen_at = existing.frozen_at  # 保留首次冻结时间
            return False

    path.write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True


def results_jsonl_path(run_dir: Path) -> Path:
    return run_dir / "results.jsonl"


def append_result_line(run_dir: Path, record) -> None:
    """把一条结果追加进 results.jsonl（真相源，一行一条）。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    with results_jsonl_path(run_dir).open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def read_result_lines(path: Path) -> list:
    from .models import CaseResultRecord

    return [
        CaseResultRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iter_result_lines(path: Path) -> Callable[[], list]:
    return lambda: read_result_lines(path)
