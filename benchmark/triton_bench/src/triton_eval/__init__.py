# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""triton-eval：面向 Triton 编译器的功能完备性与综合性能评测体系（MVP）。

分层约定：
    L0 —— 无 NPU 依赖，macOS / Linux 通用：规范加载、corpus 校验、判定、评分、存储、报告
    L1 —— 只能在昇腾主机运行：envprobe 采集、bench 计时、npu_exec 编译执行

两层的唯一契约是 `runs/<id>/results.jsonl`。
"""


__version__ = "0.1.0"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE_GROUPS = {
    '_coverage': (
        'backend_capture',
        'backend_evidence',
        'compilation_evidence',
        'coverage_evidence',
        'coverage_gate',
        'coverage_generation',
        'coverage_overlay',
        'coverage_reference',
        'implementation_evidence',
        'pattern_contracts',
        'pattern_evidence',
        'pattern_generation',
        'pattern_release',
        'reviewed_kernels',
    ),
    '_references': (
        'official_reference',
        'official_reference_cpu',
        'refcheck',
        'reference_execution',
        'oracle',
        'official_catalog',
        'official_generation',
        'ascend_contracts',
        'ascend_lowering',
    ),
    '_submissions': (
        'independent_submissions',
        'submission_http',
        'submission_preflight',
        'submission_service',
        'submission_simulation',
    ),
}

__path__.extend(str(Path(__file__).parent / group) for group in _MODULE_GROUPS)


def framework_path(relative: str, root: Path | None = None) -> Path:
    """Resolve grouped source assets without changing public module names."""
    relative_path = Path(relative)
    project_root = PROJECT_ROOT if root is None else root
    if len(relative_path.parts) == 3 and relative_path.parts[:2] == ("src", "triton_eval"):
        for group, modules in _MODULE_GROUPS.items():
            if relative_path.stem in modules:
                return project_root / "src/triton_eval" / group / relative_path.name
    from .fixture_sources import fixture_path

    return fixture_path(project_root / relative_path, project_root)

