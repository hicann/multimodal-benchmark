# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Public submission limits shared with the actual HTTP and package validators."""

from triton_eval import package

MAX_REQUEST_BYTES = 64 * 1024 * 1024


def submission_policy() -> dict:
    return dict(
        content_type="application/zip",
        max_request_bytes=MAX_REQUEST_BYTES,
        max_entries=package.MAX_ENTRIES,
        max_total_bytes=package.MAX_TOTAL_BYTES,
        max_file_bytes=package.MAX_FILE_BYTES,
        max_compression_ratio=package.MAX_RATIO,
        allowed_suffixes=sorted(package.ALLOWED_SUFFIXES),
        required_manifest="manifest.yaml",
        rules=[dict(rule_id=key, detail=value) for key, value in sorted(package.RULES.items())],
    )


def profile_capabilities() -> list[dict]:
    # These describe the evaluator's capabilities, not certification of a run.
    return [
        dict(
            id="smoke",
            label="功能快测",
            default=True,
            experimental=False,
            performance_scoring=False,
            performance_accepted=False,
            full_suite_accepted=False,
        ),
        dict(
            id="standard",
            label="标准评测（试运行）",
            default=False,
            experimental=True,
            performance_scoring=True,
            performance_accepted=False,
            full_suite_accepted=False,
        ),
    ]
