# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""环境指纹解析（L0）。

输入是 npu-smi 等命令的**原始文本输出**（由 L1 采集落盘），
输出是规范化的字典 + 确定性哈希。

这样设计的好处：把真实输出存成 fixture 后，解析逻辑在没有 NPU 的机器上
也能完整测试——环境一旦采集过一次，就长期可测。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


COVERAGE_ENVIRONMENT_KEYS = (
    "arch",
    "npu_model",
    "cann_version",
    "compiler_name",
    "compiler_version",
    "triton_import_version",
    "torch_version",
    "torch_npu_version",
    "venv_lock_hash",
)

# 不同环境的 npu-smi 输出格式不同，实测见过两种：
#   旧环境：| 3     Ascend910          | OK |   → Ascend910
#   新环境：| 4     910B4              | OK |   → 910B4
# 用两个精确分支而非宽松通配，避免把版本号 25.5.1 之类误当成型号
MODEL_RE = re.compile(r"\b(Ascend\s*\d+\w*|\d{3}[A-Z]\w*)\b")
HBM_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")
# 芯片行含 PCI Bus-Id，用它把真正的 HBM 行挑出来，
# 否则表头的 Hugepages 列（形如 `0 / 0`）会被误当成 HBM 用量
CHIP_LINE_RE = re.compile(r"\d{4}:[0-9A-Fa-f]{2}:")
TOOL_VERSION_RE = re.compile(r"npu-smi\s+(\d+\.\d+(?:\.\d+)?)")


def parse_npu_smi_info(text: str) -> dict[str, Any]:
    """从 npu-smi info 输出中抽取硬件事实。

    型号行形如 `| 4     910B4    | OK |`，需跳过表头（含 "NPU"、"Name" 等）。
    """
    out: dict[str, Any] = {}

    model = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        # 跳过表头与分隔行
        if "Name" in line or "===" in line or "---" in line:
            continue
        m = MODEL_RE.search(line)
        if m:
            model = m.group(1).replace(" ", "")
            break
    out["npu_model"] = model

    tool = TOOL_VERSION_RE.search(text)
    out["tool_version"] = tool.group(1) if tool else None

    hbm: list[tuple[int, int]] = []
    for line in text.splitlines():
        if not CHIP_LINE_RE.search(line):
            continue
        pairs = HBM_RE.findall(line)
        if pairs:
            # 芯片行上 Memory-Usage 在 HBM 之前（形如 `0 / 0`），取最后一个才是 HBM
            used, total = pairs[-1]
            hbm.append((int(used), int(total)))
    if not hbm:
        # 回退：输出格式不含 Bus-Id 时，取所有 X / Y 对
        hbm = [(int(a), int(b)) for a, b in HBM_RE.findall(text)]

    if hbm:
        out["hbm_total_mb"] = max(total for _, total in hbm)
        out["hbm_used_mb"] = [used for used, _ in hbm]
    else:
        out["hbm_total_mb"] = None
        out["hbm_used_mb"] = []

    out["chip_count"] = len(out["hbm_used_mb"])
    return out


def parse_npu_smi_version(text: str) -> str | None:
    m = VERSION_RE.search(text)
    return m.group(1) if m else None


def parse_cann_version(cann_home: str) -> str | None:
    """从 CANN 安装路径中抽取版本，如 /home/x/Ascend/cann-9.0.0 → 9.0.0。"""
    m = re.search(r"cann-(\d+\.\d+(?:\.\w+)?)", cann_home or "")
    return m.group(1) if m else None


def build_env(
    *,
    arch: str,
    os_name: str,
    kernel: str,
    python_version: str,
    torch_version: str,
    torch_npu_version: str,
    compiler_name: str,
    compiler_version: str,
    compiler_commit: str | None,
    compiler_install_mode: str,
    triton_import_version: str,
    venv_path: str,
    venv_lock_hash: str,
    clock_locked: bool,
    npu_id: str,
    ascend_visible_devices: str,
    npu_smi_info_text: str = "",
    npu_smi_version_text: str = "",
    cann_home: str = "",
    driver_version: str | None = None,
    env_health: str = "ok",
    env_health_notes: str = "",
    toolchain_commit: str | None = None,
) -> dict[str, Any]:
    """构造环境指纹字典。字段顺序不影响结果——规范化时会排序。"""
    hw = parse_npu_smi_info(npu_smi_info_text) if npu_smi_info_text else {
        "npu_model": None, "hbm_total_mb": None, "hbm_used_mb": [], "chip_count": 0
    }

    env: dict[str, Any] = {
        "arch": arch,
        "os": os_name,
        "kernel": kernel,
        "npu_model": hw["npu_model"],
        "npu_count": hw["chip_count"],
        "npu_id": npu_id,
        "ascend_visible_devices": ascend_visible_devices,
        "driver_version": driver_version
        or (parse_npu_smi_version(npu_smi_version_text) if npu_smi_version_text else None),
        "cann_version": parse_cann_version(cann_home),
        "cann_path": cann_home or None,
        "python_version": python_version,
        "torch_version": torch_version,
        "torch_npu_version": torch_npu_version,
        "compiler_name": compiler_name,
        "compiler_version": compiler_version,
        "compiler_commit": compiler_commit,
        "compiler_install_mode": compiler_install_mode,
        # 实测 import 值，而不是 pip 元数据：triton-ascend 会覆盖 triton 目录，
        # 二者在实测环境下并不一致（元数据 3.5.0 / 实测 3.2.0）
        "triton_import_version": triton_import_version,
        "venv_path": venv_path,
        "venv_lock_hash": venv_lock_hash,
        "clock_locked": clock_locked,
        "env_health": env_health,
        "env_health_notes": env_health_notes,
        "toolchain_commit": toolchain_commit,
    }
    return env


#: 不参与指纹的易变字段——它们描述「这次探测的时刻与产物状态」而非
#: 「环境身份」，一变指纹就变，同环境的两轮会被误报为跨环境不可比。
#: 2026-10-12 真机实测：同窗口两轮（同机同卡相隔 47 分钟）仅因
#: probed_at（探测时间戳）与 venv_lock_hash（每次 pip install 抖动）
#: 不同，compare 即警告「跨环境对比需谨慎」。与 C12 的 frozen_at
#: 教训同源：指纹必须只覆盖身份，不覆盖时间。
FP_VOLATILE_KEYS: frozenset[str] = frozenset({
    "probed_at",         # 探测时间戳：每轮必不同
    "venv_lock_hash",    # pip freeze 哈希，重装即抖；版本身份已由各 *_version 字段承担
    "npu_smi_info_text",  # npu_smi 原始输出，可能含功耗/温度等实时读数
})


def env_fingerprint(env: dict[str, Any]) -> str:
    """确定性指纹（仅环境身份字段，易变字段见 FP_VOLATILE_KEYS）。

    键排序后序列化，保证同样的环境永远得到同样的值。
    """
    stable = {k: v for k, v in env.items() if k not in FP_VOLATILE_KEYS}
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def comparable(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """两次运行是否可直接比较——指纹不同即不可比。"""
    return env_fingerprint(a) == env_fingerprint(b)
