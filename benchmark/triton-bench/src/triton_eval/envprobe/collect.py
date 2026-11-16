# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""环境采集（L1）——只能在昇腾执行机上运行。

只做「执行并把原始输出落盘」，解析交给 parse.py（L0）。
这样只要采集过一次，解析逻辑就能长期在没有 NPU 的机器上测试。
"""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parse import build_env, env_fingerprint
from ..console import emit


def kernel_dependencies_available(device: str) -> bool:
    try:
        importlib.import_module("torch")
        importlib.import_module("triton")
        if device.startswith("npu"):
            importlib.import_module("torch_npu")
    except ImportError as exc:
        emit(f"探针需要 torch/triton（NPU 设备 {device}）：{exc}", file=sys.stderr)
        return False
    return True


def run_cmd(cmd: list[str], timeout: int = 30) -> str:
    """执行命令并返回原始文本输出。失败时返回带标记的空串，不抛异常。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.stdout or ""
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<command failed: {exc}>"


def collect_raw() -> dict[str, str]:
    """采集硬件与软件栈的原始输出。"""
    return {
        "npu_smi_info": run_cmd(["npu-smi", "info"]),
        "npu_smi_version": run_cmd(["npu-smi", "version"]),
        "uname_r": run_cmd(["uname", "-r"]) or platform.release(),
        "cann_home": os.environ.get("ASCEND_HOME_PATH", ""),
    }


def venv_lock_hash() -> str:
    """当前 venv 的依赖锁定哈希。用于判断被测编译器环境是否被重建过。"""
    freeze = run_cmd([sys.executable, "-m", "pip", "freeze"], timeout=120)
    return hashlib.sha256(freeze.encode("utf-8")).hexdigest()[:16]


def _torch_versions() -> tuple[str, str]:
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        torch_version = ""
    try:
        import torch_npu  # noqa: F401

        torch_npu_version = getattr(torch_npu, "__version__", "") or ""
    except Exception:
        torch_npu_version = ""
    return torch_version, torch_npu_version


def triton_import_version() -> str:
    """实测 import 得到的 triton 版本。

    不能用 pip 元数据：triton-ascend 会接管 triton 目录，
    实测环境下元数据是 3.5.0 而实际是 3.2.0。
    """
    try:
        import triton

        return triton.__version__
    except Exception:
        return ""


def probe_env(
    *,
    compiler_name: str,
    compiler_version: str,
    compiler_commit: str | None,
    compiler_install_mode: str,
    clock_locked: bool = False,
    env_health: str = "ok",
    env_health_notes: str = "",
    toolchain_commit: str | None = None,
) -> tuple[dict[str, Any], str]:
    """采集并构造环境指纹。返回 (env 字典, env_fp)。"""
    raw = collect_raw()
    torch_version, torch_npu_version = _torch_versions()

    env = build_env(
        arch=platform.machine(),
        os_name=f"{platform.system()} {platform.release()}",
        kernel=raw["uname_r"].strip(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        torch_npu_version=torch_npu_version,
        compiler_name=compiler_name,
        compiler_version=compiler_version,
        compiler_commit=compiler_commit,
        compiler_install_mode=compiler_install_mode,
        triton_import_version=triton_import_version(),
        venv_path=sys.prefix,
        venv_lock_hash=venv_lock_hash(),
        clock_locked=clock_locked,
        npu_id=os.environ.get("ASCEND_VISIBLE_DEVICES", "0"),
        ascend_visible_devices=os.environ.get("ASCEND_VISIBLE_DEVICES", "0"),
        npu_smi_info_text=raw["npu_smi_info"],
        npu_smi_version_text=raw["npu_smi_version"],
        cann_home=raw["cann_home"],
        env_health=env_health,
        env_health_notes=env_health_notes,
        toolchain_commit=toolchain_commit,
    )
    env["probed_at"] = datetime.now(timezone.utc).isoformat()
    return env, env_fingerprint(env)


def save_raw(raw: dict[str, str], target_dir: str | Path) -> None:
    """把原始输出落盘，供 fixture 与事后追溯使用。"""
    d = Path(target_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, text in raw.items():
        (d / f"{name}.txt").write_text(text or "", encoding="utf-8")


def probe_remote_health() -> dict:
    import torch
    import torch_npu
    import triton
    import triton.language.extra.cann
    import yaml
    import pydantic
    import jinja2

    if not torch.npu.is_available() or torch.npu.device_count() < 1:
        raise RuntimeError("NPU unavailable")
    # Real allocation, device execution, synchronization and CPU round trip.
    value = torch.ones(4, device="npu") + 1
    torch.npu.synchronize()
    if value.cpu().tolist() != [2.0] * 4:
        raise RuntimeError("NPU probe result mismatch")
    return dict(
        ok=True,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        devices=torch.npu.device_count(),
    )
