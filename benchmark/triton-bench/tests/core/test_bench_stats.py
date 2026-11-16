# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""性能统计与环境指纹解析的单元测试（L0，无需 NPU）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton_eval.bench.stats import (
    cv,
    geometric_mean,
    is_stable,
    mad,
    median,
    speedup,
    summarize,
)
from triton_eval.envprobe.parse import (
    build_env,
    comparable,
    env_fingerprint,
    parse_cann_version,
    parse_npu_smi_info,
    parse_npu_smi_version,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


# ---------- 统计（配合任务 5.3 的 L0 部分） ----------


def test_median() -> None:
    assert median([0.412, 0.409, 0.415, 0.410, 0.411]) == pytest.approx(0.411)
    assert median([]) is None


def test_mad() -> None:
    assert mad([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cv() -> None:
    assert cv([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert cv([1.0, 3.0]) == pytest.approx(0.5)
    assert cv([1.0]) is None, "样本不足无法算 CV"


def test_speedup() -> None:
    assert speedup(1.204, 0.411) == pytest.approx(1.204 / 0.411)
    assert speedup(None, 0.4) is None
    assert speedup(1.0, 0.0) is None, "零耗时无法算加速比"


def test_is_stable_gate() -> None:
    assert is_stable(0.01, 0.03) is True
    assert is_stable(0.05, 0.03) is False


def test_geometric_mean() -> None:
    assert geometric_mean([2.0, 8.0]) == pytest.approx(4.0)
    assert geometric_mean([]) is None
    assert geometric_mean([0.0, 2.0]) == pytest.approx(2.0), "非正值被剔除"


def test_summarize_with_real_fixture_samples() -> None:
    """用真实采集的样本验证统计链路。

    注意：分位数与中位数来自**不同次**测量（实测 0.0350 vs 0.0406），
    不能混成一个样本数组。夹具里用 samples_ms 承载同一次批量测量的
    样本序列，quantiles 仅作参考。
    """
    payload = json.loads((FIXTURES / "samples_ms.json").read_text(encoding="utf-8"))
    samples = payload.get("samples_ms")
    if not samples:
        pytest.skip("夹具未含 samples_ms（需重新采集：bash env/collect-fixtures.sh）")

    assert len(samples) >= 3, "样本数不足，无法计算 CV"
    s = summarize(samples)
    assert s.median_ms == pytest.approx(payload["median_ms"], rel=1e-6)
    assert s.mad_ms is not None and s.mad_ms >= 0
    assert s.cv is not None
    # 真实测量存在波动，但不该大到离谱；这正是不稳定门禁存在的理由
    assert s.cv < 0.5, f"样本波动异常：cv={s.cv}, samples={samples}"


# ---------- 环境指纹（任务 5.2） ----------


def test_parse_real_npu_smi_output() -> None:
    text = (FIXTURES / "npu_smi_info.txt").read_text(encoding="utf-8")
    info = parse_npu_smi_info(text)
    assert info["npu_model"] == "Ascend910"
    assert info["hbm_total_mb"] == 65536
    assert info["chip_count"] == 2, "两张卡，Hugepages 的 0/0 不能被误计"
    assert info["tool_version"] == "25.5.5"


def test_parse_model_without_ascend_prefix() -> None:
    r"""型号不带 Ascend 前缀的环境也要能识别。

    实测两种 npu-smi 输出格式：
        旧环境：| 3     Ascend910          | OK |
        新环境：| 4     910B4              | OK |
    曾经只匹配 `Ascend\d+`，在新环境上 npu_model 为 None。
    """
    text = (FIXTURES / "npu_smi_info_b.txt").read_text(encoding="utf-8")
    info = parse_npu_smi_info(text)
    assert info["npu_model"] == "910B4", f"实际解析为 {info['npu_model']}"
    assert info["hbm_total_mb"] == 32768
    assert info["chip_count"] == 1
    assert info["tool_version"] == "25.5.1"


def test_model_is_not_confused_with_version_number() -> None:
    """不能把 npu-smi 的版本号 25.5.1 误当成芯片型号。"""
    text = (FIXTURES / "npu_smi_info_b.txt").read_text(encoding="utf-8")
    info = parse_npu_smi_info(text)
    assert info["npu_model"] != "25"
    assert info["npu_model"] != "255"


def test_parse_ignores_hugepages_column() -> None:
    """表头的 Hugepages-Usage 形如 `0 / 0`，不能被当成 HBM。"""
    info = parse_npu_smi_info(
        "| NPU  Name | Health | Power(W)  Temp(C)  Hugepages-Usage(page)|\n"
        "| 3    Ascend910 | OK | 165.1  44   0    / 0             |\n"
        "| 0    6    | 0000:0A:00.0 | 0   0 / 0   2909 / 65536  |\n"
    )
    assert info["chip_count"] == 1
    assert info["hbm_total_mb"] == 65536


def test_parse_cann_version_from_path() -> None:
    assert parse_cann_version("/home/developer/Ascend/cann-9.0.0") == "9.0.0"
    assert parse_cann_version("") is None


def test_parse_npu_smi_version() -> None:
    assert parse_npu_smi_version("npu-smi 25.5.5") == "25.5.5"
    assert parse_npu_smi_version("no version here") is None


def _env(**over: object) -> dict:
    base: dict = dict(
        arch="aarch64", os_name="Ubuntu 20.04.5", kernel="5.10.0-hce2",
        python_version="3.11.4", torch_version="2.7.1+cpu",
        torch_npu_version="2.7.1.post4", compiler_name="triton-ascend",
        compiler_version="3.2.1", compiler_commit=None,
        compiler_install_mode="wheel", triton_import_version="3.2.0",
        venv_path="/home/developer/venv/te-compiler", venv_lock_hash="sha256:abc",
        clock_locked=False, npu_id="0", ascend_visible_devices="0",
        npu_smi_info_text=(FIXTURES / "npu_smi_info.txt").read_text(encoding="utf-8"),
        cann_home="/home/developer/Ascend/cann-9.0.0",
    )
    base.update(over)
    return build_env(**base)  # type: ignore[arg-type]


def test_env_built_from_real_probe_output() -> None:
    env = _env()
    assert env["arch"] == "aarch64"
    assert env["npu_model"] == "Ascend910"
    assert env["npu_count"] == 2
    assert env["cann_version"] == "9.0.0"
    assert env["triton_import_version"] == "3.2.0"
    assert env["clock_locked"] is False


def test_fingerprint_is_deterministic() -> None:
    """同样的环境必须永远得到同样的指纹，否则跨 run 比对无意义。"""
    assert env_fingerprint(_env()) == env_fingerprint(_env())


def test_fingerprint_ignores_volatile_fields() -> None:
    """易变字段不参与指纹（2026-10-12 真机实测穿帮后修复）。

    同窗口两轮（同机同卡相隔 47 分钟）仅因 probed_at（探测时间戳）与
    venv_lock_hash（每次 pip install 抖动）不同，compare 就误报
    「跨环境对比需谨慎」——指纹必须只覆盖环境身份，不覆盖时间
    （与 C12 的 frozen_at 教训同源）。
    """
    a = _env()
    b = _env()
    # 易变字段直接塞进 dict（probed_at 不走 build_env——它是采集后才加的）
    b.update(probed_at="2026-10-12T09:22:17+00:00",
             venv_lock_hash="deadbeef1234",
             npu_smi_info_text="Name  Health  Power(W)  Temp(C)\n910B4  OK  302.1  42")
    assert env_fingerprint(a) == env_fingerprint(b), "易变字段不得影响指纹"
    assert comparable(a, b) is True, "同环境两轮必须直接可比"


def test_fingerprint_changes_when_env_changes() -> None:
    assert env_fingerprint(_env()) != env_fingerprint(_env(compiler_version="3.2.2"))


def test_fingerprint_ignores_key_order() -> None:
    a = _env()
    b = dict(reversed(list(a.items())))
    assert env_fingerprint(a) == env_fingerprint(b)


def test_comparability() -> None:
    assert comparable(_env(), _env()) is True
    assert comparable(_env(), _env(clock_locked=True)) is False
