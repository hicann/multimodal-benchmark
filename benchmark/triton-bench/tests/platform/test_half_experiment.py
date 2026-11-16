# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import importlib.util
from pathlib import Path
import zipfile
import yaml


def test_prepared_half_is_repeatable_covers_extensions_and_keeps_full_operator_cases(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "half_prepare", root / "scripts/platform_examples/prepare_half_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.prepare(tmp_path / "one")
    again = module.prepare(tmp_path / "two")
    assert plan == again
    assert (tmp_path / "one/submission.zip").read_bytes() == (tmp_path / "two/submission.zip").read_bytes()
    assert plan["operator_count"] == 37 and plan["cases"] == 746 and len(plan["categories"]) == 12
    assert set(module.TARGETS) <= set(plan["operators"])
    assert module.prepare(tmp_path / "one") == plan
    with zipfile.ZipFile(tmp_path / "one/submission.zip") as archive:
        payload = yaml.safe_load(archive.read("manifest.yaml"))
        assert {o["name"] for o in payload["operators"]} == set(plan["operators"])
        assert all(n == "manifest.yaml" or n.endswith(".py") for n in archive.namelist())
