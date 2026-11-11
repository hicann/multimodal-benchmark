# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Offline acceptance checks for complete, partial, and corrupted result sets."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_run_completeness.py"
MODULE_SPEC = importlib.util.spec_from_file_location("check_run_completeness", SCRIPT)
AUDIT = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(AUDIT)


def result(code="a", operator="Add", status="PASS", run_id=1):
    return json.dumps({"case_code": code, "operator": operator,
                       "status": status, "run_id": run_id})


class RunCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"version": "test", "cases": [
            {"code": "a", "operator": "Add"},
            {"code": "b", "operator": "Mul"},
        ]}

    def test_complete_results_can_contain_real_failures(self):
        summary = AUDIT.audit_results(self.manifest, [result(), result("b", "Mul", "COMPILE_FAIL")])
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["all_pass"])
        self.assertEqual(summary["by_status"], {"COMPILE_FAIL": 1, "PASS": 1})

    def test_duplicate_cannot_replace_missing_case(self):
        summary = AUDIT.audit_results(self.manifest, [result(), result()])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["missing"], ["b"])
        self.assertEqual(summary["duplicates"], {"a": 2})

    def test_wrong_operator_and_foreign_case_are_rejected(self):
        summary = AUDIT.audit_results(self.manifest, [result(operator="Mul"), result("foreign")])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["unexpected"], ["foreign"])
        self.assertEqual(summary["operator_mismatches"][0]["case_code"], "a")

    def test_partial_json_unknown_status_and_mixed_runs_are_rejected(self):
        for lines in ([result(), "{"], [result(), result("b", "Mul", "RUNNING")],
                      [result(), result("b", "Mul", run_id=2)], [result(), "[]"],
                      [result(), result("b", "Mul", run_id=True)]):
            with self.subTest(lines=lines):
                summary = AUDIT.audit_results(self.manifest, lines)
                self.assertFalse(summary["complete"])
                self.assertTrue(summary["errors"])

    def test_empty_results_are_incomplete(self):
        summary = AUDIT.audit_results(self.manifest, [])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["missing"], ["a", "b"])

    def test_cli_distinguishes_complete_from_all_pass(self):
        lines = "\n".join([result(), result("b", "Mul", "WRONG_RESULT")])
        with patch.object(Path, "read_text", return_value=json.dumps(self.manifest)), \
                patch.object(AUDIT.sys, "stdin", lines.splitlines()), \
                patch("builtins.print"):
            self.assertEqual(AUDIT.main(["--results", "-"]), 0)
            self.assertEqual(AUDIT.main(["--results", "-", "--require-pass"]), 1)


if __name__ == "__main__":
    unittest.main()
