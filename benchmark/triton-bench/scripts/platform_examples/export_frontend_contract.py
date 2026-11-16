#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Export the current API and reproducible, non-executing frontend examples.

Uses a temporary service with no Worker, cloud configuration or socket. Examples
are service responses, not NPU results. IDs/times are normalized for review.
"""

from __future__ import annotations


import argparse
from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
import zipfile

from triton_eval.console import emit

from triton_eval.submission_http import openapi
from triton_eval.submission_service import EvaluationService, ServiceError
from triton_eval.submission_simulation import DEFAULT_CORPUS

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/frontend-integration"


def package_bytes() -> bytes:
    stream = BytesIO()
    files = {
        "manifest.yaml": "submit_id: frontend-contract\noperators:\n" "- name: Exp\n  entry: exp\n  path: kernel.py\n",
        "kernel.py": "def exp(x, base=-1.0, scale=1.0, shift=0.0):\n    return x\n",
    }
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), content)
    return stream.getvalue()


class Scenarios:
    def __init__(self, service):
        self.service = service
        self.items = []

    def record(self, name, method, path, status, response, request=None):
        value = dict(name=name, method=method, path=path, http_status=status, response=response)
        if request is not None:
            value["request"] = request
        self.items.append(value)

    def submissions(self):
        self.record("catalog-worker-disabled", "GET", "/api/v1/catalog", 200, self.service.catalog())
        accepted = self.service.upload(package_bytes(), key="frontend-upload")
        rejected = self.service.upload(b"not-a-zip")
        if accepted["status"] != "READY" or rejected["status"] != "REJECTED":
            raise ValueError("Regression acceptance condition failed")
        self.record(
            "submission-ready", "POST", "/api/v1/submissions", 201, accepted,
            dict(content_type="application/zip", body_fixture="submission-valid.zip",
                 headers={"Idempotency-Key": "frontend-upload"}),
        )
        self.record(
            "submission-rejected-is-still-201", "POST", "/api/v1/submissions", 201, rejected,
            dict(content_type="application/zip", body_text="not-a-zip"),
        )
        return accepted, rejected

    def task(self, accepted):
        request = dict(submission_id=accepted["id"], scope="operators", profile="smoke", operators=["Exp"])
        task = self.service.create_task(request, key="frontend-task")
        prefix = f'/api/v1/evaluations/{task["id"]}'
        self.record(
            "task-queued-worker-disabled", "POST", "/api/v1/evaluations", 202, task,
            dict(content_type="application/json", body=request, headers={"Idempotency-Key": "frontend-task"}),
        )
        if self.service.create_task(request, key="frontend-task")["id"] != task["id"]:
            raise ValueError("Regression acceptance condition failed")
        self.record("task-list", "GET", "/api/v1/evaluations?limit=30", 200, self.service.list_tasks())
        self.record("results-score-pending", "GET", prefix + "/results", 200, self.service.results(task["id"]))
        self.record(
            "artifacts-not-yet-available", "GET", prefix + "/artifacts", 200,
            dict(items=self.service.artifacts(task["id"])),
        )
        self.record("task-cancelled-before-execution", "POST", prefix + "/cancel", 200, self.service.cancel(task["id"]))
        self.record(
            "events-queued-and-cancelled", "GET", prefix + "/events?after=0", 200, self.service.events(task["id"])
        )
        return task

    def errors(self, rejected, task):
        operations = [
            ("idempotency-conflict", "/api/v1/submissions",
             lambda: self.service.upload(b"changed-content", key="frontend-upload")),
            ("rejected-submission-cannot-run", "/api/v1/evaluations",
             lambda: self.service.create_task(dict(submission_id=rejected["id"]))),
            ("recovery-not-applicable", f'/api/v1/evaluations/{task["id"]}/recover',
             lambda: self.service.recover_task(task["id"], dict(acknowledge_recovery=True))),
        ]
        for name, path, operation in operations:
            try:
                operation()
            except ServiceError as exc:
                self.record(name, "POST", path, exc.http_status, dict(error=dict(code=exc.code, detail=exc.detail)))
            else:
                raise AssertionError(f"{name}: expected ServiceError")


def examples(corpus: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="te-frontend-contract-") as directory:
        service = EvaluationService(Path(directory), corpus)
        scenarios = Scenarios(service)
        try:
            accepted, rejected = scenarios.submissions()
            task = scenarios.task(accepted)
            scenarios.errors(rejected, task)
        finally:
            service.close()
        return dict(
            source="EvaluationService without Worker; IDs and times normalized",
            npu_execution=False,
            successful_evaluation_fixture=False,
            scenarios=_normalize_examples(scenarios.items, directory),
        )


def _normalize_examples(items, directory):
    ids = {}

    def normalize(value, key=""):
        if isinstance(value, dict):
            return {name: normalize(child, name) for name, child in value.items()}
        if isinstance(value, list):
            return [normalize(child, key) for child in value]
        if isinstance(value, str):
            if key in {"created_at", "updated_at", "time"}:
                return "2026-11-13T00:00:00+00:00"
            value = value.replace(directory, "<local>")

            def identifier(match):
                original = match.group(0)
                if original not in ids:
                    ids[original] = f"{len(ids) + 1:032x}"
                return ids[original]

            return re.sub(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", identifier, value)
        return value

    return normalize(items)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on missing/stale exports; do not write")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    exports = {
        "openapi.json": (json.dumps(openapi(), ensure_ascii=False, indent=2) + "\n").encode(),
        "examples.json": (json.dumps(examples(args.corpus), ensure_ascii=False, indent=2) + "\n").encode(),
        "submission-valid.zip": package_bytes(),
    }
    stale = []
    for name, data in exports.items():
        target = OUTPUT / name
        if args.check:
            if not target.is_file() or target.read_bytes() != data:
                stale.append(name)
        else:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if stale:
        emit("Stale frontend contract exports: " + ", ".join(stale))
        return 1
    emit(f'Frontend contract {"checked" if args.check else "exported"}: {len(exports)} files; no execution')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
