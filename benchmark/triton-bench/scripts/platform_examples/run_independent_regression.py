#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Submit new implementations to the existing platform; audit its frozen-case results."""

from __future__ import annotations


import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from triton_eval.console import emit
from triton_eval.corpusmod import load_manifest, verify_corpus_files
from triton_eval.orchestrator import CannCloud, RemoteConfig

TERMINAL = {"COMPLETED", "REJECTED", "FAILED", "CANCELLED", "INTERRUPTED"}


def log_event(**value):
    emit(json.dumps(value, ensure_ascii=False), flush=True)


@dataclass
class Observation:
    args: argparse.Namespace
    out: Path
    expected: dict
    package: bytes
    package_hash: str
    cloud: object
    result: dict
    began: float
    task: dict | None = None

    def request(self, path, value=None, *, raw=False, key=None):
        data = value if raw else json.dumps(value).encode() if value is not None else None
        headers = {"Content-Type": "application/zip" if raw else "application/json"}
        if key:
            headers["Idempotency-Key"] = key
        request = urllib.request.Request(self.args.url + path, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=35) as response:
            return json.load(response)

    def prepare_task(self):
        self.result["initial_cloud_state"] = self.cloud.state()
        if self.result["initial_cloud_state"] != "stopped" and not self.args.resume_task:
            raise RuntimeError("instance is busy; preserve the existing owner")
        catalog = self.request("/api/v1/catalog")
        if catalog["manifest_hash"] != self.expected["manifest_hash"]:
            raise ValueError("Platform corpus identity differs")
        if not catalog["worker_enabled"] or catalog["execution_backend"] != "remote" or catalog["worker_paused"]:
            raise ValueError("Remote Worker is not available")
        submission = self._resume() if self.args.resume_task else self._submit()
        self.result.update(submission=submission, task=self.task)
        (self.out / "task.json").write_text(json.dumps(self.task, indent=2) + "\n")
        if set(self.task["case_codes"]) != set(self.expected["case_codes"]):
            raise ValueError("Task cases differ from frozen expectation")
        log_event(kind="submitted", task_id=self.task["id"], planned_cases=self.task["planned_cases"],
                  url=self.args.url + "/workspace/jobs/" + self.task["id"])

    def wait_terminal(self):
        previous = None
        while time.monotonic() - self.began < 3900:
            self.task = self.request("/api/v1/evaluations/" + self.task["id"])
            state = (self.task["status"], self.task.get("remote", {}).get("phase"), self.task["completed_cases"] // 10)
            if state != previous:
                log_event(kind="progress", status=self.task["status"], remote=self.task.get("remote"),
                          completed=self.task["completed_cases"])
                previous = state
            if self.task["status"] in TERMINAL:
                return
            time.sleep(2)
        raise TimeoutError("bounded regression exceeded its time budget")

    def collect(self):
        base = "/api/v1/evaluations/" + self.task["id"]
        results = self.request(base + "/results")
        self.result.update(task=self.task, results=results, events=self._events(base))
        hashes = self._artifacts(base)
        self.result["artifact_hashes"] = hashes
        rows = results["cases"]
        issues = self.result["issues"]
        codes = [row["case_code"] for row in rows]
        if len(codes) != len(set(codes)) or set(codes) != set(self.expected["case_codes"]):
            issues.append("case identity differs from the frozen expectation")
        for row in rows:
            _audit_samples(row, issues)
        self.result["per_operator"] = _operator_summary(rows)
        self.result["status_counts"] = dict(Counter(row["status"] for row in rows))
        self.result["success"] = _accepted(self.task, results, self.result, self.expected, hashes)

    def cleanup(self):
        try:
            self._cancel_pending()
        except Exception as error:
            self.result["cleanup_error"] = type(error).__name__
        try:
            self.result["final_cloud_state"] = self.cloud.state()
        except Exception as error:
            self.result["final_cloud_state"] = type(error).__name__
        self.result["success"] = self.result["success"] and self.result["final_cloud_state"] == "stopped"
        self.result["elapsed_s"] = round(time.monotonic() - self.began, 2)

    def _resume(self):
        candidate = self.request("/api/v1/evaluations/" + self.args.resume_task)
        journal = json.loads((self.out / "task.json").read_text())
        if candidate["id"] != journal["id"] or candidate["package_sha256"] != self.package_hash:
            raise ValueError("Resume task or submission identity differs")
        if candidate["manifest_hash"] != self.expected["manifest_hash"] or candidate["profile"] != "standard":
            raise ValueError("Resume corpus or profile differs")
        self.task = candidate
        self.result["resumed_observer"] = True
        return self.request("/api/v1/submissions/" + candidate["submission_id"])

    def _submit(self):
        upload_key = "independent-v19-upload-" + self.package_hash
        submission = self.request("/api/v1/submissions", self.package, raw=True, key=upload_key)
        if submission["status"] != "READY" or submission["package_sha256"] != self.package_hash:
            raise ValueError("Submission failed admission or package identity differs")
        repeated = self.request("/api/v1/submissions", self.package, raw=True, key=upload_key)
        if repeated["id"] != submission["id"]:
            raise ValueError("Upload idempotency failed")
        payload = dict(submission_id=submission["id"], profile="standard", scope="operators",
                       operators=self.expected["operators"])
        key = "independent-v19-task-" + self.package_hash + "-" + self.args.revision
        self.task = self.request("/api/v1/evaluations", payload, key=key)
        if self.request("/api/v1/evaluations", payload, key=key)["id"] != self.task["id"]:
            raise ValueError("Task idempotency failed")
        return submission

    def _events(self, base):
        events, cursor = [], 0
        while True:
            page = self.request(base + "/events?after=" + str(cursor))
            events.extend(page["items"])
            if not page["items"] or page["next_cursor"] == cursor:
                return events
            cursor = page["next_cursor"]

    def _artifacts(self, base):
        directory = self.out / "artifacts"
        directory.mkdir(exist_ok=bool(self.args.resume_task))
        hashes = {}
        for item in self.request(base + "/artifacts")["items"]:
            if Path(item["name"]).name != item["name"]:
                raise ValueError("Unsafe artifact name")
            with urllib.request.urlopen(self.args.url + base + "/artifacts/" + item["name"], timeout=35) as response:
                data = response.read()
            (directory / item["name"]).write_bytes(data)
            hashes[item["name"]] = len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
        return hashes

    def _cancel_pending(self):
        if not self.task or self.task["status"] in TERMINAL:
            return
        self.request("/api/v1/evaluations/" + self.task["id"] + "/cancel", {})
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            self.task = self.request("/api/v1/evaluations/" + self.task["id"])
            released = self.task.get("remote", {}).get("release_state") == "SHUTDOWN"
            if self.task["status"] in TERMINAL and released:
                break
            time.sleep(3)
        self.result["task"] = self.task


def _audit_samples(row, issues):
    if row["status"] != "PASS":
        return
    code = row["case_code"]
    values = row.get("samples_ms") or []
    if len(values) != 3 or not all(math.isfinite(value) and value > 0 for value in values):
        issues.append(code + ": missing or invalid standard samples")
        return
    if not math.isclose(statistics.median(values), row["perf_median_ms"], rel_tol=1e-8):
        issues.append(code + ": incorrect median")
    cv = statistics.pstdev(values) / statistics.mean(values)
    if not math.isclose(cv, row.get("perf_cv") or 0, rel_tol=1e-8, abs_tol=1e-12):
        issues.append(code + ": incorrect CV")
    baseline = row.get("baseline_median_ms")
    if not baseline or not math.isclose(baseline / row["perf_median_ms"], row.get("speedup") or 0, rel_tol=1e-8):
        issues.append(code + ": incorrect speedup")


def _operator_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["operator"]].append(row)
    summary = {}
    for name, group in groups.items():
        complete = all(row.get("speedup") and row["status"] == "PASS" for row in group)
        geo = math.exp(statistics.mean(math.log(row["speedup"]) for row in group)) if complete else None
        summary[name] = dict(
            status_counts=dict(Counter(row["status"] for row in group)), geometric_speedup=geo,
            unstable_cases=[row["case_code"] for row in group if (row.get("perf_cv") or 0) > 0.05],
        )
    return summary


def _accepted(task, results, result, expected, hashes):
    checks = (
        task["status"] == "COMPLETED", result["status_counts"] == expected["expected_status_counts"],
        not result["issues"], task["remote"]["release_state"] == "SHUTDOWN", task["remote"]["artifacts_verified"],
        bool(hashes), all(hashes.values()), results.get("performance_available"), bool(results.get("score")),
    )
    if not all(checks):
        return False
    score = results["score"]
    return score["coverage_ops_core"] == 100 and score["coverage_strict"] == 100


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--resume-task", help="observe an existing task without submitting or rerunning it")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:5177")
    parser.add_argument(
        "--remote-config", type=Path, default=ROOT / "runs/evaluation-access-20261114/remote-worker.json"
    )
    return parser.parse_args()


def _observation(args):
    out = args.prepared.parent / args.revision
    out.mkdir(exist_ok=bool(args.resume_task))
    expected = json.loads((args.prepared / "expectations.json").read_text())
    corpus = ROOT / expected["corpus"]
    manifest = load_manifest(corpus / "manifest.json")
    verify_corpus_files(corpus, manifest)
    if manifest.manifest_hash != expected["manifest_hash"]:
        raise ValueError("Frozen corpus identity differs")
    package = args.submission.read_bytes()
    package_hash = hashlib.sha256(package).hexdigest()
    cloud = CannCloud(RemoteConfig.load(args.remote_config))
    result = dict(success=False, manifest_hash=manifest.manifest_hash, package_sha256=package_hash,
                  revision=args.revision, expected_cases=expected["case_codes"], issues=[])
    return Observation(args, out, expected, package, package_hash, cloud, result, time.monotonic())


def _keep_awake(result):
    executable = shutil.which("caffeinate") if sys.platform == "darwin" else None
    if executable is None:
        return None
    process = subprocess.Popen(
        [str(Path(executable).absolute()), "-i", "-w", str(os.getpid())],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    result["idle_sleep_prevention"] = "process-scoped caffeinate -i"
    return process


def main():
    observation = _observation(_arguments())
    awake = _keep_awake(observation.result)
    try:
        observation.prepare_task()
        observation.wait_terminal()
        observation.collect()
    except Exception as error:
        observation.result["failure"] = dict(type=type(error).__name__, detail=str(error))
        log_event(kind="failure", **observation.result["failure"])
    finally:
        observation.cleanup()
        if awake:
            awake.terminate()
            awake.wait(timeout=5)
        result = observation.result
        (observation.out / "observation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        log_event(kind="finished", success=result["success"], counts=result.get("status_counts"),
                  issues=result["issues"], final_cloud_state=result["final_cloud_state"])
    return 0 if observation.result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
