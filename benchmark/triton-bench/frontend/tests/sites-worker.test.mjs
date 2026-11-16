/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import worker from "../worker/index.js";

function assetService(statusForPath) {
  const paths = [];
  return {
    paths,
    environment: {
      ASSETS: {
        async fetch(request) {
          const url = new URL(request.url);
          paths.push(url.pathname + url.search);
          const status = statusForPath(url.pathname);
          return new Response(`asset:${url.pathname}`, { status });
        },
      },
    },
  };
}

const browserCases = [
  { path: "/assets/app.js", method: "GET", accept: "*/*", status: 200, attempts: ["/assets/app.js"] },
  { path: "/workspace/jobs", method: "GET", accept: "text/html", status: 200,
    attempts: ["/workspace/jobs", "/index.html"] },
  { path: "/submit?source=review", method: "GET", accept: "text/html", status: 200,
    attempts: ["/submit?source=review", "/index.html"] },
  { path: "/api/v1/catalog", method: "GET", accept: "text/html", status: 404,
    attempts: ["/api/v1/catalog"] },
  { path: "/api/missing", method: "GET", accept: "application/json", status: 404, attempts: ["/api/missing"] },
  { path: "/submit", method: "POST", accept: "text/html", status: 404, attempts: ["/submit"] },
];

for (const scenario of browserCases) {
  test(`hosting preserves ${scenario.method} ${scenario.path} behavior`, async () => {
    const assets = assetService((path) => ["/assets/app.js", "/index.html"].includes(path) ? 200 : 404);
    const request = new Request(new URL(scenario.path, "https://triton-eval.test"), {
      method: scenario.method, headers: { accept: scenario.accept },
    });
    const result = await worker.fetch(request, assets.environment);
    assert.equal(result.status, scenario.status);
    assert.deepEqual(assets.paths, scenario.attempts);
  });
}

test("hosting output contains the current Worker and project configuration", async () => {
  const root = new URL("../", import.meta.url);
  const html = await readFile(new URL("dist/client/index.html", root), "utf8");
  assert.match(html, /<html/);
  for (const [source, deployed] of [
    ["worker/index.js", "dist/server/index.js"],
    [".openai/hosting.json", "dist/.openai/hosting.json"],
  ]) {
    const expected = await readFile(new URL(source, root));
    const actual = await readFile(new URL(deployed, root));
    assert.deepEqual(actual, expected, deployed);
  }
});
