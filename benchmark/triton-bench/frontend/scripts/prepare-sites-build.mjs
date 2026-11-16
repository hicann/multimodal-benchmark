#!/usr/bin/env node
/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { copyFile, mkdir, stat } from "node:fs/promises";

// Package the CANN evaluation UI. API execution remains in frontend_gateway;
// this Worker only serves the static prototype and its browser routes.
const project = new URL("../", import.meta.url);
const deploymentFiles = new Map([
  ["worker/index.js", "dist/server/index.js"],
  [".openai/hosting.json", "dist/.openai/hosting.json"],
]);

async function requireFile(relative) {
  const entry = await stat(new URL(relative, project));
  if (!entry.isFile()) throw new Error(`Expected a build file: ${relative}`);
}

export async function prepareDeployment() {
  await requireFile("dist/client/index.html");
  for (const input of deploymentFiles.keys()) await requireFile(input);
  for (const [input, output] of deploymentFiles) {
    const target = new URL(output, project);
    await mkdir(new URL("./", target), { recursive: true });
    await copyFile(new URL(input, project), target);
  }
  return [...deploymentFiles.values()];
}

const packaged = await prepareDeployment();
console.log(`Prepared Triton Eval hosting files: ${packaged.join(", ")}`);
