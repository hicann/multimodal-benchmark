/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { accessSync, constants } from "node:fs";
import { dirname, resolve, isAbsolute, delimiter } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const requested = process.env.TRITON_EVAL_PYTHON ||
  (process.env.VIRTUAL_ENV ? resolve(process.env.VIRTUAL_ENV, "bin/python") : "python3");
const candidates = isAbsolute(requested) ? [requested] :
  (process.env.PATH || "").split(delimiter).map((directory) => resolve(directory, requested));
const executable = candidates.find((candidate) => {
  try { accessSync(candidate, constants.X_OK); return true; }
  catch { return false; }
});
if (!executable) throw new Error("需要 Python 3.11+ 评测环境；可用 TRITON_EVAL_PYTHON 指定解释器。");
const result = spawnSync(executable, [resolve(root, "scripts/platform_examples/build_platform_examples.py")], {
  cwd: root,
  stdio: "inherit",
  env: { ...process.env, PYTHONPATH: [resolve(root, "src"), process.env.PYTHONPATH].filter(Boolean).join(delimiter) },
});
if (result.error) throw result.error;
if (result.status !== 0) process.exitCode = result.status || 1;
