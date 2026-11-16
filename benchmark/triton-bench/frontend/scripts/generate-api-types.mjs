/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import openapiTS, { astToString, COMMENT_HEADER } from "openapi-typescript";

const input = new URL("../../docs/frontend-integration/openapi.json", import.meta.url);
const output = new URL("../src/api/schema.d.ts", import.meta.url);
const source = readFileSync(fileURLToPath(import.meta.url), "utf8");
const license = source.slice(0, source.indexOf("import {"));
const generated = license + COMMENT_HEADER + astToString(await openapiTS(input));
if (process.argv.includes("--check")) {
  if (readFileSync(output, "utf8") !== generated) {
    throw new Error("API 类型与当前接口不一致；请运行 npm run api:types");
  }
} else {
  writeFileSync(output, generated);
}
