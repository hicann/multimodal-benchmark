/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { DownloadSimple } from "@phosphor-icons/react";

export function ExamplePackages() {
  return (
    <section className="example-packages" aria-label="下载多算子评测包">
      <h3>下载评测包，直接开始测试</h3>
      <div className="example-package">
        <strong>37 算子 · 746 用例（推荐）</strong>
        <p>
          可直接上传运行。本次生成的包已做静态预检，真机成绩以实际评测结果为准。
        </p>
        <a className="button" href="/examples/v19-half-37.zip" download>
          <DownloadSimple size={18} /> 下载 37 算子 ZIP
        </a>
      </div>
      <details className="technical-details">
        <summary>需要全部算子，或查看最小文件结构</summary>
        <p>
          完整包支持选择 74 个算子、1486
          个用例。上传后可只选需要的算子；全部运行会更久，目前尚未完成整包真机验证。
        </p>
        <a
          className="button secondary"
          href="/examples/v19-full-74.zip"
          download
        >
          <DownloadSimple size={18} /> 下载 74 算子 ZIP
        </a>
        <p>
          <a className="text-link" href="/examples/abs-smoke.zip" download>
            Abs 最小结构示例（仅 1 个算子）
          </a>
        </p>
      </details>
    </section>
  );
}
