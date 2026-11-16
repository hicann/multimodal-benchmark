<!--
Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Triton Bench

此目录是从 triton-eval 工作区迁入的完整前后端源码及冻结评测资产。Python 包名仍为 `triton_eval`；目录名为 `benchmark/triton-bench`。

- 后端、CLI 与远端 Worker：`src/triton_eval/`
- 前端：`frontend/`；启动与构建见 [前端 README](frontend/README.md)
- 远端平台启动及外部凭据要求见 [远端 Worker](docs/remote-worker.md)
- 需求覆盖及已知限制见 [覆盖报告](docs/requirements-coverage-report.md)
- ba04485 修复保留、增量验证及官方复扫边界见 [质量复核](docs/quality/codecheck-recovery-20261116.md)

在本目录安装 Python 3.11+ 环境与依赖：`python -m pip install -e '.[dev]'`。远端模式必须提供仓库外、权限 600 的有效 CANNLab OAuth 文件；仓库仅保存示例路径，不附带凭据。云端登录、连接脚本及插件环境由运营者独立配置。

迁移保留冻结用例与 manifest；框架修订通过可追溯的资产索引迁移记录认证，不复制 `.git`、运行记录、历史归档、依赖缓存、本机凭据或本机导出表格。历史实验文档中的 `runs/` 链接用于说明原工作区证据位置，对应运行产物未随源码发布。本地语料构建和预检不代表已完成新机器的真机验收。


开发检查：在本目录运行 `pytest tests/test_reviewed_baseline.py`，对原基线文件、许可证头和单文件变更记录进行校验。

提交钩子同时扫描真实暂存区凭据并检查已修复基线。独立仓库在根目录执行 `pre-commit install --config .pre-commit-config.yaml`；主仓库从 Git 根目录执行 `pre-commit install --config benchmark/triton-bench/.pre-commit-config.yaml`。钩子失败不能使用跳过参数绕过，应核对具体差异和记录。
