# Proposal

## Why

MVP 已完成并归档，验证了「规范—生成—运行」三层链路能跑通。但它有两个根本局限：

1. **算子单一**：只有 5 个手写算子（add / mul / sum / softmax / layer_norm），
   全部属 Elementwise 与 Reduction，无法反映编译器在归约、归一化、
   矩阵运算、融合等方向上的真实能力差异。
2. **评测对象错位**：当前 case 绑定 kernel（`kernel_path` + `entry`），
   等于**评测方替被测方写实现**。这使平台无法服务真正的外部提交者——
   用户无法上传自己的算子。

参考 `reference/cann-bench/`（已就位）的做法，本变更引入**算子分类分级体系**
与**用户提交契约**，使平台从「跑一批固定 kernel」升级为
「接收外部算子实现并鲁棒地评测、出分、报错」。

## What Changes

- 新增算子分类分级：13 个 `category`（语义族）为主维度，`level`（L1–L4 难度）为算子属性
- 新增算子规格 schema（`proto.yaml` 简化版）：声明支持的 dtype / shape 约束 / attrs
- **改变 case 契约**：case 只描述「输入数据 + 期望行为」，**不再绑定 kernel**
  —— kernel 由提交方提供，这是支持用户上传的前提
- 新增提交契约：目录式（非 wheel），零构建
- 新增提交校验：分三阶段（静态 AST → 加载 → 执行），规则编号化，失败即明确报错
- 新增反作弊行为规则（SUB-BEH-*），**利用 category 做差异化拦截**，
  把 cann-bench 需人工审查的项自动化
- 新增按 category / level 的分组评分与报告

**明确不做（本变更范围外）**：

- 不支持 AscendC / PyPTO / TileLang，只支持 triton-ascend 提交
- 不引入跨赛道合成总分（遵守规范 §3.3，与 cann-bench 的三元加权不同）
- 不做 Web 提交入口与榜单站（CLI 优先）
- 不做人工审查流程（本变更目标是尽可能把审查项自动化）
- VVFusion 分类下无可行的 triton 算子，留占位不生成 case

## Capabilities

### New Capabilities

- `operator-taxonomy`：算子的分类分级体系——13 个 category 的定义与归属、
  level 属性、算子元数据（来源算子名、参考实现、可行性标注）。
- `operator-spec`：算子规格定义——dtype / shape 约束 / attrs / 输入输出 schema，
  作为 case 生成与提交校验的共同依据。
- `submission-contract`：用户提交的目录契约与发现机制——目录结构、
  manifest 声明、入口函数签名约定、多算子批量提交。
- `submission-validation`：提交的三阶段校验（静态 / 加载 / 执行），
  规则编号化，每条规则有明确失败结果与合法/非法样例测试。
- `anti-cheat`：反作弊行为规则与拦截——静态 AST 检测、按 category 差异化白名单、
  运行时兜底 guard，防止用 torch 内置 API 代算。

### Modified Capabilities

- `case-execution`：case 不再自带 kernel，改为由提交方提供实现；
  执行器需按新契约加载入口函数。终态状态机扩展以覆盖提交类失败。
- `corpus-freeze`：case 来源从「绑定 kernel 的单一 corpus」
  变为「按 category 分层的算子规格 + 派生 case」。
- `scoring`：新增按 category / level 的分组聚合与披露，仍不出跨赛道总分。

## Impact

- **代码**：新建 `src/triton_eval/{taxonomy,spec,submission,validator,anticheat}/`；
  改造 `harness/npu_exec.py` 的执行入口；扩展 `scorer.py` 与 `report.py`。
- **契约变动（破坏性）**：`CaseSpec.kernel_path` / `entry` 不再指向评测方自己的 kernel，
  而由提交提供。既有 20 个 case 需迁移到新契约（算子规格 + 提交实现）。
- **依赖**：新增无。仍禁止 `triton` 进入评测工具链依赖。
- **参考资产**：`reference/cann-bench/`（16MB）入仓作为规范参照，不参与运行。
- **数据**：算子规格与 case 定义入 Git（纯文本）；`runs/` 仍不入库。
