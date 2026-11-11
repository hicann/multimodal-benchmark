# Proposal

## Why

现有 Triton 编译器评测依赖分散的内部测试用例与单点性能测试，缺少统一口径、可复现规则与标准化流程。本项目已完成体系设计（`docs/Triton编译器评测体系技术报告.md`），但至今没有一条可执行的最小链路来验证这套设计是否成立。

本变更以「打通评测流程」为唯一目标，交付一个端到端可运行的 MVP：从冻结测试集、执行判定、评分到生成报告，一条命令跑通。规模与复杂度一律让位于「链路是否成立」，为后续扩规模与建平台提供可验证的地基。

## What Changes

- 新增执行环境固化机制：`env.sh` + 独立的被测编译器 venv + 环境指纹采集 + run 前健康探针
- 新增冻结测试集（corpus）：以 triton-ascend 自带 `tutorials/` 与 `unittest/pytest_ut/` 算子为 seed，20–30 个 case，全部打 `vendor_selftest` 标签
- 新增单 case 执行与判定：8 个终态状态机、单 oracle（torch eager）+ 第二参考实现交叉校验、按 dtype 的容差表
- 新增性能测量：`do_bench` 封装、同会话 torch eager 基线（B1）、CV 稳定性门禁
- 新增评分：`OpScore` / `OpStrictPass` 为功能主得分，`P_geo` 与 `fast_p` 为性能口径，首轮基线冻结为 100 分锚点
- 新增结果存储：7 张表的 SQLite 库，`results.jsonl` 为真相源、数据库可重建
- 新增 CLI 与本地报告：`run` / `score` / `report` / `import` 子命令，输出 Markdown + HTML

**明确不做（本变更范围外）**：

- 不做 Web 平台与榜单站（MVP 仅 CLI + 本地报告）
- 不做 1000+ case 的生成层（抽取器、libcst 变异、覆盖引导引擎、四道门禁）
- 不做 FA 反投影诊断（FA 标签只存不查）
- 不做跨编译器对照基线 B0（昇腾上不存在上游 Triton，`Performance` 无法按报告 §6.3 归一化）
- 不做 `UNSUPPORTED` / capability manifest 机制（MVP 全部 case 计入分母，防止以「声明不支持」剔除难题）
- **不出跨赛道 `Total` 分**（报告 §3.3 禁止两赛道隐式换算，与 §6.4 的加权合成冲突，本变更采用 §3.3）

## Capabilities

### New Capabilities

- `env-baseline`：执行环境的固化、采集与校验——CANN 环境变量注入、被测编译器 venv 隔离、环境指纹 `env_fp` 计算、run 前健康探针。换硬件或换编译器版本时由此能力承载。
- `corpus-freeze`：测试集的定义、冻结与完整性校验——case schema、算子与权重、`manifest.json`、内容哈希与冻结后防篡改校验。
- `case-execution`：单 case 的加载、编译、执行与结果判定——8 个终态状态机、oracle 比对、容差判定、失败归因线索留存。
- `performance-measurement`：性能计时与稳定性门禁——`do_bench` 封装、同会话基线交错测量、原始样本留存、CV 超标判定。
- `scoring`：run 级分数的计算与重算——`OpScore` / `OpStrictPass` / `P_geo` / `perf_score` / `fast_p`，以及首轮基线锚点。
- `result-storage`：结果数据的落库、查询与迁移——7 张表的 SQLite 结构、`results.jsonl` 真相源约定、从 jsonl 重建数据库。
- `reporting`：评测报告的生成——按模板渲染 Markdown 与 HTML，披露主得分、辅助指标、静默错误清单与环境指纹。

### Modified Capabilities

无。本项目此前不存在任何 spec，全部为新建。

## Impact

- **代码**：新建 `src/triton_eval/`（L0/L1 分层）、`spec/`、`corpus/`、`env/`；`openspec/changes/build-mvp-eval-pipeline/` 下新增本 change。
- **依赖**：拆成两份——`env/requirements-eval.txt`（pydantic / pyyaml / jinja2 / pytest，跨平台，macOS 与 Linux 通用）与 `env/requirements-compiler.txt`（triton-ascend 等，仅 NPU 主机）。**前者禁止出现社区 `triton`**，否则会覆盖 triton-ascend 的同名目录并静默破坏被测对象。
- **系统**：执行点固定为 CANNLab 昇腾主机（`Ascend910` 64GB ×2，**共享非独占**、CANN 9.0.0、`triton-ascend` 3.2.1）。该环境无锁频权限，产出的性能数据一律标注不可上榜。开发可在 macOS 上进行（L0 层全部可跑）。
- **数据**：`runs/` 不进 Git，需定期导出；这是唯一不可重建的资产（代码、corpus、环境、数据库均可重建）。
- **文档**：技术报告 §3.3 与 §6.4 存在冲突（前者禁止跨赛道换算，后者定义 `Total` 加权合成），本变更采用 §3.3，技术报告需相应修订。
