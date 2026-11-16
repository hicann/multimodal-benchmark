# 2026-11-09 Code Check 修复与回归

本轮对应 `CANN_multimodal-benchmark_全量导出（未解决）_20261109124716.xlsx` 的 28 条记录。

## 四类修复

- **OAT.1（10 条）**：根目录 `OAT.xml` 的 `binaryFileTypePolicyFilter` 仅登记报告中的 8 个 YAML 和 2 个 Jinja 模板的精确仓库路径，每项说明用途。UTF-8 解码、无 NUL、YAML/Jinja 解析检查保留在回归脚本中。文件内容与扩展名保持原样。这是对已确认文本文件的 OAT 策略配置，是否消除线上告警必须重新扫描确认。
- **G.NAM.04（6 条）**：四个算子的 Python 入口张量参数统一为 `input_value`，同步注册表的签名、输入名称和参数列表，修改六处实现入口及 Cummin 故障场景生成器。参数名、数量和位置参数限制仍严格校验；旧的 `input` 参数名会被拒绝。外部自有提交如实现这四个算子，需要同步改名；张量位置传参和算子属性的调用方式保持一致。
- **G.ERR.11（11 条）**：不可信边界捕获 `BaseException` 后，原样传播普通 `Exception` 和操作者 `KeyboardInterrupt`，将其他控制流异常转为带原始原因链的 `SubmissionEscape`。真实退出攻击源码作为 `import_exit_kernel.py.txt` 测试输入，经过原有带哈希认证的样例装配机制生成独立提交中的 `kernel.py`，保持原始 `raise SystemExit(0)` 攻击语义；测试实际导入和实际子进程退出，不以普通异常代替攻击。CLI 拒绝场景通过子进程断言退出码 2。Worker 恢复测试在 RUNNING、设备执行之前终止真实子进程，检查重启后的 INTERRUPTED、QUEUED、暂停标记及包哈希。
- **G.TYP.07（1 条）**：张量子类测试改为明确的 child/grandchild 分支，保留四种类型与两种嵌套形式的八组验证。

14 套现有冻结资产通过仓库已有 `scripts/codecheck_revision.py` 发布显式修订 `codecheck-20261109-contract-and-control-flow`。旧索引与迁移元数据逐字节归档，新索引保留完整校验。用例 manifest、参考源码、容差及评分逻辑不变。迁移元数据中的硬件验证仍为 `NOT_RUN`，不得将本机回归视为真机认证。

回归还发现并修复了两处已有的旧参考目录读取问题：官方算子冻结仅复制所选算子经过认证的参考源码，模式版本构建使用现有 `copy_reference_tree` 重建参考目录。未修改原仓库上的两项对应测试同样失败；修复保留参考字节与独立 CPU 门禁。

## 2026-11-10 默认改为失败项定向复测

根据用户缩短真机运行时间的要求，`scripts/codecheck-regression-verify.sh` 无参数时默认 `failed-last`。
仅复测基线 `codecheck-20261110-073225-597670` 未通过的 6 个 normal 完整场景：
`OK-EXP`、`OK-ADD`、`OK-GROUPED`、`OK-MULTI`、`MIXED-RUNTIME`、`EXEC-PARTIAL`。
跳过上一轮已通过的 20 个 independent 与 44 个 normal 场景，不减少所选场景内的任何用例，
不修改预期、评分或反例；保留 ZIP 上传到报告下载的完整 HTTP/NPU 链路。
轻量静态身份检查仍保留，不运行 CPU 全量回归或新增探针。

定向结果标记 `scope=SUBMISSION_E2E_FAILED_LAST`，`selection` 记录基线、6 个场景名和
`full_regression=false`。验收需 `simulation-normal=PASS`、总退出码 0，normal 观察报告
`selected_scenario_count=matched=6`、`mismatched=0`、`all_selected_observed=true`，
且场景名与清单完全一致、没有重复。定向通过不代表当前提交全量回归通过。
该清单明确绑定上述基线，不会根据后续远端结果自动变化。

Mac 上传、真机运行、Mac 拉取仍统一在开发工作区的 `.local-only/codecheck-workflow.md`，
不按轮次创建命令文件。显式 `full` 可以恢复下文的原有 70 场景范围。

## 原有全量真机入口（需要时显式选择）

在已有环境的隔离 NPU 真机上，于 `benchmark/triton_bench` 执行：

```bash
bash scripts/codecheck-regression-verify.sh full
```

入口沿用 `scripts/submission-simulation-verify.sh` 的环境自举、真实 HTTP 提交和结果发布流程。轻量校验十项 OAT 文本、样例装配哈希与冻结资产后，只运行以下两个端到端阶段：

1. `simulation-independent`：20 个独立合法提交，验证 ZIP 上传、解析、任务创建、真实 NPU 评测、评分、报告生成和下载，并核对幂等性及任务状态链。
2. `simulation-normal`：50 个既有正常与异常提交场景，检查合法算子运行、非法内容拒收、加载错误、编译错误、执行错误、错误输出、多输出及部分通过时的计分。其中 Cummin、ArgMax、SwiGlu、CrossEntropyLoss 均通过实际 ZIP 提交执行冻结用例，覆盖本轮公开参数迁移涉及的四类算子。

默认真机入口不执行全量 CPU pytest、单独健康探针或 54 项直接接口数值测试。目标是验证用户提交 ZIP 到评分与结果下载的完整链路。沿用联调默认的 `smoke` 档，核对功能得分和严格通过率；不作性能验收或所有实现均受支持的声明。

任一阶段失败即停止后续阶段，后续保持 `NOT_RUN`。异常/超时归档为失败；超时或中断留下暂停标记。一个运行目录同时归档两档结果，退出时自动尝试上传 `npu-results`，成功上传不代表测试通过。目录：`runs/codecheck-regression/records/codecheck-<UTC时间>`。上传失败只执行打印的补传命令，保留原始记录，不重新跑评测。

控制台实时显示子进程输出，同时保存分阶段日志；阶段运行中每 15 秒输出等待进度。场景日志显示 ZIP 解析状态、任务 ID、任务状态变化、等待时长、评分及下载产物数。`execution.json` 在开始阶段时即记录 `RUNNING`，完成后保存终态和退出码；`environment.json` 记录解释器、包版本及设备选择；`static-checks.json` 记录文本和源码哈希、冻结资产身份。

HTTP 证据位于 `independent/`、`normal/` 子目录。验收时核对被测提交一致、总退出码为 0、两个阶段均为 PASS，且两份 `*-observations.json` 的 `all_selected_observed=true`、`mismatched=0`。合法场景须完成评测并返回预期功能得分，异常场景须命中原有预期；不能把预期负例误认为整轮失败。缺失、空白、不完整或含不匹配的观察报告均判失败。报告 Markdown、HTML 和原始结果会经 HTTP 下载并校验哈希。

只做本机 CPU 回归（不会上传结果）：

```bash
PYTHONPATH=src python3 scripts/verify_codecheck_regression.py \
  --record /tmp/codecheck-cpu-record-20261109 \
  --work /tmp/codecheck-cpu-work-20261109 --cpu-only
```

记录与工作目录必须为新目录，禁止覆盖旧证据。完整真机模式要求 `--acknowledge-isolation`，封装脚本已经显式提供。

## 发布与扫描

发布时必须同时提交仓库根目录 `OAT.xml` 和 `benchmark/triton_bench` 的修改。执行真机回归后，还需要在 Code Check 服务对同一提交重新扫描，确认报告中的四类告警关闭且没有新增。真机测试和本机静态检查均不能替代服务端质量门禁结论。

本机路径、分支名与上传/运行/拉取三组完整命令存放在开发工作区 `.local-only/codecheck-20261109-workflow.md`，该文件不发布进产品仓库。

## Local validation (2026-11-09)

The complete CPU regression finished with 2101 passed, 5 skipped, 239 deselected and 21 subtests passed in 452.48 seconds. No failures remain. All 14 frozen asset indexes passed validation. The 54 device cases were collected successfully; device execution is NOT_RUN locally.

The five skips cover the inapplicable no-torch branch, a torch_npu guard, the standalone torch-prefix case checked by the precise scanner, and two HTTP tests blocked by this sandbox's loopback policy. The isolated NPU host must execute both HTTP workflows. The 54 direct device tests remain available separately and are not part of the default end-to-end entry.

Four offline shell scenarios verified success, test failure, publication failure and bootstrap failure without losing the test status. Four implementation modules retain the same arithmetic AST after normalizing the reviewed parameter names and wrapper docstrings. All 87 original fixture JIT hash sets and every case manifest remain unchanged.

Local evidence belongs to the isolated candidate checkout. File hashes authenticate the copy applied to the target repository; the temporary checkout commit is not the target's publication commit. Server-side Code Check rescan: NOT_RUN.

本次端到端入口调整的 135 项定向检查已全部通过，覆盖 ZIP 场景装配与接收、HTTP 编排及评分产物、本地发布回归、失败中止、观察证据完整性、实时日志与等待进度。入口和命令模板的 Shell 语法检查、OAT 文本与全部冻结资产检查通过。这些本地结果不属于真实 NPU 执行证据；本轮真机端到端结果仍为 NOT_RUN。


## 17:33:23 导出报告的规范修复

新导出包含 11 条未解决问题：日志 2、assert 2、外部程序路径 2、行宽 1、
圈复杂度 1、嵌套深度 2、函数长度 1。原四类共 28 条在该导出中未出现。
报告部分片段来自旧入口，修复以已具备实时日志、默认仅两条 HTTP 链路的当前版本为基线。

- 入口自身阶段/心跳消息使用 logging；子进程标准输出原样透传，保留增量 UTF-8 解码与日志文件。
- Git 通过 shutil.which 定位并转换为绝对路径，找不到时明确失败，不硬编码主机路径。
- worker 重启辅助进程以显式条件及 AssertionError 代替 assert，在 python -O 下仍检查状态。
- CrossEntropyLoss 的签名仅拆分字符串字面量，拼接结果不变。
- 按进程生命周期、证据校验、环境采集及计划执行拆分入口；超时清理、失败中止、
  RUNNING/终态落盘和暂停标记保留。
- 直接设备测试按算子拆分参考值构造和输出校验，仍收集 54 项；不加入默认真机流程。

默认真机命令与上传、拉取模板保持不变。已知既有算子失败不在本次规范修复范围，
不放宽其验收预期。最终质量门禁仍须对实际发布提交执行 Code Check 服务端复扫。
