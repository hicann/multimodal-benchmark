# 平台化开工复核记录（2026-11-12）

复核对象：`cannbench-platform-reference.md`、`remote-npu-evaluation-architecture.md`，以及当前工作树的评测后端。日期按 Asia/Shanghai。

本次结论：**本机后端与最小远端评测闭环已重新验证，可以进入平台化开发。** 按“cann_retry 启动 → VS Code 插件连接/刷新 → SSH”完成整包同步、资产校验、60 条真实用例、结果回收及本机离线重算，最终实例已确认 SHUTDOWN。无人值守仍需解决插件冷恢复依赖；两张表、SSE 和生产编排器仍待实现，不能把验证脚本当成已建成的平台。

## 1. 本轮实测结果与证据

证据目录：`runs/platform-preflight-20261112/`。本轮仅新增复核文档与验证资产；未调整冻结输入、容差、参考实现或评分公式。

| 验证 | 本轮结果 | 证据文件 |
|---|---|---|
| 全量 L0：排除 npu/torch/triton 标记 | 1375 passed、3 failed、3 skipped、445 deselected，另有 21 subtests passed | `l0-pytest.log`、`l0-pytest.xml` |
| 后端专项，允许真实 loopback socket | **144 passed，无跳过** | `backend-tests.log`、`backend-tests.xml` |
| 真实 HTTP 内容拒收集 | **14/14 MATCH** | `content-observations.json`、`local-http.log` |
| 合法提交 → 创建任务 → 查询 → 排队取消 | READY → QUEUED，计划 20 个 Abs case，取消为 CANCELLED | `local-http.json` |
| v0.18 冻结语料 + 资产复核 | **通过**，74 算子、1385 case | `corpus-check.json` |
| 设计示例 v0.9 + 当前代码复核 | **失败**，68 算子、1360 case，冻结资产与当前源码不一致 | `corpus-check.json` |
| CANNLab 状态与凭据续期 | 成功，初始 SHUTDOWN | `cloud-probe.log` |
| POST start → 运行态 | 各轮 HTTP 200，约 19–45 秒；原始 `status=PROCESSING` 对应 helper 的 running | `attempt-1/cloud-result.json`、`cloud-result.json`、`attempt-3/cloud-result.json` |
| SSH / bash 环境探针 | **最终通过**；首轮冷恢复失败，按用户补充顺序经插件刷新后恢复 | `attempt-1/`、`vscode-ssh-check.json`、`attempt-3/ssh_environment.stdout` |
| finally stop → SHUTDOWN | 各轮均确认关停，最终执行轮约 92 秒；最终 `state=stopped` | `attempt-3/cloud-result.json` |
| 关机后的 SSH | 最终执行轮在运行时 SSH 成功，确认 SHUTDOWN 后同配置命令 22 秒超时；不可依赖停机后补取 | `attempt-3/cloud-result.json` |
| 缓存隧道地址恢复 hub | HTTP 200，但 success=false，WSS 握手返回 400 | `cached-hub-restore.json` |
| 用启停凭据查询插件 connect-urls | **401**，不能混用两套认证链 | `connect-url-check.json` |
| CANNBench 首页、提交文档路由 | HTTPS 200；返回 SPA 外壳，不能据此证明 UI 功能 | `cannbench-public-fetch.json` |
| CANNBench 公开榜单接口 | HTTP 200，返回字段与调研记录相符 | `cannbench-public-contract.json` |
| CANNBench 浏览器登录态交互 | 未完成：浏览器工具启动/创建页面超时 | 本次工具记录 |
| SCP 整包 / 真机依赖 / NPU 正负例 / 结果取回 | **通过**：ZIP/冻结资产校验、实际 import、60 条真机结果、80 个产物哈希 | `attempt-3/remote-summary.json`、`attempt-3/artifact-hash-verification.json` |
| 本机离线结果恢复 → 重算 → 报告 | **通过**：保留身份元数据、清空结果/评分后重新 import 真实 JSONL，20 PASS 与 100% 功能分一致 | `attempt-3/offline-replay/verification.json` |

全量 L0 的两个 HTTP 测试在沙箱下因禁止绑端口跳过，已在后端专项中用真实 socket 补跑通过。另一个跳过是当前环境已有 torch、不适用“无 torch”测试。445 项是显式 deselect，不是通过或本轮真机覆盖。

### 1.1 三项已有基线失败

均位于 `tests/test_codecheck_revision.py`：

- `test_revision_preserves_immediate_predecessor_bytes_and_is_idempotent`
- `test_forged_predecessor_is_rejected_before_writing`
- `test_same_revision_cannot_silently_rebind_changed_source`

`codecheck_revision.revise()` 无条件从目标平台加载 `src/triton_eval/migration_metadata.py`，测试的临时平台没有该文件；当前真实仓库中也没有此文件。它们是资产修订工具的基线缺陷，与 HTTP 编排专项分开归因。发布资产迁移工具前需补齐真实依赖和测试夹具，不能把全量回归标为全绿。

### 1.2 真机闭环与验证边界

首轮使用 cann_retry 启动后直接尝试原文描述的非交互 SSH 冷恢复，隧道地址刷新失败，finally 关停。用户补充应先启动再用 VS Code 连接；随后日志出现 bootstrapRefresh.done，真实 SSH echo 成功。第二轮验证脚本的等待探针只收未发 SSH 客户端 banner，误判超时后按预算关停；这是本次验证工具的缺陷，不是后端或真机故障。修正该探针后，最终执行轮完整通过；证据保存在 `attempt-3/`。当前电脑控制服务不可用，插件界面连接由用户协助完成，不能将本轮称为完全无人值守冷启动。

| 场景 | 真实冻结 case | 实际终态 | 功能评分 | 对照结果 |
|---|---:|---|---|---|
| NEW-ABS | 20 | PASS ×20 | 两项功能分均 100%（专项范围） | MATCH |
| EXEC-WRONG | 20 | WRONG_RESULT ×20 | 两项功能分均 0% | MATCH |
| EXEC-INPUT | 20 | SUBMISSION_INVALID ×20 | null，无有效计分项 | MATCH |

三个任务均完成评测流程；COMPLETED 不代表算法全部正确。合法场景还核对了 QUEUED → RUNNING → SCORING → REPORTING → COMPLETED、同键幂等、实际/计划 case 集合、功能分、报告与结果下载。80 个下载产物在本机重新计算大小与 SHA-256，零差异。最终证据包在 stop 前约 3 秒已取回本机，共 525428 字节。

真机实际复用 `/home/developer/venv/te-compiler/bin/python`：Python 3.11.4、torch 2.7.1+cpu、torch_npu 2.7.1.post4、triton 导入版本 3.2.0，CANN 扩展可导入，torch.npu.is_available() 为真、可见 1 卡；pydantic 2.13.5、PyYAML 6.0.3、Jinja2 3.1.6。未安装或替换编译器。torch 版本中的 +cpu 后缀不能单独用来判断 NPU 不可用，实际 NPU 探针和 kernel 结果才是证据。

通过的范围是 **60 条真实执行记录、3 个提交场景、smoke 档**。不等于全套 1385 case，不等于三项覆盖认证，也不等于跨编译器性能榜单验收。

## 2. 当前后端实际具备什么

| 层 | 已实现能力 | 实际边界 |
|---|---|---|
| 上传与内容检查 | ZIP 原始二进制、SHA-256、路径安全解包、静态规则、manifest 解析、登记算子校验、READY/REJECTED 持久化 | 上传阶段不 import 用户实现；加载/执行结论要等引擎 |
| HTTP | OpenAPI 3.1 的 10 个业务路径，另有 `/openapi.json`；创建提交 201、创建任务 202、结构化错误 | loopback、严格 Host/Origin；不是公开多租户服务 |
| 幂等 | 创建提交与任务均支持 Idempotency-Key；同键异内容 409 | 相同 ZIP、不同幂等键仍可独立提交；不能套用 CANNBench 的全局内容去重语义 |
| 任务范围 | operators/full、smoke/standard；专项使用所选算子的全部冻结 case | 不允许客户端覆盖容差/seed/任意命令；service 默认 v0.18 |
| 持久化 | service.sqlite：objects、events；进程锁、状态事件与任务同事务 | 尚无 remote_job、remote_job_event |
| Worker | 单工作目录单所有者、串行领取、每任务独立 DB/运行/解包目录 | 执行发生在服务所在主机，`_command()` 全是本机路径 |
| 执行与监督 | 父进程静态校验、子进程加载运行、文件事件、硬超时与部分结果恢复 | 本机测试中的设备结果夹具不等于真实 NPU 成绩 |
| 完成校验 | 包身份、case 集合、环境/评测方异常、评分记录和必要产物检查 | case 不全 PASS 也可正常 COMPLETED；COMPLETED 表示流程完成 |
| 评分与报告 | 复用 CLI score/report，按 category/level 输出；JSONL、MD、HTML | smoke 无性能分；独立任务数据库没有跨用户统一性能锚点 |
| 查询与下载 | 真实数据库进度、状态游标、结果、固定产物白名单、SHA-256 | events 为最多 200 条状态事件轮询；没有逐 case SSE |
| 取消与重启 | 排队取消直接 CANCELLED；运行取消终止进程组并暂停；重启将原 ACTIVE 标 INTERRUPTED | 不是“所有取消都会暂停”；也不是“所有 ServiceError 都会自动暂停” |
| 覆盖认证 | 静态/编译/执行证据审计模块已有独立资产 | service 认证仍恒 NOT_COMPUTED、full_suite_accepted=false，设备 NOT_PROBED |

还没有：前端、远端编排器、run-job.sh、两张新表、SSE、远端结果导入桥、用户/团队治理、正式榜单、配额系统、编译器安装制品上传协议。CANNBench 的 wheel/build.sh 布局与本项目的 manifest.yaml + 源码提交协议也不能混为一套。

## 3. 需要在开工前收紧的设计假设

### 3.1 插件冷恢复依赖必须显式化

首轮不是“start 不成功”，而是“云端已运行、旧会话的本机 SSH 无法恢复隧道”。bootstrap 日志为 `timeout waiting for refreshed connect_url`，外层 SSH 表现为 `kex_exchange_identification: Connection closed by remote host`。

本机安装的 1.1.2 实际文件位于 `dist/src/helper/forward-bootstrap.js`。核验当前源码发现：

1. bootstrap 调用 hub 的 `/bootstrap-forward`；
2. hub 没有可用隧道时返回 refreshRequired/requestId；
3. bootstrap 轮询 `/bootstrap-forward-status`；
4. 扩展宿主的后台刷新逻辑要求受信任 workspace、有效插件 OAuth，并负责取新 connect_url；
5. 扩展通过 `/bootstrap-forward-resolve` 把新地址交回 hub。

**本轮源码不能支持“仅有 SSH config、关掉 VS Code 后就必然能恢复一个全新的 hub”这个无条件结论。** 已存在可用 hub 的复用与冷启动后的地址刷新是两个场景。插件版本字符串相同也不足以锁定行为，本轮已保存三个 JS 文件 SHA-256。

直接用落盘 serverAddr 调 `/create-forward-hub` 也返回 WSS 握手 400，说明缓存地址本轮不可用；不能仅凭 400 就断言唯一原因是令牌过期。启停凭据访问插件 API 得 401，则说明两套凭据在本轮不可互换，不代表插件账户本身一定已过期。

本轮最终已通过有效插件刷新恢复通道并跑通评测。无人值守验收仍需验证 **冷 hub + 实例重启 + 无人工插件操作** 的完整路径；目前不能把一次用户协助的成功连接推导成任意会话冷恢复成功。刷新失败时不要重复启动实例。原架构 §2、§10.1 的“无阻塞”以及 §10.3 的“未复现”需与此记录一起解读。

### 3.2 `final` 即 stop 与补取远端报告的时序有冲突

目录跨轮持久化不能证明 SHUTDOWN 后 SSH 可访问。最终执行轮已得到受控对照：同一 SSH 配置在运行期间成功执行、上传并取回文件；确认 SHUTDOWN 后命令在 22 秒预算内无法完成。它不证明任意时长后绝对不可达，但足以否定首版对“停后立即补取”的无条件依赖。

建议首版时序：**执行结束 → 拉取原始结果/身份/必要产物并校验 → 确认本机持久化 → stop**，取回阶段设置短硬上限；超时照常 stop，将任务标为“结果不完整/待恢复”，不伪称所有产物已归档。报告可在本机生成，以缩短卡时。final 得分可先展示，但“结果已出”与“实例释放完成”分开呈现。

若坚持 final 后立即停，须先证明 final 发出前所有必要字节都已到本机，或提供另外一个**已实测**的停机可读存储通道。该通道目前没有证据。

### 3.3 资产版本要锁三个指纹，不能只锁 corpus manifest

当前 v0.9 配合当前源码会被 verify_assets 拒绝；本轮未重写旧版冻结索引来凑通过。v0.18 校验通过，服务默认也是 v0.18，而 CLI 默认仍为 v0.5、run_ascend_suite 默认仍为 v0.9。**远端必须显式传 corpus，不能依赖默认值。**

建议 Job 锁定：manifest_hash（case 身份）+ assets_hash（判定资产身份）+ bundle_sha256（实际同步的完整引擎/服务版本）。当前工作树有大量既有修改和未跟踪源码，单记 git HEAD 不能复现本轮代码。

文件通道清单不能只打包 `src/ + corpus/ + run_ascend_suite.py`。`corpus_assets.FRAMEWORK_FILES` 当前还引用大量 scripts/templates、spec、`reference/ascend-official` 两份 JSON 以及 `corpus/v0.9-official-candidates/manifest.json`。即使跑 v0.18，这些文件仍是校验依赖。首版建议按实际依赖清单全量包同步，保留目录结构，远端复验指纹后执行；优化增量传输放后续。

本轮已构造并成功上传包含 314 个仓库文件及提交套件的约 1.4 MB gzip 包，保存逐文件 SHA-256；远端 sha256sum、verify_corpus_files、verify_assets 均通过。这是验证资产，不是已实现的生产 orchestrator。

### 3.4 stdout 不能直接冒充既有事件协议

`supervisor.py` 明确将 stdout/stderr 当诊断流。权威引擎事件由 `evalchild._emit()` 写 events.jsonl，现有字段叫 `event`，没有 runner event_seq；`final` 也不是既有引擎事件。

建议首版从远端**专用事件文件**取回完整 JSONL 行，记录字节偏移，断连后补读；将旧事件包在版本化 envelope 中，不解析普通 CLI stdout 来“识别 case”。取回协议应拒绝半行、错误 job_id、超大行和同序号异内容，原始诊断日志另外保存。

### 3.5 JSONL 可恢复 case 结果，不含完整运行身份

本轮在新数据库实测 `cli import --run 1` 返回 1，提示“请先创建运行记录”。`cmd_import()` 不负责创建 corpus/compiler/run/submission 元数据。

本轮另将真机数据库复制到本机，保留身份元数据、删除 case_result/run_score/group_score 后，重新 import 真实 JSONL，再 score/report 成功，得到 20 PASS、两项功能分 100%、perf_score=null。该试验证明了带身份种子库的恢复路径，不是“裸 JSONL 重建全部元数据”。

远端回传至少包含 manifest 与资产指纹、编译器身份、env.json/env_fp、profile、submission 指纹、run 状态/范围、结果 JSONL；先创建/映射运行记录再导入。若从 events 恢复，须提取 `event=case` 的 record，并核对 case 集合。不能把 events.jsonl 原样传给结果导入器。缺失 final 时也不得把部分结果的成功率当全量得分。

### 3.6 本机开发仍有同源条件

loopback 不自动等于同源。若前端开发服务器在 5173、后端在 8765，当前 Origin/Host 校验会拒绝跨源请求。首版应让前端与 API 同源，或在前端开发服务器配置保留后端 Host 的 API/SSE 代理。无需为此开放公网或放宽现有 Host 白名单。

### 3.7 凭据与产物边界

仅检查形态发现当前 origin URL 仍有密码段；未输出该值，未修改远端配置。原设计提出的凭据移出 URL/轮换工作尚不能视为完成。同步包排除了 .git、.env.gatecheck、本机 OAuth、SSH 私钥与 bootstrap JSON。不要把 git remote、完整环境变量或 WSS 地址带入公开报告。

## 4. 将原 §14.2 未决项收敛为首版实现约束（建议，尚未实现）

| 原未决项 | 建议首版规则 |
|---|---|
| 回传、事件字段、序号（1/2/6） | 远端专用 JSONL 文件 + 本机拉取；Runner 包装层持久递增 event_seq；envelope 含 protocol_version/job_id/event_seq/kind/payload；崩溃后从日志末尾续号，不从 1 重启 |
| SSE（3） | 统一 `/api/v1/evaluations/{id}/stream`；id 使用本机提交成功的全局 seq；Last-Event-ID 断点续读；15 秒注释心跳、2 秒 retry；每批 200 条，空批等待，不当终态；连接受限 |
| DDL 与映射（4/5） | 与 objects/events 同库，版本化迁移；job_id 唯一并关联已有 task_id；队列加 (status,created_at,id) 索引；终态事务更新；SQLite 原子领取而非先查后改 |
| 隧道刷新（7） | 将“插件有效认证、受信任 workspace、刷新服务可用”作为显式 preflight；冷 hub 场景单独验收；失败即停且保留原因，不静默循环 |
| 同步/路径（8/14/15） | 首版全量、ZIP 远端再 ingest；job_id 独立目录；显式 corpus；打包前本机校验、远端校验再执行；不一致直接失败并 stop；只允许一次有界传输重试 |
| 排队（9） | 一个实例、一个 orchestrator 锁、FIFO、每次一个 Job；首版不批次合并；故障恢复先核实实例与遗留进程，不自动重跑已运行 Job |
| 前端（10） | 持久 status 与展示 phase 分开；展示“申请/启动/同步/执行/评分/取回/释放”；保留五种成功阶段与四种失败终态；completed 不等于 all_pass |
| 健康（11/13） | source CANN 后实际 import、验证 triton.language.extra.cann、torch.npu 可用性和设备小探针；find_spec 仅作发现，不能代替 import/运行检查 |
| 公网（12） | 本期本机同源；保留现有绑定，暂无公网扩展 |
| 复用边界（16） | 平台复用 upload/create_task/query；远端包装层以固定 argv 调既有 CLI run/score/report，不在 DB 存 shell，不用字符串替换本机路径 |

任务状态继续采用既有九态；编排 phase 存在 remote_job 或事件 payload。RUNNING 可包含申请/启动/同步/执行，SCORING 对应评分，REPORTING 可包含生成/取回/释放。COMPLETED 要求必要结果已在本机持久化、实例停止已确认；停止失败应保留“实例仍未释放”的独立字段，不能只报评测成功。

两张表的 `(job_id,event_seq)` 唯一约束只能防重复，**不能检测同序号异内容**。重放相同 payload 允许幂等，内容不同应拒绝并告警；事件插入与派生状态更新要同事务。SSE 以全局 seq 续读允许单个 Job 的序号不连续，不应把其他 Job 造成的全局 seq 间隙误判成丢事件。

## 5. CANNBench 参考复核

只读请求了 [平台首页](https://cannbench.com/)、[提交规范路由](https://cannbench.com/docs/submission-validation) 和 [公开榜单接口](https://cannbench.com/api/public/leaderboard?version=latest&mode=solution&detail=summary)。未向 CANNBench 提交算子、创建方案或生成 Token。

公开接口本轮返回 total=239、available_hardwares=[910c,950dt,950pr]、available_versions=[2.0,1.1]；返回项仍含 benchmark_version / benchmark_version_source、detected_language、best_job_id、op_names、metric、hidden_metric 等字段，metric 仍有总/通过用例、算子数、两类几何平均加速比及总分。可见 2.0 / 2.0.0 双版本字段，语言枚举含 triton_ascend。

原文的“44 个方案”等是 11 月 11 日筛选态快照，当前 API 的 total 也不能脱离请求参数当成同一 UI 筛选的对照。前端应显示实际筛选范围与时间，不能把这些数写死。

浏览器工具本轮不可用，因此 ZIP 10 MiB、固定 Owner、Token 永久有效、登录态表单、KPI 联动、他人 Job 权限、对比导出等仍沿用原文历史证据，**没有升级为本轮复验通过**。“UI 显示 10 MiB”也仅证明文案，未经阈值两侧上传请求不能断言后端实际限制就是 10 MiB；“他人详情加载中”不能单独证明访问控制正确。

## 6. 开工顺序与验收门禁

1. **通道基线**：已有插件刷新后的 SSH、多行脚本、npu-smi、SCP 上传与 SSH 文件取回实测。补充完全无人协助的冷 hub 刷新验收；失败路径 finally stop，并确认 SHUTDOWN。
2. **版本锁定**：选择通过校验的 v0.18；补齐迁移工具缺失模块或明确隔离其已知失败；记录当前工作树逐文件指纹。
3. **最小远端闭环**：传整包、校验 ZIP/manifest/assets；实际 import 依赖；跑 NEW-ABS 全部 20 个冻结 case，再跑 EXEC-WRONG/EXEC-INPUT；核对状态链、case 集合、功能分、结果与报告下载哈希。本轮该小范围试验已通过，并补充本机离线重放；整套 independent/normal/full 尚未全跑。
4. **控制与事件**：两表、领取互斥、远端文件读取、幂等回放、断连补读、终态恢复；先文件/轮询验证，再接 SSE。
5. **本机前端**：上传、范围选择、排队/资源状态、逐 case 结果、分数口径、报告；状态从真实 API 来，认证不靠 PASS 推断。
6. **运营与榜单**：设备准入、预算/停机看门狗、统一性能锚点、正式榜单分区与版本治理；完整 1385 case 和 standard 性能另行验收，不能由这次小范围 smoke 推导。

本轮结束时已确认实例 SHUTDOWN。可以按上述顺序开工；完全无人值守冷恢复、事件桥/SSE、全量与性能仍是明确的后续验收项，不应由这次小范围 smoke 替代。

## 7. 最终交付索引

- `attempt-3/cloud-result.json`：最终完整执行轮，evaluation_returncode=0、final_state=stopped。
- `attempt-3/remote-summary.json`：三个真机场景、60 条记录的对照摘要。
- `attempt-3/retrieved/evidence/`：依赖、远端资产校验、HTTP 原始观测与下载产物。
- `attempt-3/artifact-hash-verification.json`：80 个产物的本机独立哈希复核。
- `attempt-3/offline-replay/verification.json`：带运行身份的真实 JSONL 离线导入、评分与报告恢复。
- `backend-tests.xml` / `l0-pytest.xml`：后端专项与全量 L0 的原始机器可读结果。
- `transfer-manifest.json`、`bundle-safety-check.json`：本轮完整传输资产指纹与已知凭据值/禁止路径扫描，零命中。
- `evidence-index.json`：最终证据文件大小与 SHA-256 索引。
