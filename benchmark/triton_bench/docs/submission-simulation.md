# 用户提交全流程仿真与接口联调

## 0. 范围与证据边界

本阶段承接已有的 ZIP 摄取、内容校验、隔离执行、评分和报告链路，补齐
用户提交场景、持久任务编排与前端可调用的最小 HTTP 接口。

**包级校验直接复用，不新增包级规则或攻击样例。** 本阶段生成的全部 ZIP
都应通过现有摄取层；差异发生在清单内容、实现文件、入口契约、编译、执行
或结果上。已有包级对抗集不计入本阶段新增场景数。

**2026-10-28 的进度口径：**首轮真机批次 `20261028-032727-036552` 已完成
normal 47 个场景，45 个符合预期、2 个样例存在设计偏差。9 个正常提交均
完成 ZIP 上传、真机执行、评分和产物下载，共192条 PASS 记录；其中重复
提交的相同用例不能重复算作独立用例。原有实现重新打包的本轮结果与其
历史通过记录分别保存，不相互替代。

后续新增20个独立实现合法提交场景，使用独立编写的内核，不复制既有
官方提交源码。新增集目前仅有本机构建、契约检查和 CPU 语义测试，尚未
取得 NPU 结果，不声明其已经真机 PASS。当前共73个场景定义。

本轮本机全库回归为1417通过、190跳过、21个 subtests 通过；跳过项不算
验证成功。CPU 内核解释测试不加载真实 Triton 编译器，HTTP 编排集成测试
显式注入设备结果夹具并调用真实评分及报告；两者都不是新增提交集的真机
编译或执行证据。

本机沙箱禁止绑定 loopback 端口，因此真实监听端口的两项测试跳过；其余
HTTP 测试使用序列化请求和内存传输，经过同一个 HTTP Handler。这验证了
接口分发，不等于实际网络连通。Worker 测试注入明确标注的设备结果夹具，
评分与报告调用真实 CLI，不把夹具当成设备执行证据。

## 1. 已确认的契约

### 1.1 提交入口与版本绑定

所有模拟用户提交都上传 ZIP。HTTP 接收端调用既有 `ingest_zip` 和
`validate_static`，不在 HTTP 进程导入用户实现；执行 Worker 再将原始 ZIP
交给既有 `triton_eval run`。不能用直接传目录绕过用户提交入口。

默认显式绑定 `v0.18-scan-certification-candidates`：74 个算子、1385 个
冻结用例。服务启动、样例构建和派发前检查冻结完整性；每个任务保存清单
指纹、上传包指纹、完整计划用例编号。独立预期清单同时保存基线源码指纹、
计划用例与算子权重，不能跟着执行结果反向修改预期。

API 不接受用户指定服务器路径、case 数量限制、参考实现、容差或执行命令。
专项任务执行选定算子的全部冻结用例；全量任务不得通过 `operators` 缩小
范围。单算子满分必须连同任务范围披露，不能被解释成全量通过。

### 1.2 不改变已有评测规则

- 缺清单等 ZIP 摄取错误沿用 `SUB-PKG-*`，不建设新的包级测试矩阵。
- 清单存在但字段缺失、未知算子、源码或入口错误沿用现有校验器归因。
  例如，现有校验器将清单解析失败统一记录为 `SUB-LOD-001`；接口返回
  该编号及诊断，不另造一套评分规则。
- 空算子列表与重复声明由服务编排策略分别拒绝为 `SERVICE-EMPTY`、
  `SERVICE-DUPLICATE`；不修改已有 `SubmissionManifest` 或评测规则表。
- 注册表存在、但不在服务绑定测试集中的算子，返回
  `SERVICE-CORPUS-OPERATOR`，不悄悄切换测试集。
- 多算子包中有入口加载错误时，保留现有整包校验拒收行为；不声称已实现
  加载失败后的部分计分。`MIXED-LOAD` 专门记录这一行为。
- 已完成任务可以包含编译失败、运行失败或错误结果。它们是评测结果，
  不等于编排服务失败；环境、参考实现和冻结用例自身异常则不发布有效分数。

### 1.3 评分与认证

评分和报告分别调用已有 `score`、`report`，不在服务内新增评分器。
仿真客户端依据独立清单中的权重和实际用例终态，另外核对功能得分、
严格通过率及用例集合，避免“仅返回一个分数就算通过”。

没有有效计分项时，API 的 `score` 为 `null`；`smoke` 没有性能得分。
`standard` 使用已有真实性能测量流程，但每个任务有独立执行数据库，
本阶段不共享跨用户性能锚点，不能据此构建可比的正式性能榜单。

静态覆盖、编译认证、执行认证不由功能 PASS 推导。本阶段接口中三项认证
均明确为 `NOT_COMPUTED`，`full_suite_accepted=false`。若后续需要发布认证，
必须接入既有证据审计器，再单独扩展契约，不能拿模拟数据补齐。

## 2. 提交场景清单

生成器：`src/triton_eval/submission_simulation.py`。

首版共50个场景；当前共 **73 个场景**，其中 normal 50 个、independent
20 个、full 1 个、slow 1 个、disruptive 1 个；normal 内包含14个内容拒收
场景。所有数值均指场景，不是冻结评测 case，也不是已完成的 NPU 次数。

| 分组 | 场景 | 目的 |
|---|---|---|
| 合法专项 | `OK-EXP`、`OK-ADD`、`OK-SOFTMAX`、`OK-TRANSPOSE` | 属性、双输入、归一化、布局变换 |
| 合法复杂算子 | `OK-MATMUL`、`OK-CUMMIN`、`OK-FLOORDIV`、`OK-GROUPED` | 矩阵乘法、多输出、历史修复实现 |
| 参数迁移 | `OK-ARGMAX`、`OK-SWIGLU`、`OK-CROSSENTROPY` | 通过实际 ZIP 评测检查迁移后的算子参数契约 |
| 合法组合 | `OK-MULTI` | 同包多算子及共享源码 |
| 合法全量 | `OK-FULL` | 74 算子、1385 用例端到端；full 档单独运行 |
| 元数据 | `META-YAML`、`META-ID`、`META-OPS-TYPE` | 语法、缺标识、列表类型错误 |
| 字段完整性 | `META-NAME`、`META-ENTRY`、`META-PATH`、`META-EXTRA` | 缺字段及未知字段 |
| 声明与实现 | `META-UNKNOWN`、`META-FILE` | 未登记算子、声明的实现文件缺失 |
| 编排边界 | `META-DUP`、`META-EMPTY` | 重复声明与空提交 |
| 源码内容 | `STATIC-SYNTAX`、`STATIC-DELEGATE`、`STATIC-CPU` | 语法错误、代算、主机侧往返 |
| 入口 | `LOAD-MISSING`、`LOAD-NONCALLABLE` | 入口缺失、不可调用 |
| 签名 | `LOAD-COUNT`、`LOAD-NAMES`、`LOAD-POSONLY` | 个数、名称、仅限位置参数 |
| 导入 | `LOAD-IMPORT`、`LOAD-RAISE`、`LOAD-EXIT` | 依赖缺失、异常、SystemExit |
| 执行异常 | `EXEC-RAISE`、`EXEC-FAKE-COMPILE` | 真实运行异常、不得按异常文本误判编译失败 |
| 数学及形状 | `EXEC-WRONG`、`EXEC-SHAPE` | 错误语义、输出形状错误 |
| 返回契约 | `EXEC-DTYPE`、`EXEC-NONTENSOR`、`EXEC-ARITY` | dtype、非张量、输出数量错误 |
| 真编译与启动 | `COMPILE-UNDEFINED`、`LAUNCH-ARGUMENT` | JIT 未定义符号、缺少启动参数 |
| 混合提交 | `MIXED-RUNTIME`、`MIXED-LOAD` | 运行错误隔离、加载整包拒收 |
| 执行防护 | `EXEC-DEVICE`、`EXEC-INPUT` | 输出设备错误、输入被修改 |
| 多输出 | `EXEC-CUMMIN-MISSING`、`EXEC-CUMMIN-INDICES` | 缺第二输出、仅索引错误 |
| 部分用例 | `EXEC-PARTIAL` | 同一算子部分 dtype 失败；验证非二元功能得分 |
| 慢场景 | `EXEC-TIMEOUT` | 超时归因；slow 档单独运行 |
| 中断场景 | `EXEC-EXIT` | 非自然退出、结果补记、暂停派发；disruptive 档单独运行 |

大部分场景要求精确终态集合。`LAUNCH-ARGUMENT` 允许 `RUNTIME_FAIL` 或
`COMPILE_FAIL`，因为不同版本可能在启动绑定或编译参数绑定时拒绝；不能
接受 `PASS`、环境错误或没有结果。`MIXED-RUNTIME` 还逐算子核对归因，
不能只检查整包里出现过两种状态。

已知需要真机确认的诊断边界：

- v18 开启编译证据捕获时，非张量返回可能在输出绑定的 `as_outputs` 处
  先抛 TypeError，因此 `EXEC-NONTENSOR` 的当前预期是 `RUNTIME_FAIL`，
  不能声称它证明了 `SUB-EXE-001`。
- `EXEC-TIMEOUT` 的目标预期是运行失败；现有执行器的 TimeoutError 分支
  设置 `compiled=False`，存在归为编译失败的风险。这里保留可发现差异的
  预期，不修改冻结执行框架，也不为凑 MATCH 而把风险隐藏掉。

首轮差异修正：`EXEC-WRONG` 原先直接返回输入，3条非连续输入先触发输出
布局违规；现改为分配同形状、同设备、同 dtype 的连续零输出，只保留数学
错误。`EXEC-INPUT` 原先填零，对全零输入没有造成数值变化；现改为就地
加一，避免全零样例没有触发输入指纹变化。预期状态、判定器与容差不变；
修正后的 normal 场景尚需真机复验。

### 2.1 独立实现的合法提交集

源文件：`scripts/templates/independent_submission_kernels.py`；构造器：
`src/triton_eval/independent_submissions.py`。这些是已登记算子的新实现，
不是新增未登记的算子语义。生成 ZIP 只包含所声明入口及依赖的独立源码，
不夹带旧官方实现，不导入评测器或参考实现，不用历史 PASS 替代重新运行。

| 场景 | 合法边界 |
|---|---|
| `NEW-ABS`、`NEW-NEG`、`NEW-ADD` | 单输入、双输入；FP16/FP32；主输入 stride=2；辅助输入独立步长 |
| `NEW-EXP` | 默认及非默认 base/scale/shift；逐元素 FP32 计算后写回 |
| `NEW-SOFTMAX`、`NEW-REDUCE` | 单列、非二次幂归约；dim 正负写法；keepdim 两种输出形状 |
| `NEW-TRANSPOSE`、`NEW-TRANSPOSE-VIEW` | 相同语义分别返回连续拷贝和合法非连续视图；包含 int16 |
| `NEW-MATMUL` | 非对齐矩阵、归约尾块及长 K；不复用旧分段矩阵乘内核 |
| `NEW-CUMMIN` | values/indices 双输出、非末轴、相等最小值末次索引、百万级长度与跨块尾部 |
| `NEW-MULTI-SHARED`、`NEW-MULTI-SPLIT` | 相同的4个算子分别共享源码和独立文件，逐包重新评测 |
| `NEW-ENTRY`、`NEW-KEYWORD-ATTR`、`NEW-KWARGS` | 非默认入口名、仅关键字属性、契约参数加可选 kwargs |
| `NEW-UNICODE-ID`、`NEW-REORDERED` | 合法 Unicode 提交标识、非注册表顺序的声明 |
| `NEW-SCOPED` | 多算子提交只选择其中一个算子；仍跑该算子的全部冻结用例 |
| `NEW-REPEAT-A`、`NEW-REPEAT-B` | 字节相同的 ZIP 和相同 submit_id；不同请求必须独立建提交及任务 |

20个场景声明9类算子，计划产生554条用例执行记录，涉及168个不同冻结
用例；重复运行用于检验提交隔离，不增加测试集覆盖率。每个场景在独立
预期清单保存输入 dtype、shape、attrs、连续性、seed 与边界标签，并保存
独立模板指纹、ZIP 指纹、完整用例编号和权重。不得从实际结果倒推预期。

每个独立提交还验证上传和建任务的同键幂等、异请求不串单、提交标识保留、
`QUEUED → RUNNING → SCORING → REPORTING → COMPLETED` 状态事件、结果与任务
包指纹一致、计划和实际用例集合一致、功能分与严格通过率均为100%，以及
报告和原始结果下载指纹一致。新实现若算错，应保留正常评测产生的错误
终态；这种情况说明平台出了结果，但不能算作正确实现的正例验收通过。

验收目标是：符合既有登记契约及输入域的实现不因源码写法、文件拆分、
入口名称或提交标识而被误拒收；每次提交均能产出可追溯结果。有限样例
不能证明所有合法实现都可运行，因此报告明确标记普遍支持为 `NOT_PROVEN`。
本轮不扩大冻结输入域，不注入空张量、标量、额外 dtype 或未登记形状；
这些不能仅因框架可创建就视为当前评测任务的合法输入。若登记契约允许
但冻结清单尚无对应样例，应另行建设并冻结输入版本，不能暗改现有 case。

## 3. 后端接口与持久任务

实现：`src/triton_eval/submission_service.py`、
`src/triton_eval/submission_http.py`。采用标准库 HTTP 和 SQLite，无新增服务
依赖；仅用于受控联调。`/openapi.json` 提供 OpenAPI 3.1 接口描述。

| 方法 | 路径 | 返回 |
|---|---|---|
| GET | `/api/v1/catalog` | 冻结版本与指纹、算子名/入口签名/用例数/权重、档位、Worker 配置；不冒充设备健康探测 |
| POST | `/api/v1/submissions` | 原始 `application/zip` 请求体，返回平台提交 ID、包指纹、内容检查结论 |
| GET | `/api/v1/submissions/{id}` | 提交详情及结构化诊断 |
| POST | `/api/v1/evaluations` | JSON：`submission_id`、可选 `scope`、`operators`、`profile`；返回任务 |
| GET | `/api/v1/evaluations/{id}` | 状态、范围、计划/已完成用例数、进度、Worker 状态 |
| GET | `/api/v1/evaluations/{id}/events?after=0` | 最多200条增量状态事件及下一游标 |
| GET | `/api/v1/evaluations/{id}/results` | 用例结果、计分范围、功能得分、性能可用性、认证状态 |
| POST | `/api/v1/evaluations/{id}/cancel` | 取消结果或取消请求已记录的任务 |
| GET | `/api/v1/evaluations/{id}/artifacts` | 允许下载的产物、大小和完整 SHA-256 |
| GET | `/api/v1/evaluations/{id}/artifacts/{name}` | 附件下载；固定白名单，不接受任意路径 |

文件上传不是 multipart，而是 ZIP 原始二进制请求体；浏览器可直接提交
File。前端开发服务器使用同源反向代理或 SSH 转发；本服务不开放跨源访问。
平台 ID 不采用包内 `submit_id`，同名包不会覆盖其他提交。

用例结果的 `log_path` 为任务产物文件名，可通过下载接口取对应日志；评分
对象沿用 `RunScoreRecord`，分组结果返回结构化对象而不是数据库 JSON 字符串。
OpenAPI 的用例与分数字段直接取自已有模型，避免与执行器字段手工漂移。

两个创建接口支持 `Idempotency-Key`：同一键、同一内容返回同一记录；同一
键、不同内容返回409。评测请求不接受任意附加字段。HTTP 错误返回
`error.code` 和 `error.detail`，内容拒收作为持久提交记录返回201，再通过
`status=REJECTED` 表达，不与 HTTP 传输失败混淆。

任务状态：

```text
QUEUED → RUNNING → SCORING（有有效计分项时）→ REPORTING → COMPLETED
       ↘ CANCELLED
RUNNING / SCORING / REPORTING → REJECTED / FAILED / CANCELLED / INTERRUPTED
```

进度来自现有执行数据库，不来自提交代码 stdout。每个任务使用独立数据库、
运行目录和解包目录；不使用 `latest` 查找运行，不复用上一任务的结果。
自然完成后核对包指纹、结果用例集合及必要产物，才允许标为 COMPLETED。

任务与状态事件在同一 SQLite 事务内更新。单工作目录加进程锁，单 Worker
串行派发。重启保留排队任务，将原运行中任务标 INTERRUPTED，不自动重跑；
运行中取消终止进程组并暂停派发，确认终止前不伪称取消成功。

## 4. 运行方式

### 4.1 本机构建与内容流程

仓库根目录，本机已配置的 Python 3.11 环境：

```bash
.venv/pattern-build/bin/python scripts/submission_simulate.py build \
  --dest runs/submission-simulation/suite

.venv/pattern-build/bin/python -m triton_eval.submission_http \
  --root runs/submission-simulation/service --port 8765
```

另一个终端执行：

```bash
.venv/pattern-build/bin/python scripts/submission_simulate.py run \
  --suite runs/submission-simulation/suite --mode content \
  --output runs/submission-simulation/content-observations.json
```

content 档只验证14个内容拒收场景，不启动 NPU。服务未带 `--execute` 时，
合法提交仍可进入排队状态，但客户端不会把未执行任务算作通过。

### 4.2 真机真实评测

以下使用真机已有评测环境中的 `python`，不安装或替换 triton，不升级真机
Python。常规真机验证在仓库根目录执行：

```bash
bash scripts/submission-simulation-verify.sh
```

入口复用 `scripts/env-bootstrap.sh` 初始化环境，自动构建提交集、启动
loopback HTTP 服务、默认执行 independent 合法新实现集、关闭服务并通过既有
`scripts/publish_ascend_results.py` 上传至 `npu-results`。失败时同样尝试
上传日志，上传失败则保留原件并给出补传命令。每批使用独立工作目录；
发现旧批次暂停标记时停止，不自动绕过健康确认。需要复验首轮修正的负例
时执行 `bash scripts/submission-simulation-verify.sh normal`，不与新实现
验收混算。不自动执行 full、slow、disruptive，也不出三项覆盖认证结论。
该入口仅用于受控隔离的真机环境。

需要手动联调时，先构建同一套提交，再启动执行服务：

```bash
python scripts/submission_simulate.py build \
  --dest runs/submission-simulation/suite

python -m triton_eval.submission_http \
  --root runs/submission-simulation/service --port 8765 \
  --execute --acknowledge-isolation
```

另一个真机终端执行常规回归和全量合法提交：

```bash
python scripts/submission_simulate.py run \
  --suite runs/submission-simulation/suite --mode independent \
  --output runs/submission-simulation/independent-observations.json

python scripts/submission_simulate.py run \
  --suite runs/submission-simulation/suite --mode normal \
  --output runs/submission-simulation/normal-observations.json

python scripts/submission_simulate.py run \
  --suite runs/submission-simulation/suite --mode full \
  --timeout 86400 --output runs/submission-simulation/full-observations.json
```

slow、disruptive 分别用对应 `--mode` 显式执行，不在常规回归中自动触发。
它们可能使 Worker 暂停，必须单独安排设备与进程健康检查。
真实性能流程可显式添加 `--profile standard`，不改变计时参数。

受控真机的一键入口为 `bash scripts/submission-simulation-verify.sh independent`，
以及 `bash scripts/submission-simulation-verify.sh normal`。两者均走真实 HTTP
上传 ZIP、内容解析、创建任务、Worker 编译执行、计分、下载报告和原始结果的
完整链路，不直接调用裸内核替代上传。默认 smoke 档只验证功能出分，不采集
正式性能分数；它不包含浏览器前端，也不能证明任意合法提交都支持。

退出时无论评测成功还是失败，入口均尝试上传该批结果到 `npu-results`，并保留
原评测退出码；上传失败则返回2，原评测退出码仍保存在 `execution.json`。
发布脚本从自身路径定位源码，不依赖调用目录、环境激活或预先设置 PYTHONPATH。
上传失败时可直接复制入口打印的绝对路径补传命令；补传成功不代表评测通过。

客户端将每个提交、任务、用例、评分和差异保存到输出 JSON，增量落盘；
报告和原始结果等产物经下载接口获取，并核对 SHA-256，保存到输出文件旁
的 `*-artifacts/`。退出0仅表示所有选中场景与预期一致，不表示所有算子
都算对，更不表示任务书或性能验收完成。

## 5. 安全与后续验收

**子进程不是完整安全沙箱。** 当前服务仅绑定127.0.0.1，无用户鉴权、
多租户授权、容器/虚机沙箱、设备配额和正式榜单，不能开放公网接收任意
未知提交。`--acknowledge-isolation` 是部署前提确认，不是隔离能力本身。

执行主机必须由部署方提供限制网络、移除凭据、限制资源和隔离文件系统的
运行环境。下载的原始日志和报告可能含执行环境信息；只允许受信任联调方
访问，不能直接作为公网产物服务。JSON 诊断仅做本机路径替换，不宣称完成
敏感信息全面脱敏。

暂停标记为服务根目录的 `worker-paused.txt`。发生非自然退出、环境异常或
服务重启后，必须先停止服务，人工确认无遗留进程、设备健康且原因已处理，
再删除该单一标记并重启；不能用自动清理标记绕过健康确认。

后续验收顺序：

1. 完成 independent 真机合法提交验收；前端同源代理仍需后续接入验证。
2. 复验 normal 两个修正样例，逐项核对阶段、规则、终态、分数与下载产物。
3. 修复差异时区分提交样例缺陷、接口编排缺陷和既有评测器缺陷，不改变
   输入/参考/容差来凑通过；如需修改冻结框架，另行构建、冻结和审计版本。
4. 完成 full 合法提交闭环；不叠加旧轮次冒充本轮全量结果。
5. 单独安排 slow/disruptive 和标准档性能流程，最后整理真机验收记录。

本阶段没有修改既有冻结 corpus、官方算子实现或评分公式，也没有把新增
提交场景数量计入任务书测试集规模。
