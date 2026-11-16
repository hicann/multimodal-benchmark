# 远程 NPU 评测闭环架构设计（按需启停 + SSH 一次性 Job）

> **稳定性约束补充**：[SSH 稳定性复盘](ssh-stability-review-20261113.md)。每个后台 Job 必须独立维持本机 hub 的 `/heartbeat`，不能依赖 VS Code 窗口续租；插件 1.1.2 在无心跳/无活动连接约 35 秒后会主动退出。一次 SSH 成功不等于整段任务生命周期已获保障。该问题已受控复现并加固，正常轮及主动断连轮均已通过真机验收。

> **2026-11-13 实施入口**：[远端 Worker 使用与验收说明](remote-worker.md)。两张表、编排器、Runner、SSE、结果回收和独立停机看门狗已落地；下文“目前不存在/待实施”保留为 11 月 12 日的设计快照，当前状态以实施说明为准。

> 验证时间：2026-11-12；**v2 修订同日**：按「本机部署 + 两张 DB 中间表」重构，补齐 HTTP 目标地址、已实测指令清单、资料地图与覆盖度自查。
> 本文与 `docs/cannbench-platform-reference.md`（CANNBench 前端参考）配套：前者是**平台形态与交互参考**，本文是**后端到真机的执行链路设计**，也是另一 Agent 开工的**主索引**。
> 证据分级：**[实测·真机]** = 在真实 Ascend NPU 实例上执行过；**[实测·本机]** = 本机命令/插件行为观察；**[代码核验]** = 本仓库源码事实；**[设计]** = 尚未实现的拟议方案；**[待实施]** = 明确需要补写的工程项；**[缺口]** = 尚未定义、开工前必须确认。
> **本文只描述设计，不落地实现。** 文中 `orchestrator.py`、`run-job.sh`、两张新表、SSE 接口均为拟议命名，仓库中目前不存在；本文不写入任何令牌、私钥或 WSS 地址的值，只记录路径与字段名。
> 落点对应 `docs/评测体系要求.md` 中「面向持续运行的评测平台与工程资产建设」一项。

---

## 1. 要解决的问题

原流程的断裂点：平台后端在真机 NPU 正式评测时，需人工把仓库推到 GitCode、用 `~/cann-lab-http-retry/start.sh` 拉起 CANNLab Web IDE、再由人在 IDE 终端敲命令，`run / score / report` 与出分都发生在浏览器里。引入前端后，无法形成「上传算子 → 一键运行 → 实时出分 → 反馈平台」的闭环。

**硬约束**：NPU 卡时极其有限。实例平时必须 `SHUTDOWN`，**只有提交之后才允许触发启动**；实例内不允许常驻 Runner；任何路径（成功、失败、超时、注入失败）都必须释放实例。这一约束否决了常驻 Runner / 长连接 Agent 方案，决定了整体形态是 **serverless batch job**。

原始 5 步设想（历史，v1 评估对象）：

1. 前端提交算子，后端解析 ZIP、提取元信息；
2. 生成 `job.json` 推到 GitCode 形成 commit；
3. 编排器监听新 commit，启动 NPU 并执行 Job 脚本；
4. 出分后把结果传到另一个托管仓库；
5. 前端通过 SSE 获取结果。

其中第 3 步是原断裂点（现已打通），第 5 步存在语义误解（见 §3.3），**第 2、4 步在 v2 中被两张本机 DB 表替代**（见 §3.4、§3.5）。

---

## 2. 结论先行：链路核验表

| 环节 | 原始设想 | v2 状态 | 证据 | 是否为阻塞性卡点 |
|---|---|---|---|---|
| 1. 提交与解析 | 前端上传 ZIP、提取元信息 | **已具备** | `[代码核验]` `POST /api/v1/submissions` + `ingest / validate / discover`；`docs/submission-simulation.md` 仿真闭环已跑通 | 否 |
| 2. 控制信道 | `job.json` commit 到 GitCode | **已替换为本机 `remote_job` 表** | `[实测·真机]` SSH/SCP 通道打通；本机 DB 即可承载参数 | 否（流程减重） |
| 3. 启动 + 执行 | 编排器启动 NPU 并跑 Job | **能力已打通** | `[实测·真机]` `POST /start` → `ssh` 执行 `rc=0` → `bash -s` 多行脚本 → `npu-smi 25.5.0` 见 `910B3 OK` | **否（原卡点已消除）** |
| 4. 结果回传 | 传到另一托管仓库 | **已替换为本机 `remote_job_event` 表 + SSH 取回** | `[实测·真机]` `scp`、`ssh 'bash -s'`、远端出网 200；`[设计]` orchestrator 主动拉取 | 否（流程减重） |
| 5. 实时展示 | 前端 SSE 获取仓库结果 | **语义已纠正，SSE 待实现** | `[代码核验]` 现有 `/events?after=` 是任务级状态轮询，无 `text/event-stream` | 否（工程项） |

**判定：能力层面已无阻塞性卡点。** 剩余工作分三类，见 §10 与 §14：

- **工程缺口**：SSE、结果导入、前端状态机、编排器与 Job 脚本、两张表的 DDL 落地、双事件层桥接；
- **既有语义冲突**：`worker-paused.txt` 人工闸门、loopback/Host 绑定、非安全沙箱等；
- **残留链路风险**：`serverAddr` 临时凭据过期可能导致 bootstrap 失败（本次未复现）。

---

## 3. 端到端链路与设计思路

### 3.1 时序（v2，本机部署）`[设计]`

```
浏览器 ──① POST http://127.0.0.1:8765/api/v1/submissions (ZIP)──▶ 平台后端（本机）
平台后端 ──② ingest/validate_static/discover ──▶ 本机 service.sqlite: remote_job(QUEUED)
orchestrator（本机）──③ 轮询 remote_job 领取 QUEUED
orchestrator ──④ POST /start（GitCode 云端）──▶ 等 RUNNING（17–43s）
orchestrator ──⑤ scp 资产包（ZIP + corpus + 评测引擎代码）→ /mnt/workspace/jobs/<job_id>/（云端 NPU）
orchestrator ──⑥ ssh <host> 'bash -s' < run-job.sh <job_id> ──▶ 一次性 Job
NPU ──⑦ 逐 case JSONL 写 stdout / events.jsonl（远端持久目录）
orchestrator ──⑧ 同一 SSH 通道流式读回 ──▶ 本机 remote_job_event 表
平台后端 ──⑨ 按 (job_id,event_seq) 去重 ──▶ SSE ──▶ 浏览器
orchestrator ──⑩ try/finally POST /stop ──▶ SHUTDOWN（约 95s 确认）
```

**关键修正（v1 → v2）**：v1 时序 ⑦ 写「NPU POST 平台后端 `/internal/ingest`」。平台后端改到本机后**不成立**：云端实例的 `127.0.0.1` 指向云端自身，且本机没有公网入站地址。因此结果由 **orchestrator 经已建立的 SSH 通道主动读回**（stdout JSONL 流式，或 `scp`/`cat` 取远端文件），而不是云端主动回连。这一改动使 GitCode 结果仓库从主链路中彻底移除。

**表只传控制信息，不传字节**：ZIP 本体、解压后的算子文件、corpus 与评测引擎代码全部走 ⑤ 的文件通道；表仅用 `submission_sha256` / `corpus_fingerprint` / `remote_workdir` 与之关联。**⑤ 文件通道的正式定义见 §3.8**（此前仅为占位名词）；资产清单与校验链见 §3.7。

### 3.2 为什么是 serverless batch job `[设计]`

- NPU 只在一次 Job 生命周期内存在；实例内**无**常驻进程；常驻的是 NPU 外的 orchestrator（本机）。
- 所有路径必须 `try/finally` 关停：启动失败、SSH 注入失败、执行异常、超时、`final` 到达、无事件心跳超时，都触发 `POST /stop`。
- 排队不烧卡时：只有领取到 Job 才 `POST /start`。
- 零空转：`final.json` 落地即停；报告和 artifacts 允许在停实例后由平台侧从远端持久目录或本机落盘补齐。
- 卡时经济：批次合并、实例 RUNNING 预算、额度与预计消耗提示、超时预算硬上限。

### 3.3 SSE 语义纠正 `[设计]`

SSE 是 **平台后端（本机）→ 浏览器**，不是浏览器连 GitCode、也不是云端回连。正确链路：

```
orchestrator 读回事件 → 本机 DB → 平台后端 SSE → 前端
```

前端不能承诺秒级实时，应显式呈现环境申请与启动耗时（启动 17–43s）。

### 3.4 评估：两个 GitCode 仓库流程是否过重 `[代码核验]` + `[实测·真机]`

**结论：过重，且在本机部署前提下不成立，应移除主链路。**

| 流程 | 成本 | 实测/核验依据 |
|---|---|---|
| `job.json` commit 到 GitCode 作控制信道 | 只传几个参数（`job_id`、ZIP `sha256`、`profile`、`operators`、预算），却要 clone/commit/push + WebHook 或 `git ls-remote` 轮询；引入 Git 凭据、commit 幂等、分支冲突、秒~分钟级延迟 | `[代码核验]` `job.json` 内容清单见 §6.1；`[实测·真机]` `scp` + `bash -s` 已可直接传参数与脚本，无需仓库中转 |
| 结果 push 到另一托管仓库 | `publish_ascend_results.py` 用单 `npu-results` 分支 + `--force-with-lease` + `trim_records(keep=3)`；多写者各自基于自己的 `previous` 会互相裁剪；冲突报「结果分支已被其他运行更新，请重试上传」需人工补传；`upload-heartbeat.sh` 用普通 `git push` 且忽略 `.git/index.lock` | `[代码核验]` `scripts/publish_ascend_results.py`、`scripts/upload-heartbeat.sh` |
| 云端 → 本机回传 | 本机后端无公网入站，云端 POST 到 `127.0.0.1` 必然打到云端自身 | `[代码核验]` `submission_http.py` 绑 `127.0.0.1:8765` + Host 白名单；`[设计]` v2 部署决定 |

**减重结果**：控制面与数据面都收敛到本机两张 SQLite 表；GitCode 只保留三件事 —— CANNLab 实例启停/OAuth（ unavoidable，云端）、可选离线归档、人工可审计。`npu-results` 不再是实时出分通道，也不再是必需环节。

### 3.5 替代方案：两张轻量级 DB 中间表 `[设计]`

放在现有 `service.sqlite`（与 `objects` / `events` 同库），**只描述设计，不落地代码**。风格对齐基准见 §3.6。

**表 1 `remote_job`：Job 控制桥梁（替代 `job.json` → GitCode commit）**

```sql
CREATE TABLE IF NOT EXISTS remote_job (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id              TEXT    NOT NULL UNIQUE,
    submission_id       TEXT    NOT NULL,
    submission_sha256   TEXT    NOT NULL,
    corpus_fingerprint  TEXT    NOT NULL,
    profile             TEXT    NOT NULL,
    operators_json      TEXT    NOT NULL,
    budget_s            INTEGER NOT NULL,
    status              TEXT    NOT NULL,
    instance_id         TEXT,
    ssh_host            TEXT,
    remote_workdir      TEXT,
    error               TEXT,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    finished_at         TEXT
);
```

**表 2 `remote_job_event`：事件/结果桥梁（替代结果托管仓库）**

```sql
CREATE TABLE IF NOT EXISTS remote_job_event (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT    NOT NULL,
    event_seq   INTEGER NOT NULL,
    kind        TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    UNIQUE (job_id, event_seq)
);
```

- `seq` 全局自增，兼作 SSE 游标，查询对齐 `submission_service.events()`：`SELECT seq,payload FROM remote_job_event WHERE job_id=? AND seq>? ORDER BY seq LIMIT 200`。
- `UNIQUE (job_id, event_seq)` 在 SQLite 中自动建复合索引，同时保证 Runner 重连续传幂等；`event_seq` 由 Runner 分配 `[缺口]`，见 §14。
- `kind` 取值沿用引擎层：`meta / env_check / probe / case / warn / fatal / done / final`。
- `payload` 存 JSON 文本，不新增列即可携带逐 case 原始量与 artifacts 引用。
- `remote_job.status` 与 `submission_service` 九态的映射尚未定义 `[缺口]`。

### 3.6 与现有库表设计风格的对齐 `[代码核验]`

基准：`src/triton_eval/store.py`（`SCHEMA` / `SUBMISSION_RECORD_DDL` / `GROUP_SCORE_DDL`）与 `src/triton_eval/submission_service.py`（`service.sqlite` 的 `objects` / `events`）。

| 既有风格 | 本文两表如何对齐 |
|---|---|
| 主键一律 `id INTEGER PRIMARY KEY AUTOINCREMENT`；事件流特例用 `seq` | `remote_job.id` 用 `id`；`remote_job_event.seq` 对齐 `events.seq` |
| 业务唯一性另设 `UNIQUE(...)`，不靠主键 | `UNIQUE(job_id)`、`UNIQUE(job_id,event_seq)` |
| JSON 用 `*_json TEXT NOT NULL` 列 | `operators_json`；`remote_job_event.payload TEXT` 对齐 `events.payload` |
| 时间 `TEXT NOT NULL` + `utcnow()` 的 ISO 字符串（`store.py L153-154`） | `created_at / updated_at / finished_at` 同为 `TEXT` |
| 规避 SQL 保留字（`eval_case` 而非 `case`，`git_commit` 而非 `commit`） | 不使用 `case` / `commit` / `index` 作表名或列名 |
| 不用 SQLite 专有类型，便于迁 PostgreSQL | 仅 `INTEGER / TEXT / REAL` |
| `CREATE TABLE IF NOT EXISTS`，DDL 常量字符串，`create_schema()` 用 `executescript` 分次执行（隐式提交） | 建议新增 `REMOTE_JOB_DDL` 常量，按同样方式挂载 `[待实施]` |
| 查询加速层，真相源可重建 | `remote_job_event` 可由远端 `events.jsonl` / `results.jsonl` 完整重建 |

### 3.7 资产流：ZIP 与配套资产如何进入 NPU `[代码核验]` + `[设计]`

> **回答「是否仅通过表传递」：不是。** 两张表只承载控制/元数据；ZIP 字节、解压后的算子文件、corpus、评测引擎代码全部走**文件通道**（本机落盘 → `scp`/`tar` → 云端目录）。表与文件通过 `submission_sha256` / `corpus_fingerprint` / `remote_workdir` 关联。**文件通道的通道本体、时序位置、步骤序列与「尚未实现」边界见 §3.8**；本节只负责资产清单与校验链。

**（1）提交后本机发生了什么 `[代码核验]`**

`POST /api/v1/submissions` → `submission_service.upload()`（`L144-182`）：

| 步骤 | 事实 |
|---|---|
| 落盘 | `hashlib.sha256(data)` → `<root>/submissions/<uuid>/submission.zip`（`write_bytes`） |
| 解码 | 立即 `ingest_zip(package, <root>/submissions/<uuid>/ingest)`；`package.py L135-260` 自写解包（防 `../` 路径逃逸），内容寻址到 `ingest/<sha[:12]>/`，要求存在 `manifest.yaml` |
| 校验 | `validate_static(ingested.dest)` → `discover(...)` 得 `submit_id` / 算子名；算子必须属于 `manifest.operators`；结果 `status=READY\|REJECTED` |
| 入库 | `objects` 表存**元数据**（`package_sha256`、`operators`、`declared_submit_id`…），**不存 ZIP 字节、不存算子源码** |

即「解码出来的算子」是本机 `<root>/submissions/<id>/ingest/<sha12>/` 目录里的文件，**不是数据库行**。`<root>` 默认 `runs/submission-service`（`submission_http.py --root`）。

**（2）NPU 真正跑起来还需要什么 `[代码核验]`**

`submission_service._command(task,'run')`（`L372-382`）在当前实现里所有路径都指向本机：

```
python -m triton_eval --db <eval_dir>/engine.sqlite run \
  --submission <root>/submissions/<id>/submission.zip \
  --corpus <corpus> --runs <eval_dir>/runs --ingest-dir <eval_dir>/ingest --profile <profile>
```

`scripts/run_ascend_suite.py`（`L66-71`、`L93-98`）还会 `verify_corpus_files(corpus, manifest)` 与 `verify_assets(corpus, manifest.manifest_hash)`，并读取 `official-catalog.json` / `evaluation-assets.json` / `coverage-plan.json`。因此除 ZIP 外，必须同步到 NPU 的配套资产：

| 资产 | 本机位置 | 依据 |
|---|---|---|
| 提交 ZIP（或已解压的 `ingest/` 目录） | `<root>/submissions/<id>/` | `[实测·真机]` `scp` 传文件 `rc=0` |
| 冻结语料与哈希锚点 | `corpus/v0.9-official-candidates/`（`manifest.json`、`evaluation-assets.json`、`official-catalog.json`、`coverage.json`、`operators/`；实测 **2.3 MB**） | `[代码核验]` `run_ascend_suite.py` 强制校验，缺失即失败 |
| 评测引擎代码 | `src/triton_eval/`、`pyproject.toml`、`scripts/run_ascend_suite.py` | `[实测·真机]` 实例**不预置仓库** |
| 环境脚本 | `env/`、`scripts/env-bootstrap.sh`；NPU 已预置 `/home/developer/Ascend/cann-9.0.0/set_env.sh` | `[实测·真机]` `source` 成功、`npu-smi 25.5.0` 见 `910B3 OK` |
| Python 依赖 | torch / torch_npu / triton-ascend / venv | `[缺口]` **未实测**（见 §14.2 第 13 条） |

**（3）传输与校验链 `[设计]`**

```
本机 <root>/submissions/<id>/{submission.zip, ingest/}  ─┐
本机 corpus/v0.9-official-candidates/                   ├─ tar/scp ─▶ /mnt/workspace/jobs/<job_id>/
本机 src/ + scripts/ + pyproject.toml                   ─┘
   ↓ 远端 sha256sum 复核 ZIP
   ↓ verify_corpus_files / verify_assets；manifest_hash 与 catalog() 指纹一致
   ↓ ssh <host> 'bash -s' < run-job.sh
       → source /home/developer/Ascend/cann-9.0.0/set_env.sh
       → python -m triton_eval run / score / report
   ↓ 产物落同一 remote_workdir，由 §3.1 ⑧ 经 SSH 取回写 remote_job_event
```

- 表侧只写 `submission_sha256`、`corpus_fingerprint`、`operators_json`、`remote_workdir`，**不写任何字节**；这与 `objects` / `events` 只存 JSON payload 的风格一致（§3.6），也避免把 ≤64 MiB 的 ZIP 塞进 SQLite。
- 三种同步策略 `[设计]`：① 每次 Job 全量打包（最简单，corpus 2.3 MB + 引擎代码，成本可接受）；② **首次全量 + 后续按 `manifest_hash` 增量**（推荐，配合 `/mnt/workspace` 持久化）；③ 远端 `git clone` 代码仓（会重新引入 GitCode 依赖，但性质是**代码同步**而非控制信道/结果回传，可作备选）。
- 「在本机解压后传目录」还是「只传 ZIP、远端再 `ingest_zip`」尚未定 `[缺口]`：前者省一次解压但要传目录树；后者复用 `package.py` 的安全解包规则，一致性更好。

### 3.8 ⑤ 文件通道的定义 `[实测·真机]` + `[设计]` + `[缺口]`

> **本节是 §3.1 ⑤、§3.1 表侧说明、§3.7 开头、§6.1 反复引用的「文件通道」的正式定义。** 在此之前该词在全文只出现 3 次且均为指代，没有独立章节、没有可执行规格，导致无法定位。
>
> **定义**：⑤ 文件通道 = 本机 orchestrator 通过 `~/.atomgitdevenv/.ssh/config` 建立的 **SSH/SCP 会话**，把本机磁盘上的文件与目录复制到 NPU 实例的 `/mnt/workspace/jobs/<job_id>/`。它**不是** HTTP API、**不是** `remote_job` 表、**不是** GitCode 仓库。`cann_retry.API` 无 exec / 上传 / init 端点（§5.1），因此这是**目前唯一能把字节送进 NPU 的通道**。
>
> **状态一句话**：只有「单文件 SCP」被实测；**整包资产同步既未实现，也未实测**。

**（1）通道本体 `[实测·真机]`**

| 项 | 值 / 形态 | 证据 |
|---|---|---|
| 调用形式 | `ssh -F ~/.atomgitdevenv/.ssh/config <host> …`、`scp -F ~/.atomgitdevenv/.ssh/config -o BatchMode=yes …` | §5.5 C 段 |
| 认证与转发 | `IdentityFile ~/.atomgitdevenv/.ssh/IdentityFile/<id>`；`HostName 127.0.0.1` + `Port 41201` 转发到远端 `22` | §5.4 |
| 自动拉起 | `Match originalhost … !exec "<forward-bootstrap/vscode/<id>.sh>"`，每次 `ssh`/`scp` 自动复用或重建本机 hub，**无需 VS Code 远程窗口** | §5.4 |
| 唯一实测传输命令 | `scp -F ~/.atomgitdevenv/.ssh/config -o BatchMode=yes <local-file> <host>:/tmp/<file>`，远端回读 `rc=0`、`TE_SCP_OK` | §5.4 验证表、§5.5 C 段 |
| 目标盘 | `/mnt/workspace` 可写且持久化；**实例不预置仓库**，故引擎代码必须由本通道送入 | §5.4 验证表 |
| `<host>` 来源 | `[缺口]` 取自 `remote_job.ssh_host` 还是每次从插件 config 动态解析，未定；`serverAddr` 过期刷新路径见 §14.2 第 7 条 | — |

**（2）⑤ 在时序中的位置 `[设计]`**

```
④ POST /start → 等 RUNNING（17–43s）
  ⑤ 文件通道：建远端目录 → 同步资产 → 远端校验     ← 本节定义
  ⑥ ssh <host> 'bash -s' < run-job.sh
  … ⑦ 事件 / ⑧ 取回 / ⑨ SSE …
  ⑩ try/finally POST /stop
```

⑤ 必须在 ④ 之后（实例 `SHUTDOWN` 时 SSH 不通）、⑥ 之前（脚本、ZIP、corpus、引擎代码须先就位）。⑤ 失败与 ⑥ 失败一样落入 ⑩ 的 `finally`，避免实例空转烧卡时。

**（3）一次 ⑤ 的步骤序列 `[设计]`（均未落地）**

1. `ssh … 'mkdir -p /mnt/workspace/jobs/<job_id>/{submission,corpus,engine,env}'`
2. 同步资产：源清单见 §3.7(2)（ZIP / `ingest/`、`corpus/v0.9-official-candidates/`、`src/` + `pyproject.toml` + `scripts/`、`env/`）；承载形式拟为 `tar czf - … | ssh … 'tar xzf - -C …'` 或 `scp -r`，**三选一策略未定**（§3.7(3)、§14.2 第 8 条）
3. 远端 `sha256sum` 复核 ZIP == `remote_job.submission_sha256`
4. 远端跑 `verify_corpus_files(corpus, manifest)` / `verify_assets(corpus, manifest.manifest_hash)`；`corpus_fingerprint` 与 `manifest_hash` 的对齐规则未定（§14.2 第 14 条）
5. 全部通过才进入 ⑥；任一失败是重试同步还是直接 `FAILED` + `stop`，未定（§14.2 第 15 条）

**（4）明确「现在不存在」的东西 `[代码核验]`**

- 仓库 `src/`、`scripts/` 中检索 `scp|rsync|/mnt/workspace|orchestrator|run-job` **0 命中**：没有打包脚本、没有远端目录创建、没有源→目标路径映射、没有清理与磁盘上限逻辑。
- §6.1 提及的 `run-job.sh` 模板同样不存在（§10 已记「仓库当前不存在」）。
- 整包（corpus + 引擎代码 + 环境脚本）同步未实测；NPU 上 `python -m triton_eval` 能否运行未实测（§14.2 第 13 条，本轮实例 `SHUTDOWN`）。
- 因此 §3.1 ⑤ 那一行是**时序占位符**，不是已实现步骤；§14.1 原先勾的 ✅ 已改为「部分覆盖」——只表示「已判定不走表 + 已列资产清单 + 已实测单文件通道 + 已补 §3.8 定义」，**不表示同步逻辑已实现或整包已实测**。

**（5）交叉索引**：资产清单 §3.7(2)；校验链示意 §3.7(3)；通道命令与真机证据 §5.4、§5.5；未决项 §14.2 第 8 / 13 / 14 / 15 / 16 条。

---

## 4. 架构与组件边界（本机部署决定）

> **部署决定（2026-11-12）**：平台运行位置先放**本机**；平台后端**先不做公网部署**，在本机 loopback 试行；全程**只有 CANNLab 环境 NPU 实例运行在 GitCode 云端**，其余资产（平台后端、orchestrator、SQLite、SSH config/私钥/hub、前端静态资源、可选 GitCode 归档副本）尽可能放在本机做本地初步开发。

| 层 | 职责 | 运行位置 | 状态 |
|---|---|---|---|
| 浏览器 | 上传 ZIP、六态进度、逐 case 与分数展示 | **本机** | `[待实施]` |
| 平台后端 | 提交校验、DB 真相源、结果导入、SSE | **本机 `127.0.0.1:8765`** | 部分已有（`submission_http.py` / `submission_service.py`） |
| 两张中间表 | `remote_job` / `remote_job_event` | **本机 `service.sqlite`** | `[设计]` 未落地 |
| orchestrator | 领 Job、`POST /start`、SCP、SSH 取回、看门狗、`finally /stop` | **本机**（必须与 SSH 资产同机） | `[待实施]` |
| 本地 SSH 资产 | `~/.atomgitdevenv/` config / IdentityFile / bootstrap / hub | **本机** | 已生成（插件一次性登录产物） |
| CANNLab 控制面 | 实例启停 API、OAuth、connect-url / private-key | **GitCode 云端** | 外部，已实测 |
| NPU 实例 | 一次性 `run-job.sh`，无常驻进程 | **GitCode 云端** | 已实测可执行 |
| GitCode 证据通道 | 可选离线归档、`npu-results` | 云端 | 降级为可选，非主链路 |

**架构级约束**：`forward-bootstrap.sh` 依赖本机 VS Code 自带二进制 `Code Helper (Plugin)`（`runtimePath`）与 `hubScriptPath`，因此 orchestrator 必须与 `~/.atomgitdevenv/` 同机 —— 本机部署决定天然满足这一点，无需再拆分 worker 或抽取独立 hub 进程（v1 中"若平台后端放公网服务器"的迁移讨论已不适用，仅在将来公网化时重新评估）。

**本机部署的直接推论**：云端实例无法主动访问本机后端，结果必须经 SSH 通道由本机侧取回（§3.1 ⑧）。`submission_http.py` 的 Host 白名单与 `server_port` 绑定问题在本机同进程调用下**不构成阻塞**；将来若跨机/公网化才需要反向代理与鉴权。

---

## 5. 能力探测与实测证据

### 5.1 按需启停 `[实测·真机]`

- `~/cann-lab-http-retry/cann_retry.py` 的 `API` 只有状态 `GET`、`POST /start`、`POST /stop`、`/git-ssh-key`、`/git-token`；**无 exec / terminal / command / init 注入端点**（探测的 5 个端点全 404）。
- `prepare_webide()` 仅拼 Web IDE URL 并打开浏览器，不拉仓库、不执行评测。
- 实例：`ASCEND 1*NPU 910B3 / 16vCPUs / 32GiB`，镜像 `cann_9_0_0-py3.11-a2-arm-20260921`，`used_duration=54000`（15h）。
- 启动 17–43s，关停确认约 95s；多轮实验全部 `try/finally` 关停并确认 `SHUTDOWN`。
- `cann_retry.py` 内部的 `Halt('保留实例当前状态')` 分支在卡时敏感场景会白烧卡时，orchestrator **不得复用该异常语义**，必须自己 `finally`。

### 5.2 正确插件与错误路径 `[实测·本机]`

| 项 | 正确 | 此前误用 |
|---|---|---|
| 插件 | `atomgit-cloudenv.CANNLabDevSpace@1.1.2` | `HuaweiCloud.developerspace@1.6.0` |
| 账号 | GitCode / AtomGit OAuth（`dev.gitcode.com/api/v1/proxy/vscode`） | 华为 devstation / `ide_key` |
| URI | `vscode://atomgit-cloudenv.CANNLabDevSpace/connect?devstage_id=…&devenv_type=CONTAINER` | `vscode://HuaweiCloud.developerspace/connect?…&devenv_type=1` |
| 配置目录 | `~/.atomgitdevenv/` | `~/.devenv/` |

- 华为开发者空间控制台只列出一台鲲鹏 `8vCPUs/16GiB`、`root:/root`、状态异常的 `DEVSPACE_cw7lKl`，**不含 CANNLab NPU 实例** → `hdspace` CLI 路线排除。
- 华为插件 `/connect` 用本地列表 `An.find(devstage_id)`，列表来自 `clouddesktopso/devenvs`，找不到即 `envNotFound` 静默返回 —— 这解释了早前多轮 URI 触发「零反应、零日志」。

### 5.3 「VS Code 连接一直未成功」的真实原因 `[实测·本机]`

插件日志显示它**已成功**完成 `ensureHubRunning`、`forwardBootstrap.persist.done`、写 SSH config 与私钥；失败的是后续「打开 VS Code 远程窗口」：`remoteUnavailable`、`authenticateable.probe.failed`、`updateSshConfig.sshAuthenticationProbe.failed`、伴随 `400 ERR_BAD_REQUEST`。

**窗口坏了，但落地的 SSH 资产是好的。** 因此不需要修复 GUI 窗口，也不需要人工粘贴命令。

### 5.4 命令行 SSH 通道 `[实测·真机]`

插件写入 `~/.ssh/config` 的 `Include ~/.atomgitdevenv/.ssh/config`；专用配置形如：

```
Host <name>.<32-hex-id>.atomgit.0
  HostName 127.0.0.1
  User developer
  Port 41201
  IdentityFile ~/.atomgitdevenv/.ssh/IdentityFile/<id>
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  PreferredAuthentications publickey
  # RemotePort 22
Match originalhost … !exec "<forward-bootstrap/vscode/<id>.sh>"
    ProxyCommand false
```

`Match !exec` 让每次 `ssh` / `scp` 自动执行 bootstrap 脚本；脚本用 `ELECTRON_RUN_AS_NODE=1` 起独立 `Code Helper (Plugin)` 进程，复用/重建本地 hub（LOCK 文件给出 `PID|port`，实测 `8672|62682`），`bootstrap.log` 记录 `hub.reuse → forward.ready localPort=41201 remotePort=22`。CHANGELOG 1.0.9 明确该机制用于「VS Code 关闭后恢复端口转发」。

| 验证 | 结果 |
|---|---|
| `ssh -F <config> <host> '…'` | `rc=0`，`developer`，`/home/developer` |
| `ssh … 'bash -s' < 多行脚本` | `rc=0`，`source /home/developer/Ascend/cann-9.0.0/set_env.sh` 成功 |
| `npu-smi` | `25.5.0`，`3  910B3  OK`，HBM `65536MB` |
| `scp` 传文件 + 远端回读 | `rc=0`，`TE_SCP_OK` |
| 出网 | `gitcode.com` GET/POST 均 `200` |
| 工具链 | git `2.25.1`、Python `3.11.4`、CANN `9.0.0` |
| 目录 | `/mnt/workspace` 可写且持久化（`.arts / build / data / log`）；`/home/developer/Ascend` 预置；**实例不预置仓库** |

`npu-smi` 默认会因缺 `libc_sec.so` 报错，需 `source /home/developer/Ascend/cann-9.0.0/set_env.sh` 或显式设 `LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common`。`run-job.sh` 第一行必须加载环境。

### 5.5 已实测指令清单（可复现）`[实测·本机]` + `[实测·真机]`

以下命令均已在本会话执行过；`<host>` / `<id>` / `<job_id>` 为占位符，**不含任何凭据值**。证据日志见 §13。

**A. 实例状态与启停（本机 → GitCode 云端）**
```bash
python3 -c "import sys; sys.path.insert(0,'~/cann-lab-http-retry'); import cann_retry as c; a=c.API(c.DEFAULT_INSTANCE, auth_file='~/cann-lab-http-retry/.credentials.json'); s,_=a.get(); print(s.state,s.status)"
python3 -c "...; a.request('POST','/start')"      # 启动
python3 -c "...; a.request('POST','/stop')"       # 关停，实验内一律 try/finally
python3 -u /tmp/cann-probe.py                     # 4 次真实实例实验脚本（含 finally 关停）
python3 -u /tmp/cann-vscode-probe{,2,3}.py        # 3 次 VS Code URI 实验脚本
```

**B. 插件与 URI（本机）**
```bash
code --version                                    # 1.146.0
code --list-extensions --show-versions            # 含 atomgit-cloudenv.cannlabdevspace@1.1.2
code --install-extension atomgit-cloudenv.CANNLabDevSpace --force   # 提示已安装
code --install-extension /tmp/developerspace.vsix --force           # 误装的华为插件（已排除）
curl -fL -o /tmp/developerspace-1.6.0.vsix "<marketplace vspackage URL>" && python3 -c "import gzip; open(...).write(gzip.open(...).read())"
open -a "Visual Studio Code" "vscode://atomgit-cloudenv.CANNLabDevSpace/connect?devstage_id=<id>&devenv_type=CONTAINER"
code --open-url "vscode://..."                   # 三种投递方式均实测，远程窗口仍失败
open "vscode://..."
```

**C. SSH / SCP / 真机环境（本机 → 云端 NPU）**
```bash
ssh -F ~/.atomgitdevenv/.ssh/config -o BatchMode=yes -o ConnectTimeout=25 \
    -o StrictHostKeyChecking=no <host> 'echo TE_PROBE_OK; id -un; pwd; git --version; python3 --version'
ssh -F ~/.atomgitdevenv/.ssh/config <host> 'bash -s' < run-job.sh      # 多行脚本（实测 rc=0）
scp -F ~/.atomgitdevenv/.ssh/config -o BatchMode=yes <local-file> <host>:/tmp/<file>
ssh -F ~/.atomgitdevenv/.ssh/config <host> \
  'source /home/developer/Ascend/cann-9.0.0/set_env.sh; npu-smi info'  # 910B3 OK
ssh -F ~/.atomgitdevenv/.ssh/config <host> 'curl -sS -o /dev/null -w %{http_code} https://gitcode.com'
```

**D. 本机诊断**
```bash
cat ~/.atomgitdevenv/.hub/vscode/*.LOCK           # PID|port
lsof -nP -iTCP -sTCP:LISTEN | head -25            # 41201 转发端口 / 62682 hub
cat ~/.atomgitdevenv/.ssh/config
sqlite3 "file:$HOME/Library/Application Support/Code/User/globalStorage/state.vscdb?mode=ro" "select key,length(value) from ItemTable where lower(key) like '%devspace%';"
/usr/libexec/PlistBuddy -c "Print :CFBundleURLTypes" "/Applications/Visual Studio Code.app/Contents/Info.plist"
osascript -e 'tell application "System Events" to tell process "Code" to get name of every window'
```

**E. 只读代码/文档核验**
```bash
grep -n '^#' docs/*.md
grep -n -E "CREATE TABLE|PRIMARY KEY" src/triton_eval/store.py
grep -n -E "def request|BASE|/start|/stop" ~/cann-lab-http-retry/cann_retry.py
```

---

## 6. 契约与数据面 `[设计]`

### 6.1 参数载体（原 `job.json` 的替代）

参数改由 `remote_job` 表一行承载：`job_id`、`submission_id` + `submission_sha256`、`corpus_fingerprint`、`profile`、`operators_json`、超时预算、`instance_id` / `ssh_host` / `remote_workdir`。**禁止存 shell 命令，也禁止存任何文件字节或算子源码**；命令只存在于平台侧 `run-job.sh` 模板（`[缺口]` 该模板当前不存在，见 §3.8(4)），算子与配套资产走 **§3.8 定义的 ⑤ 文件通道**（资产清单见 §3.7），表只保留哈希与路径用于关联和校验。原 `job.json` 中的 `callback_url` 在本机部署下无意义（云端无法回连本机），已移除 `[缺口]`：若将来公网化再补。

### 6.2 双事件层与幂等 `[代码核验]` → `[设计]`

| 层 | 内容 | 位置 |
|---|---|---|
| 引擎级 | `meta / env_check / probe / case / warn / fatal / done`，逐 case append+flush | 远端 `run_dir/events.jsonl`（`evalchild.py`） |
| 服务级 | `QUEUED / RUNNING / SCORING / REPORTING / COMPLETED / …` 状态转换 + `next_cursor` | 本机 `service.sqlite` `events` 表，`GET /events?after=`（`LIMIT 200`） |
| 桥梁（v2 新增） | 上述两者的统一承载 | 本机 `remote_job_event`（`seq` 作游标，`event_seq` 作幂等键） |

- 引擎 `events.jsonl` 中**逐 case 无 score 字段**，只有 `status` 与性能原始量；分数是 run 级 `RunScoreRecord`，由 `score` 阶段生成。
- 平台按 `(job_id, event_seq)` 去重；Runner 重连按 `after` 续传。
- 「实时出分」定义：逐 case 推事件 → `score` 完成即推 `final.json` → 报告/artifacts 随后取回。

### 6.3 前端状态机 `[设计]`

`QUEUED → 申请资源中 → 环境启动中 → 执行中 → 上传结果 → 释放环境 → COMPLETED`，另设 `FAILED / CANCELLED / INTERRUPTED` 分支。必须展示「环境启动中」等耗时态，不得承诺秒级实时。

### 6.4 结果契约

`results.jsonl` 是可移植结果契约（`cli.py import --run --jsonl` 可离线导回）；`final.json` 承载总分与终态；artifacts 沿用固定白名单 + SHA-256 校验。所有结果先落远端持久目录 `/mnt/workspace/jobs/<job_id>/`，再由 orchestrator 取回写本机表。

---

## 7. 失败恢复矩阵 `[设计]`

| 故障 | 检测 | 处理 | 实例 |
|---|---|---|---|
| `POST /start` 失败 / 启动超时 | 状态轮询 + 启动超时预算 | `remote_job.status=FAILED`，alarm，不重试烧卡时 | 必须 `stop` |
| SSH 注入失败 / bootstrap 失败 | `ssh … 'echo OK'` 预检 | 重试上限内重连；超限触发重新 OAuth 刷新资产 | 必须 `stop` |
| 执行中 SSH 断连 | 心跳 + `event_seq` 续传 | Runner 侧继续写远端 `events.jsonl`，orchestrator 重连后按 seq 补读 | 看门狗超时 `stop` |
| SIGKILL / 进程被强杀（无 trap） | 远端文件 + 心跳证据 | 从远端持久目录 / `results.jsonl` 补读导入 | 立即 `stop` |
| `final.json` 丢失 | 无 final 超时 | 以已收 case 事件 + `import` 回放结算 | 立即 `stop` |
| `serverAddr` 过期 | 预检失败 | 触发插件刷新 / 重新 OAuth；仍失败则告警人工 | 立即 `stop` |
| 本机平台后端不可用 | 本机进程探活 | orchestrator 先把事件落本机文件，恢复后写表；**不依赖云端回连** | 不阻塞 `stop` |
| 并发/多 Job | `flock` 单实例 | 本机单 Worker 串行领 `QUEUED`，同一时刻只一个实例 RUNNING `[缺口]` | — |
| Worker 中断 | `worker-paused.txt` | **不可直接复用现有人工闸门语义**（见 §9.1） | 立即 `stop` |

> v1 中「并发发布冲突（force-with-lease）」已从主链路移除：GitCode 归档降级为可选，不再是实时结果通道。

---

## 8. 卡时经济 `[设计]`

- 实例平时 `SHUTDOWN`；只有提交后领取 Job 才启动。
- `final` 到达 / 无事件超时 → 立即 `stop`；不等待报告取回完成。
- 排队、解析、静态校验、结果导入都不占用实例。
- 批次合并：同一次实例 RUNNING 期间可合并同 profile 的多个 Job，但需保证 `job_id` 级隔离与证据可分辨。
- 实例 RUNNING 预算：`Σ(case.timeout_s + 180) + 600` 为服务层既有预算，orchestrator 需另设「实例总 RUNNING 上限」，到点无条件 `stop`。
- 前端展示额度与预计消耗；不可承诺秒级实时。

---

## 9. 必须处理的既有约束 `[代码核验]`

1. **`worker-paused.txt` 人工闸门与无人值守冲突**：`submission_service.py` 在任何取消/超时/异常/重启后写暂停标记并要求人工确认，重启把 ACTIVE 标为 `INTERRUPTED` 且不自动重跑。设计上平台后端只复用其提交解析与持久化，真机执行交给独立 orchestrator；若复用 worker，必须改造恢复策略。
2. **网络边界（本机试行口径）**：`submission_http.py` 绑 `127.0.0.1:8765`，`Host` 必须恰为 `localhost:<server_port>` 或 `127.0.0.1:<server_port>`，零鉴权，自述「非安全沙箱」。**当前决定本机部署，该限制恰好满足，暂不需要反向代理 / 鉴权 / CORS**；仅将来公网化时才需处理。
3. **设备健康未接入调度**：`device_health='NOT_PROBED'`、三项认证恒 `NOT_COMPUTED`、`full_suite_accepted=False`。需把真机 `npu-smi` / `probe` 事件接入调度准入。
4. **双事件层**：`/events?after=` 只有任务级状态、无逐 case 事件，且是轮询不是流；需新增 SSE 与引擎事件桥接。
5. **发布并发（已降级）**：`publish_ascend_results.py` 的单分支 + `--force-with-lease` + `trim_records(keep=3)` 冲突问题仍存在，但只在**可选离线归档**路径上出现；主链路改走本机表后不再阻塞。
6. **目录持久化（约束变更）**：2026-11-12 确认「真机每两小时自动清空目录」机制**已取消**，现行环境不会自动清空，`/mnt/workspace`、`/home` 与 SQLite 锚点跨轮保持。因此不再按两小时窗口组织流程，`standard` 档全量测速不必另开窗口。SIGKILL 下 trap 仍不可捕，现由持久本地记录、平台 SSH 结果回收及租约 watchdog 处理。2026-11-15 已删除旧式 Git 心跳上传和跨清空周期的 wheel 备份脚本，移除验证脚本的强制 clone/reset/clean；手动发布走外部 Git 认证。平台事件心跳与自动关停仍保留，它们服务于任务存活与资源释放，不依赖两小时清空假设。
7. **凭据与安全**：GitCode OAuth token 曾以明文出现在 `git remote`；需轮换、改为环境变量/凭据文件并扫描历史。第三方算子在真机执行仍需隔离（`submission_service.py` 明确不是安全沙箱）。
8. **Workspace Trust**：插件 README 说明未信任工作区会延迟写 SSH config 与启动本地 helper，首次需人工授权一次。

---

## 10. 卡点与待实施清单

### 10.1 无阻塞性能力卡点

§2 的 5 步链路，能力全部经实测或现有代码覆盖；原断裂点（提交后自动启动 NPU 并无人值守执行命令）已由 `POST /start` + `ssh -F ~/.atomgitdevenv/.ssh/config` 打通。

### 10.2 工程缺口 `[待实施]`

| 项 | 说明 |
|---|---|
| 两张表 DDL 落地 | `REMOTE_JOB_DDL` 常量 + `create_schema()` 挂载点（对齐 `store.py` 风格） |
| SSE | `text/event-stream` 接口；现无 Web 框架，需新增 handler |
| 结果取回 / 导入 | orchestrator 从 SSH 通道读回 JSONL 写 `remote_job_event`；现仅本机 `cli.py import` |
| 前端状态机 | 六态 + 逐 case/分数展示；仓库当前无前端源码 |
| orchestrator / run-job.sh | 领 Job、启停、SCP、SSH、看门狗、`finally stop`；仓库当前不存在 |
| 双事件层桥接 | `events.jsonl` → DB/SSE，统一 `event_seq` 幂等 |
| 设备健康接入 | `npu-smi` / `probe` 作为调度准入 |
| ~~公网鉴权 / 反向代理~~ | **按本机部署决定，暂不需要**；仅在将来公网化时重新评估 |

### 10.3 残留链路风险（唯一需运行期兜底项）

`forward-bootstrap/.../<id>.json` 的 `serverAddr` 是约 183 字符的 `wss://` 地址，可能携带临时凭据。若过期，bootstrap 无法建隧道，SSH 通道失效。对策：每次 Job 前 `ssh … 'echo OK'` 预检；失败则触发插件刷新/重新 OAuth 并告警；不可静默重试烧卡时。该风险本次未复现，属可工程化处理项，不是能力阻塞。

---

## 11. 与现有资产对照 `[代码核验]`

**可直接复用**
- `src/triton_eval/submission_http.py`：10 个路由、OpenAPI、ZIP 上传、幂等键、产物 SHA-256 白名单下载。
- `src/triton_eval/submission_service.py`：状态机、SQLite、`events()/next_cursor`、`_command()` 构造 `run --submission … --corpus … --profile …`。
- `src/triton_eval/store.py`：**库表设计风格基准**（见 §3.6）。
- `src/triton_eval/cli.py`：`ingest / run / score / report / import / rescore / compare`；`results.jsonl` 契约。
- `src/triton_eval/supervisor.py`、`evalchild.py`：逐事件硬超时、append+flush 事件文件、抗 SIGKILL。
- `scripts/run_ascend_suite.py`：`--submission/--corpus/--profile/--out` 真机入口。
- `scripts/submission-simulation-verify.sh` + `submission_simulate.py`：完整仿真闭环，可拆成平台侧与 Runner 侧两端复用。
- `scripts/upload-heartbeat.sh` / `publish_ascend_results.py`：心跳与可选归档兜底。

**需改造**
- `submission_http.py`：SSE、结果导入接口（**绑定/鉴权/CORS 按本机决定暂不改造**）。
- `submission_service.py`：`worker-paused` 恢复语义、设备健康、事件桥接。
- `publish_ascend_results.py`：仅在保留可选归档时需要并发安全。

**需新建（本文不落地）**
- `orchestrator.py`、`scripts/run-job.sh`、两张表 DDL、SSE handler、前端状态机。

---

## 12. HTTP 请求目标地址清单 `[代码核验]` + `[实测]`

> **所有地址均不含凭据值。** 认证头只列字段名。

### 12.1 平台后端（本机 loopback，v2 主链路）

基址 `http://127.0.0.1:8765`；`Host` 必须为 `127.0.0.1:8765` 或 `localhost:8765`，否则 403 `HOST`；`Origin` 仅接受空或 `http://<host>`。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/openapi.json`、`/api/v1/catalog` | 契约与目录 |
| POST | `/api/v1/submissions` | ZIP 原始二进制上传（`application/zip`，`Idempotency-Key`） |
| POST | `/api/v1/evaluations` | 建评测任务（JSON，202） |
| GET | `/api/v1/submissions/{id}` | 提交详情 |
| GET | `/api/v1/evaluations/{id}` | 任务详情 |
| GET | `/api/v1/evaluations/{id}/results` | 结果 |
| GET | `/api/v1/evaluations/{id}/events?after=<cursor>` | 状态事件轮询（`LIMIT 200`，非 SSE） |
| GET | `/api/v1/evaluations/{id}/artifacts`、`/artifacts/{name}` | 产物清单与白名单下载 |
| POST | `/api/v1/evaluations/{id}/cancel` | 取消 |
| GET | `/api/v1/jobs/{id}/stream` `[待实施]` | v2 拟议 SSE 端点（路径未定，`[缺口]`） |

ID 约束 `^[0-9a-f]{32}$`；请求体上限 64 MiB；socket 30s。

### 12.2 CANNLab 实例启停（GitCode 云端，`cann_retry.API`）

基址 `https://web-api.gitcode.com/score-proxy/api/v1/shop/third-party/resources/<instance>`，`<instance>` 实测为 32 位 hex；**所有请求带 `?type=cann`**。

| 方法 | 完整路径 | 结果 |
|---|---|---|
| GET | `…/resources/<instance>?type=cann` | 状态/资源快照（`vscode_jump_url` 等） |
| POST | `…/resources/<instance>/start?type=cann` | 启动（实测 17–43s 后 RUNNING） |
| POST | `…/resources/<instance>/stop?type=cann` | 关停（约 95s 确认） |
| POST | `…/resources/<instance>/git-ssh-key?type=cann` | 返回 `public_key` 等，仅 git 用途 |
| POST | `…/resources/<instance>/git-token?type=cann` | git 凭据准备（`prepare_webide` 用） |
| POST | `https://web-api.gitcode.com/uc/api/v1/user/token/refresh` | OAuth token 续期 |

请求头字段：`Authorization: Bearer <access_token>`、`Origin: https://gitcode.com`、`Referer: https://gitcode.com/`、`x-platform: web`、`x-internal-source: gitcode-ssr-fe`、`User-Agent: CANNLabRetry/1.0`。**探测为 404 的端点**：`/exec`、`/terminal`、`/command`、`/actions`、`/status`（即无任何命令注入 API）。

### 12.3 CANNLab Dev Space 插件（GitCode 云端，仅首次登录/刷新资产时用）

基址 `https://dev.gitcode.com/api/v1/proxy/vscode` + `/DevContainerService/open-api-inner/v1/devenv`。

| 用途 | 路径 |
|---|---|
| 环境列表 | `GET /instances?devenv_type=<…>` |
| 单环境 | `GET /instances/{id}` |
| 隧道地址 | `GET /instances/connect-urls/{id}?source=vscode` |
| SSH 私钥 | `GET /instances/private-keys/{id}?source=vscode` |
| 认证头 | `access-token: <oauth_access_token>` |

OAuth 相关端点：`https://gitcode.com/oauth/authorize`、`https://web-api.gitcode.com/score-proxy/api/v1/oauth/token`、`https://api.gitcode.com/api/v5/user`。网页正规入口 `https://gitcode.com/org/cann/cannlab/environment` 生成的跳转 URI 为 `vscode://atomgit-cloudenv.CANNLabDevSpace/connect?devstage_id=…&devenv_type=CONTAINER`。

### 12.4 已排除路径（华为 devstation，记录以免误用）

| 端点 | 裸调结果 |
|---|---|
| `https://devstation-svc.connect.huaweicloud.com/ide/index/islogin` | `400 invalid or expired ide_key` |
| `…/open-web-guest/v1/clients/version-requirements?type=vscode&current_version=1.6.0` | `200`，`minimum_version=1.3.0`（版本门禁通过） |
| `…/userApi/v1/clouddesktopso/devenvs/{id}/connect-url`、`private-key` | `403 not login` |

控制台入口（仅核对资源用）：`https://developer.huaweicloud.com/space/home`、`https://developer.huaweicloud.com/space/devportal/platform/devEnvironment?tab=virtualMachine`。

### 12.5 本机 hub 与远端出网

| 目标 | 说明 |
|---|---|
| `http://127.0.0.1:<LOCK 内端口>/create-forward-hub` | 建立转发通道；需 `devenvId` + `serverAddr` |
| `…/start-forward` | 实测**未先 create** 直接调用返回 `ForwardHub not found, and creation not allowed` |
| `…/stop-forward`、`/delete-forward-hub`、`/heartbeat`、`/clear`、`/bootstrap-forward-requests` | 插件内部端点 |
| LOCK 文件 `~/.atomgitdevenv/.hub/vscode/1.1.2.LOCK` | 实测内容 `8672\|62682`（PID\|hub 端口）；SSH 转发端口 `41201` |
| `https://gitcode.com` | 云端 NPU 侧 `curl` GET/POST 实测均 `200`（仅证明出网可达，v2 主链路不依赖它回传） |

---

## 13. 资料地图与开工索引

> 供另一 Agent 开工时按图索骥；**先看本文 → 再看 `submission-simulation.md` → 再看代码与实验日志**。

### 13.1 校验过程中沉淀落盘的 MD 文件

| 文件 | 用途 | 状态 |
|---|---|---|
| `docs/remote-npu-evaluation-architecture.md` | **本文**：链路总纲、两表设计、HTTP 地址、实测指令、缺口清单、资产对照 | v2，现行 |
| `docs/cannbench-platform-reference.md` | CANNBench 平台形态/站点地图/提交规则/额度与限流文案；含 3.6 节「站内文档与 UI 口径差异」的文档漂移教训（规则单一事实来源） | 现行，配套前端参考 |
| `docs/submission-simulation.md` | 现有 HTTP 契约（10 路由、九态状态链、产物白名单、SHA-256 校验）、仿真闭环、`--acknowledge-isolation` 语义 | 现行，后端契约基准 |
| `docs/评测体系要求.md` | 需求落点「面向持续运行的评测平台与工程资产建设」 | 现行 |
| `docs/operator-evaluation-hardening-plan.md` | 评测加固计划 | 现行 |
| `docs/requirements-coverage-report.md` | 覆盖报告；含历史「两小时窗口工作法」口径（机制已取消，属历史记录，未改写） | 历史口径 |
| `docs/remote-verification-playbook.md` | 原「昇腾机闭环验证手册」，按两小时清空硬约束组织全文 | **已删除**（2026-11-12，机制取消后整份删除；git 已跟踪，可恢复） |

### 13.2 代码与库表风格基准

`src/triton_eval/store.py`（`SCHEMA` / `*_DDL` / `utcnow()`）、`submission_service.py`（`service.sqlite` 的 `objects`/`events`、`events()/next_cursor`、`_command()`）、`submission_http.py`（路由）、`cli.py`、`evalchild.py`、`supervisor.py`、`scripts/run_ascend_suite.py`、`scripts/submission_simulate.py`、`scripts/publish_ascend_results.py`。

### 13.3 实验证据（`/tmp`，不随仓库分发）

| 文件 | 用途 |
|---|---|
| `/tmp/cann-probe{,-result.json,-.log}.py/json/log` 及 `probe2/3/4` | 4 次真实实例实验：API 字段、exec 端点探测、Web IDE 注入、容器内探针、`finally` 关停 |
| `/tmp/cann-vscode-probe{,2,3}.py` 及同名 `.log` / `-result.json` | 3 次 VS Code URI 实验：三种投递方式、hub 信号采集、登录态判定 |
| `/tmp/developerspace*.vsix`、`/tmp/devspace-ext/` | 误装华为插件的 VSIX 与解包目录（已排除路径，仅留证） |
| `/tmp/te_scp_probe.txt` | `scp` 验证用临时文件，已删除 |

### 13.4 外部资产（本机，不随仓库分发）

`~/cann-lab-http-retry/cann_retry.py` + `.credentials.json`、`~/.vscode/extensions/atomgit-cloudenv.cannlabdevspace-1.1.2/`（含 `README.md` / `CHANGELOG.md` / `dist/src/extension.js` / `helper/forward-bootstrap.js`）、`~/.atomgitdevenv/.ssh/config`、`~/.atomgitdevenv/.ssh/IdentityFile/<id>`、`~/.atomgitdevenv/forward-bootstrap/vscode/<id>.{json,sh}`、`~/.atomgitdevenv/.hub/vscode/*.LOCK`、`~/.atomgitdevenv/logs/vscode/latest.log`、`~/.ssh/config` 的 `### ATOMGIT DEVENV CONFIGURATION START` Include 区块。

---

## 14. 覆盖度自查与缺口清单

### 14.1 逐项确认

| 用户要求 / 链路环节 | 是否覆盖 | 位置 |
|---|---|---|
| 原始 5 步链路逐环节判定 | ✅ | §2 |
| 两个 GitCode 仓库流程是否过重的评估 | ✅ | §3.4 |
| 两张轻量级 DB 表替代方案 | ✅（草案，未落地） | §3.5 |
| ZIP / 解码算子 / 配套资产如何进入 NPU（表是否传字节） | **部分覆盖**：已判定不走表、已列资产清单、已补通道定义；整包同步未实现、未实测 | §3.1 ⑤、§3.7（清单）、§3.8（通道定义）、§6.1、§14.2 第 8 条 |
| ⑤ 文件通道是否有独立定义与规格 | 已补（此前该词只被引用 3 次、无定义章节） | §3.8 |
| 与现有库表风格对齐 | ✅ | §3.6 |
| 各 HTTP 请求目标地址 | ✅ | §12 |
| 已实测指令清单 | ✅ | §5.5 |
| 落盘 MD 文件位置与用途 | ✅ | §13.1 |
| 本机部署 / 不做公网 / 只有 NPU 在云端 | ✅ | §4、§9.2、§10.2 |
| 按需启停与 `finally` 关停证据 | ✅ | §5.1 |
| SSH/SCP/真机 NPU/出网实测 | ✅ | §5.4 |
| 正确插件与错误路径纠正 | ✅ | §5.2、§5.3 |
| 失败恢复、卡时经济、既有约束 | ✅ | §7、§8、§9 |
| 工程缺口与残留风险 | ✅ | §10、§14.2 |

### 14.2 具体缺口 `[缺口]`（开工前必须确认）

1. **云端 → 本机回传方向**：v2 已改为 orchestrator 经 SSH 主动取回（§3.1 ⑧），但**具体形态未定**：是 `ssh … 'bash -s'` 的 stdout JSONL 流式读，还是周期性 `scp`/`cat` 拉取远端 `events.jsonl`。两者对"实时"与断连补读的实现差异较大。
2. **Runner 事件输出格式**：`run-job.sh` 输出 JSONL 的字段、`kind` 枚举、`final.json` 结构、artifacts 引用方式未定义。
3. **SSE 接口契约**：路径（§12.1 仅为拟议）、`Last-Event-ID`、心跳帧、重连策略、`LIMIT` 与游标语义未定义。
4. **两张表 DDL 尚未落地**：`REMOTE_JOB_DDL` 常量、`create_schema()` 挂载点、迁移方式未定；本文只有设计草案。
5. **状态机映射**：`remote_job.status` 与 `submission_service` 九态、前端六态之间的转换表未定义。
6. **`event_seq` 分配方**：由 Runner 递增还是平台分配，重连时如何避免重复/跳号未定。
7. **`serverAddr` 刷新路径**：只写了"触发插件刷新 / 重新 OAuth"，**具体命令、API、触发条件未定义**；插件会话有效期也未实测（本次未复现过期）。
8. **资产同步策略未定**：全量打包 / 首次全量 + `manifest_hash` 增量 / 远端 `git clone` 三选一；解压位置（本机 `ingest/` 目录直接传 vs 只传 ZIP、远端再 `ingest_zip`）；`--submission` / `--corpus` / `--runs` / `--ingest-dir` 在远端的路径映射；`/mnt/workspace/jobs/<job_id>/` 的清理策略、磁盘上限、停实例后补取报告的时限，均未定。通道本体与步骤序列见 §3.8，资产清单见 §3.7。
9. **并发与排队**：本机 `flock` 单实例 + 单 NPU 实例下的多 Job 排队、优先级、批次合并规则未定。
10. **前端**：仓库无前端源码，SSE 消费、六态渲染、逐 case/分数展示均未设计。
11. **设备健康接入**：`device_health='NOT_PROBED'` 到调度准入的判定规则未定。
12. **将来公网化**：跨机/公网部署时的鉴权、反向代理、Host 白名单、hub 独立化路径**未设计**（当前决定不做，仅记录）。
13. **NPU Python 依赖未实测**：torch / torch_npu / triton-ascend / venv 是否齐备、`python -m triton_eval` 能否在真机运行，均未验证（本轮实例为 `SHUTDOWN`，未额外开机）。实例 RUNNING 时的零卡时验证命令：`ssh -F ~/.atomgitdevenv/.ssh/config <host> 'python3 -c "import importlib.util as u; print({m: bool(u.find_spec(m)) for m in [\"torch\",\"torch_npu\",\"triton\",\"triton_eval\",\"yaml\"]})"'`，另需 `ls /mnt/workspace` 确认是否已有仓库。
14. **corpus 版本一致性**：`remote_job.corpus_fingerprint` 与远端 `verify_assets` 的 `manifest_hash` 如何对齐、corpus 升级时如何防止旧 Job 用错语料，未定。
15. **资产校验失败的处理**：远端 `sha256sum` 复核 / `verify_corpus_files` / `verify_assets` 失败时是重试同步、还是直接 `FAILED` 并 `stop`，未定。
16. **`submission_service` 复用边界**：现有 `_command()` 用 `sys.executable -m triton_eval` 且路径全在本机；搬到 NPU 后是复用该构造函数（替换路径前缀）还是由 `run-job.sh` 自行拼命令，未定。

---

## 15. 附录：关键路径与参考

**配套文档**：`docs/cannbench-platform-reference.md`（前端参考）、`docs/submission-simulation.md`（HTTP 契约与仿真闭环）、`docs/operator-evaluation-hardening-plan.md`、`docs/评测体系要求.md`。`docs/remote-verification-playbook.md` 已删除，见 §13.1。

**本文不记录**：GitCode OAuth token、SSH 私钥内容、`serverAddr` 的完整值、任何 WSS 凭据；仅记录路径、字段名与形态。
