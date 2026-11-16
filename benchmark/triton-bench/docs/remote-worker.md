# 远端评测 Worker

2026-11-13。实现位置：`src/triton_eval/orchestrator.py`、`remote_jobs.py`、`remote_bundle.py`、`remote_runner.py`、`remote_watchdog.py`，入口为既有 `submission_http.py`。沿用提交检查、九态任务、评分和报告；不修改冻结语料、容差、参考实现或评分公式。

本机 hub 空闲回收的根因和本轮加固见 [SSH 稳定性复盘](ssh-stability-review-20261113.md)。当前 Worker 独立持有每 5 秒的 HTTP 心跳租约，采用任务私有 SSH 复用会话与脱敏诊断；不依赖 VS Code 窗口持续发心跳。原始记录中的 187/41 项是前一检查点，本次新增专项为 196 项后端检查、45 项外部脚本检查。

页面结构、现有 API 映射、同源代理约束和后续联调验收见 [前端与接口联调方案](frontend-platform-plan.md)；当前生成契约及无 Worker 响应样例见 [前端联调契约包](frontend-integration/README.md)。

首版页面与浏览器真机验收现已完成，见 [前端实施记录](frontend-implementation-20261113.md)；启动方式见 [前端 README](../frontend/README.md)。相关后端回归扩展至 220 项通过。

## 启动

在已有 Python 3.11+ 评测环境中，复制 `env/remote-worker.example.json` 到不提交的本机配置，例如 `runs/remote-worker.json`，填写实例 ID 和 `~/.atomgitdevenv/.ssh/config` 中对应的 Host。配置由服务运营者管理，HTTP 请求不能覆盖 SSH、Python、路径或命令。凭据文件仅保存绝对路径，实际凭据由既有 cann_retry 客户端读取/续期，不进入同步包。

```sh
python -m triton_eval.submission_http \
  --root runs/remote-platform \
  --corpus corpus/v0.18-scan-certification-candidates \
  --remote-config runs/remote-worker.json \
  --acknowledge-isolation
```

`--execute` 仍启用旧本机 Worker，与 `--remote-config` 互斥。不传两者则仅提供上传/排队/查询。首次启用远端模式会将服务库中尚未领取的 QUEUED 任务绑定到远端队列，不重复创建任务。

本机必须具备有效的 CANNLab Dev Space OAuth 和受信任 workspace。已补全外部 `~/cann-lab-http-retry/cann_retry.py`：`--run --connect-vscode` 在启动后准备 git-ssh-key/git-token，POST 本机 hub 的 start-forward，必要时 create-forward-hub/bootstrap-forward 并等待插件刷新；实际核验 SSH banner 和认证命令后才认定连接就绪。`start.sh` 默认追加 `--open-vscode` 打开工作区；Worker 直接复用相同函数，不打开窗口、不要求人工点击。健康连接复用，不强制重建。

脚本原版备份在 `~/cann-lab-http-retry/backup-20261113-vscode/`，心跳加固前版本备份在 `backup-20261113-stability/`。原 34 项加 7 项连接测试、4 项租约测试共 45 项通过，checksums.json 已更新；凭据和 SSH 配置未改写。脚本仍依赖首次插件配置及有效 OAuth，不能替代插件登录；通道失败会在预算内明确退出。外部启动工具单独运行时保留实例状态，平台 Worker 才负责 finally/watchdog 关停。不要据此推导任意无登录/无插件进程的冷恢复必然成功。

保持 loopback Host/Origin 边界。前端需同源部署或使用保留后端 Host 的开发代理。

## 执行与完成条件

1. SQLite 同一事务创建 task 和 remote_job；原子 FIFO 领取，同一服务工作目录有进程锁，同机不同服务目录也通过实例级 flock 排他。
2. 在本机校验冻结 corpus 和判定资产、原始 ZIP 指纹；按明确清单复制完整引擎与依赖到 staging，再计算逐文件指纹和最终 bundle SHA-256。排除 Git 历史、运行目录、OAuth 和 SSH 资产；即使选择 v0.18，也保留冻结依赖所需的 v0.9 manifest。
3. 确认实例初态为 SHUTDOWN。若已被占用则暂停派发，既不启动也不停止它。租约绑定 instance，配置变化不能误停另一个实例。
4. 在请求 start 前持久化 start_intent，并先启动独立本机看门狗。实例启动后，经 SSH/SCP 上传，在任务独立目录核验外层包 SHA，再解包。
5. 经远端 Python subprocess.Popen 显式分离会话和标准流，启动 `scripts/run-job.sh`；避免后台 shell 的 AND-list 持有 SSH 通道导致“已启动却等待到超时”。脚本 source CANN、选择显式 Python，调用一次性 Runner。Runner 逐文件复核 bundle、manifest_hash、assets_hash、ZIP；实际导入 torch/torch_npu/triton CANN 扩展，并执行 NPU 小探针。通过后才创建远端本机 EvaluationService 执行 run/score/report。
6. 专用 `remote-events.jsonl` 写入带版本、job_id 和持久递增 event_seq 的 envelope。引擎 events.jsonl 作为 `kind=engine` 的 payload 回传；stdout/stderr 仅作诊断。正常执行每十秒写心跳。
7. 本机定期从已提交字节偏移取回完整行。事件入库、派生状态与偏移推进同事务；半行不消费；重放相同事件幂等，同序号异内容、序号跳跃、错误 job_id、无效 case、阶段倒退和超大行均拒绝。final 后不能追加事件。
8. final 到达后，在实例仍运行时取回包含 engine.sqlite 身份元数据、JSONL、报告和日志的结果包。解包拒绝链接、目录逃逸、重复成员、大小超限和指纹不符；逐文件 fsync 后原子安装到本机任务目录。再核对原始任务、包身份、语料、档位、实际 case 集合、评分和必需报告。
9. finally 请求 stop，等待 SHUTDOWN。只有产物已核验且关停已确认才允许 COMPLETED。错误算法可以流程 COMPLETED，但功能分为 0；COMPLETED 不等于全部 PASS。stop 未确认单独记录 UNCONFIRMED，保留租约并阻止新派发。

远端 Runner 会拒绝重跑已开始但未完成的目录。服务重启对遗留 ACTIVE 租约先释放实例，再标记 INTERRUPTED；不会自动重新执行提交。运行中取消与预算超限也进入释放路径。旧本机 Worker 的人工暂停语义保留。显式 recover 支持 FAILED/INTERRUPTED、同一实例已确认 SHUTDOWN、结果包尚未安装的任务；重用原 job_id/remote_workdir/字节偏移，只启动资源读取旧日志和结果，不调用同步或 launch。已安装结果目录不会被覆盖；失败历史和旧事件保留。SSH 读取断连时，会有界调用外部脚本的 HTTP reconnect，再从提交偏移补读。

独立看门狗继承实例 flock，在主服务被 SIGKILL 后仍保有实例排他权；检测所有者退出或实例预算耗尽后直接使用 cann_retry 请求 stop，写 `watchdog-result.json`。它与主进程 finally 双重兜底，不能覆盖整台本机断电、系统休眠或凭据撤销；云端若需覆盖这些情况，仍需独立于本机的关停机制。

## 查询、实时事件和恢复

新增接口：

| 方法 | 路径 | 口径 |
|---|---|---|
| GET | `/api/v1/evaluations?before=<rowid>&limit=30` | 按创建顺序倒序分页；limit 1–100，next_cursor=null 表示本页为空 |
| GET | `/api/v1/evaluations/{id}/stream` | 同源 SSE；状态、阶段、健康、引擎逐 case、心跳和 final |
| POST | `/api/v1/evaluations/{id}/recover` | JSON `{"acknowledge_recovery":true}`；仅恢复已关停的原任务事件和结果，不重新上传/执行提交 |
| POST | `/api/v1/worker/resume` | JSON `{"acknowledge_health":true}`；存在 ACTIVE/未释放租约时 409，清除人工暂停标记 |

SSE 帧使用 `id: <events 全局 sequence>`、`event: update`、单行 JSON data；首次给出 `retry: 2000`，空闲每 15 秒发注释心跳。`Last-Event-ID` 优先于 `?after=`，非负 64 位游标；批次 200 条只限制单次查库，不能当作流结束。最多 8 条连接，每条最多 5 分钟后客户端按游标重连。终态排空后发送 `event: end`，客户端应关闭 EventSource，避免终态重复重连。其他任务造成的全局序号间隙不是丢事件。

既有 `/events?after=` 现在也能读取远端 envelope；其旧状态字段保留。`task.remote` 给出 phase、release_state、artifacts_verified、device_health 和三项指纹。`completed_cases` 和 `/results` 在结果包取回前从已持久化 case 事件展示，得分仍等 COMPLETED 后才发布。认证保持 NOT_COMPUTED、full_suite_accepted=false，smoke 不提供性能分。

`GET /api/v1/catalog` 返回 execution_backend=disabled/local/remote。完整契约见 `/openapi.json`。

数据库迁移使用 `PRAGMA user_version=1`，在原 objects/events 同库添加 remote_job/remote_job_event；未知较高版本拒绝启动。原始事件的唯一键为 (job_id,event_seq)，SSE 的 seq 引用同库 events，因此状态与逐 case 不需要两套客户端游标。

本机证据目录为 `<root>/remote/<job_id>/`：bundle-index.json、bundle.tar.gz、result.tar.gz、lease.json、watchdog-config.json 和看门狗结果。配置包含本机凭据文件的路径，应作为本机运维资产保管，不能公开归档。正常完成后可以从 `<root>/evaluations/<task_id>/engine.sqlite` 与 results.jsonl 重算；只有裸 JSONL 不包含完整运行身份。

## 验证与边界

`tests/test_remote_platform.py` 覆盖原子队列、取消、事件部分行与重放、SSE 超过 200 条排空/续读、结果取回前可见用例、真实 SQLite 评分/报告、多个故障的 finally、拒停他人实例、遗留租约恢复和实际 SIGKILL 独立看门狗。设备结果是明确的 CPU 测试夹具，不能冒充真机成绩。

本轮记录位于 `runs/platform-implementation-20261113/`。全量 L0 的三个 `codecheck_revision` 缺失 migration_metadata.py 的已有失败仍单列；没有为消除它们改写冻结资产。开发过程中对新 NPU Runner 补充了 L1 导入点契约。

目前交付远端后端和前端可消费的接口；尚无可交互前端、正式榜单/跨提交性能锚点、用户团队配额、编译器上传安装、远端历史目录自动清理。结果包上限默认 256 MiB，实例预算默认 1800 秒；这不构成第三方代码的安全沙箱或整个远端磁盘配额。标准档与全量 1385 case 仍须单独验收。


## 2026-11-13 实施验收记录

- 既有后端加新平台专项（含最终恢复入口）：187 passed，真实 loopback HTTP/SSE 无跳过；之后释放加固 18 项、整包/独立报告 2 项、最终 Runner/报告 2 项定向检查均通过。独立进程 SIGKILL 关停和真实描述符分离均有测试。
- 本轮整套 L0（恢复入口加入前）：1410 passed、3 failed、1 skipped、445 deselected，另 21 subtests passed。三项失败仍是预先确认的 codecheck_revision 缺失 migration_metadata.py，未新增其他失败。唯一 skip 是当前已有 torch，不适用无 torch 测试。
- 本机外部启动工具：原有 34 项加新增 7 项，41 项通过，已安装文件和 checksums.json 全部一致；可用 `start.sh` 完成启动、HTTP 连接核验并打开 VS Code。打开界面的动作由 CLI 发起，验收使用无窗口模式核验真实 SSH，不将界面是否渲染当作评测证据。
- 独立 bundle 的子进程可生成 MD/HTML；离线构建 wheel 成功，确认两个 Jinja 模板和远端模块都在 wheel 中。冻结判定资产未改写。

本轮失败也完整留存，不能只看最后一次成功：首轮暴露 HTTP 连接准备遗漏；第二轮暴露后台 shell 持有 SSH 描述符；随后将 VS Code POST/认证核验正式写入外部脚本并处理短暂重连；第四轮 20 PASS，但报告模板漏包导致 ENGINE-REPORT，未发布成功分数；第五轮 19 PASS + 1 ORACLE_FAIL，报“参考实现修改了输入”，用例为 `official-abs-float16-28-normal-c`，任务按 EVALUATOR-FAILURE 失败、未发布分数，已回收 25 个产物且哈希全符，最终 SHUTDOWN。该参考输入一致性异常的根因尚未查明；不通过改变 v0.18 冻结输入、容差或判定来掩盖它，后续独立跟踪。

证据根目录：`runs/platform-implementation-20261113/`；`verification-summary.json` 汇总各项结果，`implementation-source-sha256.json` 记录实现指纹，`attempt-5/wheel-verification.json` 记录分发包验证。真机完整结果以具体 attempt 的 `real-npu-result.json` 为准；旧 attempt 的失败不能用新结果覆盖。


最后一次原任务的 SSH 读取中断后，使用正式 recover 接口补取，未同步代码、未重新调用 Runner。已从原 job 取回 20 PASS、MD/HTML 报告和原始运行数据库，全部身份/范围/评分/产物检查通过。该恢复同时覆盖专用事件文件的迟到补读与 SSE 游标续读；最终 `recovery-6/real-npu-result.json` 已确认 success=true、任务 COMPLETED、实例 SHUTDOWN；29 个 HTTP 下载产物的大小/SHA-256 全符，20 条 SSE case 事件断线续读无重复，两项专项功能分均为 100%，smoke 的 perf_score=null。结果取回完成后才确认关停，不用阶段性日志代替释放确认。187 项后端专项是在恢复入口、HTTP reconnect 和新看门狗就绪检查加入后运行的最终检查。

## 2026-11-15：外部凭据为远端启动硬要求

CANNLab 启停凭据不放进仓库。当前本机使用仓库外的 `~/cann-lab-http-retry/.credentials.json`，配置仅存路径。远端平台在打开 HTTP 监听、创建服务目录及领取任务之前，检查凭据为当前用户所有的普通文件、权限恰为 `600`、不为符号链接且实际位置在仓库外，JSON 包含有效格式的 `access_token` / `refresh_token`。缺失或不符合要求立即拒绝启动，错误不会回显凭据。实际创建云客户端时再次检查；续期仍由已有外部脚本负责。

路径优先级：`--credentials-file /外部路径/oauth.json` > 环境变量 `CANNLAB_CREDENTIALS_FILE` > 远端配置 `credentials` > 既有用户目录默认路径。不要把令牌作为命令行参数或环境变量值传入；这些入口仅接受文件路径。先在外部工具完成登录并保存凭据，再使用 `chmod 600` 设置权限。

此检查只读取本地文件，不会启动实例或验证在线登录状态；过期且无法续期、被撤销的凭据仍会在云请求阶段报错。只读联调模式无需云凭据；不能凭此运行远端任务。原来的两小时清空相关自动克隆、强制重置、定时推送和编译器轮子备份已移除，任务超时及 finally/watchdog 关停继续保留。
