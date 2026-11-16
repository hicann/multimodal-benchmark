# 前端首版实施与联调验收

日期：2026-11-13。设计依据为 [前端方案](frontend-platform-plan.md)，用户选择了第二张展示的视觉稿。已实现真实后端驱动的本机 Web 工作台，未发布公网服务。

## 实施范围

`frontend/` 使用 React、TypeScript、Vite、SWR、React Router 与 Phosphor 图标，版本由 package-lock.json 固定。界面沿用浅暖色侧栏、橙色主操作和纵向分步流程；实际数据从 `/api/v1` 获取，不以设计示意数据作为运行成绩。

| 页面 | 已实现 |
| --- | --- |
| 概览 | 当前 corpus、真实算子/用例数、最近任务及统计范围说明 |
| 评测集 | 服务绑定版本、manifest_hash、算子签名/参数/权重/用例数、搜索 |
| 提交 | 原始 ZIP 上传进度、READY/REJECTED 分流、逐规则错误、算子选择、档位与范围、双阶段幂等、会话内恢复已上传提交 |
| 任务列表 | 游标分页、已加载数据内搜索/状态筛选、状态和用例进度、详情跳转 |
| 任务详情 | 阶段/进度/资源释放、SSE 历史与续读、断线轮询、取消、服务端资格约束的原任务补取、逐 case 搜索/详情、评分与产物下载 |
| 文档 | API 生成的上传限制、包级规则、示例 ZIP、评分与恢复说明、显式健康确认的本机运维入口 |

只有已声明算子能进入专项选择；默认档位与范围由 catalog 下发。未启用/暂停 Worker 时仍能上传校验，开始评测按钮不可用。COMPLETED、case PASS、评分可用、资源 SHUTDOWN 分别表达，未发布分数和没有性能分均不会显示为 0。

## 后端与入口补齐

- `submission_policy.py` 从实际 ZIP 校验常量生成公开规则，HTTP 请求上限也复用同一常量。catalog 增加 submission_policy、profile_capabilities、default_profile/default_scope；standard 为试运行，不宣称性能/全量已验收。
- `task.available_actions` 给出 cancel/recover 的 enabled/reason；恢复资格与 POST recover 共用同一服务端检查，不允许页面自行拼凑隐藏条件。
- OpenAPI 0.2.0 补齐 state/engine/phase/health/heartbeat/final 结构，逐 case 引用 CaseResult 模型；任务状态和旧接口保持兼容。
- `frontend_gateway.py` 仅绑定 loopback，固定目标为本机后端端口；先校验入口 Host/Origin，再映射后端头。支持静态 SPA 路由刷新、ZIP 字节与幂等头、SSE 即时转发、附件下载；拒绝路径逃逸、隐藏文件和符号链接。
- Vite 开发入口同样仅绑定 loopback，按相同原则校验 API 请求后代理；未知 API 命名空间返回 JSON 404，不回退为 HTML。

运行方式见 [frontend/README.md](../frontend/README.md)。凭据、SSH、云启停与 hub 心跳始终由后端管理；前端没有这些资产。

## 自动检查

- 相关后端回归：**220 passed，无跳过**。包含真实 loopback HTTP/SSE、取消/恢复资格、代理安全边界、静态文件边界、结果下载及既有远端 CPU 夹具回归。
- 前端事件客户端：**6 passed**。验证全局游标与重放去重、精度保护、SSE 失败降级、终态关闭、卸载中止，以及传输层忽略 abort 时不接受迟到回放。
- OpenAPI 导出 `--check`、生成 TypeScript `api:check`、严格类型检查及生产构建通过。
- Vite 实际代理检查覆盖同源接受、异源/错误 Host 拒绝、OpenAPI、ZIP/幂等、201+REJECTED、SSE 首帧即时到达、取消及终态排空。测试强制确认后端 Worker 为 disabled，不执行设备任务。

证据根目录：`runs/frontend-foundation-20261113/`，后端 XML 为 tests.xml，开发代理检查为 vite-proxy-checks.json。此处“后端 220 项”是本次相关回归，不替代全仓 L0 的单独结果。

## 浏览器真机端到端

IAB 中实际执行：选择 `public/examples/abs-smoke.zip` → 校验通过 → 点击“创建评测任务” → 自动跳到详情 → 在 RUNNING 阶段刷新 → 继续读取同一任务 → 查看最终分数 → 点击下载 HTML 报告。任务创建由浏览器完成，验收脚本没有代替页面创建任务。

| 项目 | 结果 |
| --- | --- |
| task_id | `b341987d818c441f98df828ef7f2781e` |
| 范围 | v0.18 冻结 Abs 专项、smoke、20 case |
| 用例与评分 | 20 PASS；功能主分 100；严格通过率 100%；静默错误 0 |
| 性能口径 | perf_score=null，页面显示“暂无”，并说明 smoke 不提供性能分 |
| 执行次数 | launch_count=1；页面刷新没有新建或重跑任务 |
| SSH | 31 条命令，0 失败，1 个私有会话 |
| hub 保活 | 成功 37 次，启动前 4 次未就绪，末次无错误；成功心跳最大间隔 5.009 秒，hub 未切换 |
| 产物 | 29 项经前端入口下载，大小/SHA-256 全符 |
| 浏览器报告下载 | `/Users/mac/Downloads/report.html`，13026 字节；SHA-256 与服务端元数据一致 |
| 资源释放 | 任务 COMPLETED、产物已核验、实例已确认 SHUTDOWN；最终云状态 stopped |

完整结果为 browser-remote-result.json，真机服务库为 browser-remote-service/；SSH 与连接证据位于其 remote/<job_id>/。验收使用 0600 临时凭据副本并在 finally 清理，不将凭据或 Git 历史传到 NPU。整个验收有时限及 finally/watchdog 释放保护。

另在独立的 Worker disabled 服务中创建一个不执行的队列任务，通过页面取消，观察到 CANCELLED 和“已同步”；没有为取消交互额外运行 NPU。

## 视觉与浏览器检查

浏览器检查了标题/页面身份、非空白、无框架错误遮罩、控制台、实际交互和截图；桌面 CSS viewport 为 1440×1024、窄屏为 390×844。截图与源稿合成对照后，将高级设置移出主配置段，修复确认按钮位置偏低；收紧窄屏导航，保持五个入口均可见。完整视觉结论见 [design-qa.md](../frontend/design-qa.md)。

浏览器原始证据位于 `/tmp/triton-eval-ui-20261113/`，验收收尾时另归档到 runs 证据目录。截图来自实际渲染与操作，不是 Image Gen 设计稿。

会话中断后 IAB 控制服务不可用，用户明确授权改用独立的本机 Playwright/Chrome 做最后复核。最终 **17 项检查通过**，桌面1440×1024、窄屏390×844；控制台错误、页面运行时错误和外部请求均为0。再次确认已修正的窄屏五项导航全部可见、任务页无页面级横向溢出、原任务刷新与报告下载正确，并且没有创建新的评测任务。此轮仅使用 disabled 后端，没有补跑 NPU。最终记录为 browser-evidence/final-browser-checks.json，视觉检查结论为 passed。

## 当前运行与边界

真机验收结束后，结果库以 **Worker disabled** 模式重新开放，前端入口 `127.0.0.1:5174` 可查看本次成绩与下载报告；上传校验仍可用。再次执行真实任务需按 README 启动带 remote-config 的后端，当前预览不会自动续开云实例。

未完成项仍是标准性能和全量 1385 case 验收、正式跨提交榜单、用户/团队/配额、编译器构建安装及公网部署。此前一次 ORACLE_FAIL 的根因仍待单独排查；本轮该 case PASS 不等于已经修复其根因。旧 codecheck_revision 三项缺失 migration_metadata.py 的全仓问题也没有在本轮处理。
