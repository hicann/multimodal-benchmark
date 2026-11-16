# Triton Eval 前端

用户选择了第二版视觉稿：浅暖色侧栏、橙色操作按钮、纵向分步提交。当前是连接真实 Python 后端的 Web 工作台，含概览、算子目录、提交、任务列表、任务详情、榜单与使用指南；所有成绩均来自真实结果，没有模拟账号或分数。

## 本机开发

在已有项目 Python 环境中启动一个独立联调库，默认不运行评测：

```sh
PYTHONPATH=src python -m triton_eval.submission_http \
  --root runs/frontend-dev \
  --corpus corpus/v0.19-workload-completion-candidates --port 8765
```

另一个终端从 `frontend/` 启动开发服务器：

```sh
npm ci
npm run dev -- --port 5173 --strictPort
```

访问开发服务器显示的 `127.0.0.1` 地址。Vite 仅绑定 loopback，通过已校验 Host/Origin 的同源代理访问后端；SSE 保持流式传输，不开放全部 CORS。可用服务端环境变量 `TRITON_EVAL_BACKEND_PORT=8766` 指定另一个本机后端端口，该值不进入浏览器配置。

## 构建与本机预览

```sh
cd frontend
npm run build
cd ..
PYTHONPATH=src python -m triton_eval.frontend_gateway \
  --static-dir frontend/dist/client --backend-port 8765 --port 5173
```

Python 入口提供构建静态文件、页面路由刷新及 API/SSE 转发。仅构建目录可以作为 static-dir；不要指向仓库、运行目录或用户目录。浏览器关闭不会停止后端任务。

启用真实远端执行时，使用 [远端 Worker 配置](../docs/remote-worker.md)启动后端，前端不变：

```sh
PYTHONPATH=src python -m triton_eval.submission_http \
  --root runs/frontend-remote \
  --corpus corpus/v0.19-workload-completion-candidates \
  --remote-config runs/remote-worker.json --acknowledge-isolation --port 8765
```

真实执行库与开发校验库必须分开；远端模式会接管同库遗留 QUEUED 任务。凭据与 SSH 配置只在 Python 后端读取，不能放入 frontend、public 或 VITE_* 变量。当前页面不是多用户公网服务；模板的 Sites 静态 Worker 不包含评测后端，本轮没有部署网站。

## 契约与检查

从仓库根目录更新后端快照，再在前端生成类型：

```sh
PYTHONPATH=src python scripts/platform_examples/export_frontend_contract.py
cd frontend
npm run api:types
npm run api:check
npm run typecheck
npm test
npm run build
```

生成类型位于 `src/api/schema.d.ts`，不要手改。运行事件模块独立处理全局游标、重复事件、断线轮询及页面卸载；它不会创建或重跑任务。运行中的任务、分数和产物以服务端快照为准。

`public/examples/abs-smoke.zip` 是已用于真机 smoke 的纯源码示例，只有 manifest.yaml 和 operators/user_impl.py。`public/assets/mark.png` 是从选定概念稿生成的透明标识素材；页面图标使用 Phosphor。页面字体使用本机中文字体回退，不依赖外部字体请求。

## 本轮验收

2026-11-13：相关后端 220 项通过；前端事件客户端 6 项通过；类型/构建/契约同步检查通过。IAB 浏览器验证了上传与拒收、选择联动、搜索、队列取消、刷新原任务、真实 NPU 运行与报告下载。真机 Abs smoke 为 20 PASS、功能主分 100、严格通过率 100%，perf_score=null；29 项下载大小与哈希全符，launch_count=1，最终 SHUTDOWN。没有因此完成标准性能、全量 1385 用例或公开排行榜验收。

完整验收见 [前端实施记录](../docs/frontend-implementation-20261113.md)，视觉检查见 [design-qa.md](design-qa.md)。真机验收结束后，结果库以 Worker 禁用模式保留供预览；需显式使用远端配置启动后端才会再次执行任务。

收尾时会话浏览器服务不可用，经用户授权改用独立 Playwright/Chrome 完成最终17项桌面/窄屏检查，控制台和外部请求均为0。该轮没有创建新评测任务或启动 NPU；最终截图与联合对照位于 `runs/frontend-foundation-20261113/browser-evidence/`。

## 2026-11-13 标准性能与交互优化

标准 Abs 专项 20/20 PASS，实测几何平均加速比 0.848267×。性能 100 分为任务内首全对基准，不能跨任务排名。详见 [验收与诊断](../docs/standard-performance-review-20261113.md)。结果 API 新增 performance 统计，页面以实测速度优先并标记测速波动。

## 2026-11-14 半量标准实验与试运行榜单

v19 保留旧版全部1385例，新增101例至1486例，6个新增算子各20例、四档齐备。最终单任务37算子、746例标准评测全部PASS，功能100，实测几何平均加速比0.823829×，25例波动超阈值，755文件大小/哈希全部一致，实例确认关停。见 [实验与修复记录](../docs/half-standard-experiment-20261114.md)。

`/leaderboard` 接入真实API，按语料、判定资产、范围、模式、环境、评分器及协议分组，持久保存每算子的共享基准；支持功能/性能切换、同ZIP去重、并列名次、基准追溯和结果报告跳转。当前真实组37个算子有33个可计相对分，首个共享基准分100；任务原有自基准分保持不变。手机端以成绩卡片呈现，规则与环境在成绩之后展开。

此前结果预览（现已开放执行，见下一节）为 `http://127.0.0.1:5177/leaderboard`，使用 `runs/half-standard-final-20261113/half-final-service`、后端8772、Worker禁用；该地址仅在本机服务运行时有效。试运行榜单不代表正式编译器对照、多用户运营或全量1486例认证。非NPU回归1720通过、3项已知缺失migration_metadata失败；前端客户端6通过，Chrome/Playwright桌面1440×1000与手机390×844共8组检查通过，无运行时错误或外站请求。

## 2026-11-14 实际评测运行与多算子下载

当前5177平台已经开启真实远端Worker（8772后端，原结果库），不再是禁用执行的预览；提交任务后才按需启动NPU，任务结束自动回收并关停。当前服务配置在 `runs/evaluation-access-20261114/remote-worker.json`，未把凭据放入前端。

提交页与指南提供推荐的 [37算子包](public/examples/v19-half-37.zip)（746例，当前共享源码生成，静态预检通过；真机成绩以实际运行结果为准）及 [74算子包](public/examples/v19-full-74.zip)（1486例对应源码，静态检查通过、未做本轮全量真机验收）。Abs仅作可选最小格式示例。`PYTHONPATH=src .venv/pattern-build/bin/python scripts/platform_examples/build_platform_examples.py` 在根目录确定性重建包及下载哈希元数据。上传后可展开选择算子、全选/清空并运行功能或性能评测。

之前“Worker禁用”的文字只描述此前结果预览阶段；当前状态以API和本节为准。[开放记录及三类问题解释](../docs/evaluation-access-20261114.md)。

开放后的最终非NPU全仓复查1723通过、0失败、21跳过、167排除，另21subtests通过；原3项资产修订工具失败已修复，未伪造缺失模块。37/74包真实下载SHA、上传READY、标准范围与可用创建按钮的Playwright检查通过；本轮网页验收未创建NPU任务。


## 已修复基线下的构建

构建前会用当前受审查的共享源码重新生成下载包，不将旧 ZIP 的真机成绩套用于新包。请激活 Python 3.11+ 项目环境，或通过 `TRITON_EVAL_PYTHON=/绝对路径/python` 指定解释器。ZIP 与包指纹清单是构建产物，不纳入源码提交。
