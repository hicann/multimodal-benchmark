# v0.7 修订与 NPU 验证指令

本文保留 v0.7 历史验证流程。新版统一来源候选及一键运行方式见
[官方来源审计与 v0.8 候选集](official-ascend-expansion.md)。复现旧冻结版本须使用其记录的代码提交。

本次复用已有覆盖引导、参考门禁、隔离监督、评分和报告机制，补齐 v0.6 的接入与实现缺口。
新语料为 `corpus/v0.7-ascend-contract`：11 个有效算子，每个 20 例，共 220 例。
原 v0.6 语料及 `reference/triton-ascend-ops` 教程源码保留；另四个未接入主题的真实功能及处置见新版来源审计，不能统一称为无效算子。

## 修改范围与评测含义

| 层面 | 本次落地 |
|---|---|
| 输入契约 | 检查 dtype、设备、连续性、索引界限、置换、唯一写入行、专家边界、卷积状态映射、KV/head/split 关系 |
| 参考 | 11 个纯 torch 主参考及各自独立 CPU 第二参考；所有输出检查确定性、输入不变性、元数、dtype、数值 |
| 生成 | 从 324 个候选调用现有 `guided_selection`，保留 dtype/规模及非整块边界；冻结前参考失败则整批拒绝，保留生成日志 |
| 判定 | 各输出分别使用其 dtype 的现有容差，并检查设备；整型精确比较，避免 int64 转浮点丢失差异；NaN/Inf 值不一致记数值错误 |
| 示范实现 | 修正 slice 行步长/边界、卷积时间窗口/状态/通道尾块、注意力 KV 路由与 fp32 累加、归一化三个输出、padding 语义 |
| 能力 | 原始 `tl.gather` float32/int32 独立探针；适配 DiscreteMemoryAccess 使用原生整数指针读取，不经过浮点转换 |
| 完整运行 | 每算子一个正式 CLI 运行，均有监督与健康探针；保留各 run_id，逐分区审计；一个算子失败后仍执行其余算子 |
| 追溯 | 冻结用例清单和参考/契约/比较器等资产指纹；提交源码单独指纹；结果、日志、环境、评分及报告直接保存在本轮目录 |

`COMPILE_FAIL` 表示失败阶段。确认是编译器缺陷还需要合法程序和最小复现证据。
适配实现的通过不能证明原 `tl.gather(int32)` 已受支持，能力探针失败不会被隐藏或折算成通过。
评分器标识升为 `0.2.4`，对应逐输出与整型比较修正；未放宽容差、未更改评分公式。
新旧语料的内容和参考语义不同，分数不能直接比较。新版本的正式性能锚点需要在真机重新建立。

## 真机完整指令

### 推荐：一条命令运行并自动上传

首次获取本次更新，在已有真机工程目录执行：

```bash
cd "$HOME/triton-eval-v07"
git pull --ff-only origin main
bash scripts/ascend-suite-verify.sh --profile smoke --operators CausalConv1d
```

去掉 `--operators CausalConv1d` 即运行完整 220 例。此后每轮只需最后一条命令。
入口自动激活原 Ascend 环境；环境启动失败、0 用例、编译失败等也会上传终端日志和退出码。
看到 `[upload] 已上传 GitCode npu-results` 后，只需通知助手“跑完了”，助手即可拉取结果。
该提示仅表示上传成功，是否通过仍由 `evidence/suite-summary.json` 决定。

凭据依次使用 `TRITON_EVAL_PUSH_URL`、原 origin/`GITCODE_TOKEN`，或现有旧上传脚本中
与 origin 仓库匹配的配置；不新增明文令牌，不修改工程的 remote、分支或暂存区。
上传有超时，失败退出码为 2，保留本地原件并打印补传命令；评测自身的退出码记录在
`execution.json`，包含实际代码提交号。无需重跑即可补传：

```bash
python3 scripts/publish_ascend_results.py runs/ascend-suite/实际时间戳
```

结果统一发布到独立的 `npu-results` 分支，`rolling-results.json` 指向最新记录，
`records/<时间戳>/` 包含终端输出、评测结果、环境、能力探针、逐例日志及报告。
数据库不上传。单轮超过 50 MiB 则上传失败并保留原件，不静默裁掉诊断证据。
远端及本地已上传记录默认只保留最近 3 轮；未上传的本地记录不会自动删除。
结果分支每次发布无父提交快照，以带租约的强制更新替换，避免旧日志在该分支历史中持续累积；
并发更新冲突会明确失败，可按提示补传。代码 `main` 的历史不改写。
已有历史归档仅保留最近 3 轮的 `.log` 文件，较早的结构化结果、环境及报告仍保留。
清理旧文件不会消除已有 Git 历史对象；GitCode 服务端回收策略决定实际对象回收时间。

本入口在结束时上传；若环境直接 SIGKILL 或断电，退出钩子无法运行，需对仍存在的目录执行补传。
以下是助手侧获取最新证据的命令，不需要在真机反复复制日志：

```bash
git fetch --depth=1 origin +refs/heads/npu-results:refs/remotes/origin/npu-results
git show origin/npu-results:rolling-results.json
```

### 手工运行与归档

在 **Linux 昇腾 NPU** 主机的 Bash 中执行，先从 GitCode 获取当前代码。
使用独立目录，避免覆盖此前运行的代码和结果；已有目录只允许快进更新，冲突时停止。

```bash
TASK_REPO="$HOME/triton-eval-v07"
if [ -d "$TASK_REPO/.git" ]; then
    git -C "$TASK_REPO" pull --ff-only origin main
else
    git clone --branch main --single-branch \
      https://gitcode.com/hkhkhkzxk/triton-eval.git "$TASK_REPO"
fi
cd "$TASK_REPO"
git log -1 --oneline
```

私有仓库按 GitCode 提示使用自己的凭据认证。不要把访问令牌写进命令或日志。
以下命令接着在上述工程目录执行。
使用已有 `torch_npu` 和 **triton-ascend** 环境，Python 至少 3.11。
若环境不在自动发现位置，先设置 `TRITON_EVAL_VENV=/实际/venv/路径`。

```bash
bash
set -eo pipefail

export ASCEND_VISIBLE_DEVICES=0
source scripts/env-bootstrap.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python -m pip install -e . --no-deps --no-build-isolation
python -m pip install 'pydantic>=2.7' 'PyYAML>=6' 'Jinja2>=3.1' 'pytest>=8' 'numpy<2'

python -c 'import sys, torch, torch_npu, triton; import triton.language.extra.cann.extension; print(sys.version); print("torch:", torch.__version__); print("triton:", triton.__version__, triton.__file__); print("NPU:", torch.npu.device_count()); assert sys.version_info >= (3,11); assert torch.npu.is_available()'
npu-smi info

python -m pytest -q tests/test_comparison.py tests/test_ascend_contracts.py tests/test_ascend_suite.py tests/test_judge.py tests/test_output_contract.py tests/test_run_completeness.py tests/test_contract_guards.py tests/test_integrity.py

python scripts/build_corpus_v06_ascend_ops.py

TASK_STAMP="$(date -u +%Y%m%d-%H%M%S)"
TASK_OUT="$PWD/verify-records/ascend-contract-smoke-${TASK_STAMP}"
TASK_RC=0
python scripts/run_ascend_suite.py --profile smoke --out "$TASK_OUT" || TASK_RC=$?
cat "$TASK_OUT/suite-summary.json"
tar -czf "${TASK_OUT}.tar.gz" -C "$(dirname "$TASK_OUT")" "$(basename "$TASK_OUT")"
echo "suite exit code: $TASK_RC"
echo "archive: ${TASK_OUT}.tar.gz"
```

不要安装社区版 `triton`，它与 triton-ascend 共用 Python 导入目录。
新入口不进行 git 同步、reset、commit 或上传；归档完成后可按已有渠道传回压缩包。
旧 `ascend-ops-verify.sh` 保留自动同步和上传行为，本轮建议使用上面的新入口以运行当前代码。

`run_ascend_suite.py` 自动完成以下步骤，无需再手工执行：

1. 校验冻结资产，并记录提交源码指纹。
2. 全 220 例 CPU 主参考/第二参考自检，再运行 NPU 主参考/CPU 第二参考自检。
3. 独立子进程运行健康、float32 gather、int32 gather 探针，保留各自日志及真实返回码。
4. 每算子运行正式 `triton_eval run`，完成其全部 20 例；运行日志随时写盘。
5. 对各原始 run_id 执行 `score` 和 `report`；分别检查漏例、重复、额外记录和终态。
6. 写入 `suite-summary.json`，完整且全部 PASS、命令正常返回时退出码才为 0。

`complete=true` 仅表示期望用例均有唯一终态；`all_pass=true` 才表示数值和契约全部通过。
参考或环境门禁失败时停止正式测评，原因写入 `gate_failure`，不能解释为用户算子得零分。
某算子运行失败时，其余算子仍执行；设备上下文故障后的剩余记录由已有监督流程补记，保留失败证据。
各算子采用不同 run_id，**不要把多个运行拼接后当成同一个 run 重评分**。

本机 CPU 参考通过不代表 NPU 兼容性通过。新内核的编译、精度和性能结论以真机日志为准。
原有回归测试中固定为 25 个注册算子、只检查 v0.5 参考的断言同步更新为当前 36 个算子，
分别从对应版本清单核验参考路径；旧语料没有因此被改写。

## 性能测试：另开环境窗口

先确认 smoke 的 `all_pass=true`，再在环境稳定、无其他负载的窗口执行。
完整集性能测量可能超过单窗口，可用 `--operators CausalConv1d,DecodeGroupedAttention` 等按算子分窗。
每个选中算子仍跑完整 20 例，不能用 `--limit` 的小样本建立正式锚点。

```bash
TASK_PERF_OUT="$PWD/verify-records/ascend-contract-standard-$(date -u +%Y%m%d-%H%M%S)"
TASK_PERF_RC=0
python scripts/run_ascend_suite.py --profile standard --out "$TASK_PERF_OUT" || TASK_PERF_RC=$?
cat "$TASK_PERF_OUT/suite-summary.json"
tar -czf "${TASK_PERF_OUT}.tar.gz" -C "$(dirname "$TASK_PERF_OUT")" "$(basename "$TASK_PERF_OUT")"
echo "performance suite exit code: $TASK_PERF_RC"
```

同一次 suite 内的算子共享该目录的独立数据库；不同 suite 不会默认合并性能锚点。
跨窗口的性能汇总应复用既有锚点/对比流程，并检查设备和环境指纹，不能平均这些报告中的总分。

## 定位单算子或检查能力

首轮真机运行的 220 例均有终态，200 例 PASS，卷积 20 例 COMPILE_FAIL。
其中已提取的 bf16 失败日志显示：卷积累加与状态拷贝复用 `value`，
Triton 分支合并时检测到 fp32 与 bf16 类型冲突。这是示范内核的局部变量类型问题。
修复为状态拷贝单独使用 `state_value`，保留原始状态 dtype；用例、参考、容差及评分不变。
真实 NPU 回归扩展为全部 20 个 bf16 卷积用例，覆盖时间窗口及通道尾块。
修复后的编译与数值结果仍须在真机重新确认；既有结果目录保留。

```bash
python scripts/run_ascend_suite.py --profile smoke --operators CausalConv1d,DecodeGroupedAttention
python scripts/probe_ascend_capabilities.py --out "runs/capabilities-$(date -u +%Y%m%d-%H%M%S)"
python -m pytest -q tests/test_ascend_npu.py -k causalconv1d
```

能力探针缺省只要求健康探针通过；需要所有探针均通过时加 `--require-pass`。
整型 gather 不支持时的失败日志与适配实现结果分别保留。

## 外部自定义提交

已注册类型继续使用原提交协议，入口参数、dtype、形状和语义遵循评测方契约。
替换 `--submission` 即可评测外部实现；只有实现部分算子时显式选择类型：

```bash
python scripts/run_ascend_suite.py --submission /实际/用户提交目录 --operators VectorAdd,CausalConv1d --profile smoke
```

任意新类型仍需评测方注册规格、参考、输入构造、容差及测试集，不能让用户提交的参考自行决定得分。
本轮连续内存及索引/状态关系是这 11 个类型的明确边界；非连续、空张量、允许非有限值等新语义需要单独登记。
当前门禁默认拒绝空或非有限的参考输出。修订生成集没有宣称覆盖所有合法输入组合或全部 71 个能力点；
它覆盖的是此次候选池的 23 个已标注原子，语言能力证据仍需独立探针和实际执行支撑。
