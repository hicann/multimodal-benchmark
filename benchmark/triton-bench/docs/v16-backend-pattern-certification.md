# v0.16：把已取得的后端 IR 接入严格认证链路

2026-10-27 真机后续：批次 `20261027-043324-279333` 的 40 例全部通过，
两个目标标签均取得正式编译、执行认证。六轮历史分区叠加为 67/71（94.37%）。
详见 `docs/v16-npu-validation.md`；下文 `NOT_RUN` 等描述保留发布时状态，
不代表该后续批次未验证，也不代表已完成同版本全量或最终验收。

## 范围与状态

这是新冻结版本 `v0.16-backend-pattern-certification-candidates`，不是对 v15
结果或冻结资产的原地修订。全部 1377 例内容哈希、输入、参考、容差、71 标签
分母保持不变，保留原 68×20 例，提交侧 Python 内核与 v15 逐字节相同。
本轮没有修改用例选择、增加用例或改变被测优化。

静态门禁仍为 **67/71（94.37%）**。v16 的正式编译、执行认证状态为
`NOT_RUN`。本机离线识别成功不更新历史覆盖报告；历史分区叠加编译与执行认证
仍为 **65/71（91.55%）**。同版本全量、性能及任务书最终验收均未由此完成，
`AWAITING_JOINT_CONFIRMATION` 保留。

manifest：`e758eaef22e06c06a0d533bc5dabb1ee2bd35b74ff2af490dd2d718c57eff546`。
评测资产指纹：`4f1688c8cbed58ddfe8e02453734bb3f25d73febbfe8ba70ad2eebe5d52d43ab`。

## 修复内容

### 1. 正式后端采集使用实测打印配置

工具帮助同时列出厂商 before/after 过滤选项时，使用诊断批次已经验证的配置：

```text
--bishengir-print-ir-before=hivm-inject-sync,hivm-inject-block-sync,hivm-graph-sync-solver
--bishengir-print-ir-after=hivm-inject-sync,hivm-inject-block-sync,hivm-graph-sync-solver
```

该路径不再追加通用 after-all、module-scope、canonicalize 和 tree-dir 组合。
其他仅支持旧选项的编译器保留原有能力探测与回退，不注入帮助未列出的选项。
不改变 target、auto-multi-buffer、hfusion/hivm/triton 编译等语义选项。
当前实测输出 pass 为 GraphSyncSolver，不能推断三个候选 pass 都输出了 IR。

### 2. 支持完整函数级快照，不伪造 module

只有成功绑定到实际 `ttadapter` 输入及返回 `npubin` 二进制的后端调用，才可
提供函数级快照。额外约束同次 invocation 的 input、output、stdout/stderr
及 saved 文件前缀，不能跨调用挪用文本。

快照校验精确目标名、单一函数、完整括号、函数终止、非 HOST 身份，保留实际
Before/After 和 pass 名。只允许已实测的编译器 target-triple warning 尾部；
其他尾部内容、截断、其他函数和跨段拼接均不接纳。前端通用解析默认不开放
函数级快照，诊断 `probe-stderr` 原文也不能直接进入模式认证。

### 3. Neg：物理别名、补零与活跃数据流

识别 `GM X -> hivm.hir.load -> UB -> vmul/vadd -> hivm.hir.store -> GM Out`。
物理地址相同、空间及大小相同的 pointer_cast 视为同一可变缓冲区，而不是
三个独立 allocation。每次写入更新该缓冲区的依赖；通过别名覆盖载入结果后，
旧 load 不再成为证据。未知地址、部分重叠、未知内存操作和不支持的视图拒绝。

条件分支仅支持已观察到的零填充、视图及同步形式，不把任意 `scf.if` 当成
无条件执行。分支中的目标 load、矩阵运算、输出写回、else 和未知控制流拒绝。
UB 容量作为显式块传输宽度记录，不声称它等于硬件 SIMD 指令 lane 数。

### 4. Matmul：真实 L1 矩阵形式

新增完整链路 `GM X/Weight -> nd2nz -> CBUF -> hivm.hir.mmadL1 -> CC -> fixpipe -> GM Out`
的识别，其中实际操作名为 `hivm.hir.mmadL1`，不是把普通 dot 改名作为证据。
双参 pointer_cast 对应两个字节偏移，多缓冲数量须与 annotation 一致；每个
物理范围须与另一输入缓冲区互不重叠。两个输入必须分别来自 X、Weight，
并在同一已知非空、正步长的 K 循环内完成 staging 和矩阵运算；当前保守实现
拒绝未知/空循环、嵌套或多个循环、条件矩阵运算和不明控制流。

最终 fixpipe 必须读取同一 CC 矩阵结果并写入真实 Out。单边输入、互换来源、
无效多缓冲、输入被覆写、死矩阵结果、无回写均不能通过。新路径只报告实际
观察到的 `cbuf/cc`，不虚构 `l0a/l0b` 阶段。原有显式 L0 形式继续保留。

操作语义参考官方文档：
- https://ascendnpu-ir.gitcode.com/en/mlir-dialect/Dialects/HIVM/HIVMOps.html
- https://ascendnpu-ir.gitcode.com/en/developer_guide/features/multi_buffer.html

## 本机证据与测试边界

真实样本来自诊断批次 `20261026-155402-979553`，结果提交 `7c3011e6`。
Neg、Matmul 各接纳两段目标函数（Before、After），HOST 辅助函数不接纳。
使用原 v15 真实 case、源代码、TTIR 和张量绑定重放新规则，两项目标模式的
两个阶段均可离线识别。重新核验源报告 290 个证据文件，报告哈希保持
`3bef203781b9f65d679050ff8b93614a14058eafa60f6f28c4dd077206c45e36`。

`tests/fixtures/ascend_backend` 保留逐字节相同、哈希钉住的原始 stderr，
仅作为识别器测试 fixture。反例覆盖截断、HOST、其他函数、跨段拼接、条件
load/mmad、别名覆盖、缓冲重叠、缺失 annotation、零/未知循环、单边输入、
死输出等；模拟编译器测试同时检查采集配置、输入/二进制绑定及跨调用隔离。
这些测试和诊断编译重放都不是新一轮真机执行认证。

本机复核报告：`/private/tmp/triton-v16-offline-pattern-review.json`；
静态及入口门禁：`/private/tmp/triton-v16-readiness.json`、
`/private/tmp/triton-v16-submission-readiness.json`。

最终本机回归：**1027 passed、188 skipped**，日志
`/private/tmp/triton-v16-tests-final.log`。跳过项不计设备认证。
发布副本静态门禁通过，74 个入口源码契约通过（`load_state=NOT_RUN`），
发布副本另有 102 项定向测试通过。Shell 语法与 diff 空白检查通过。

## 真机只运行 Neg、Matmul 40 例

### 本机显式提交并发布

发布副本同步后，在本机执行以下命令。只暂存列出的改动，不使用 `git add -A`；
不会把工作区其他历史改动一起提交。推送不加 force。

```bash
bash <<'SH'
set -euo pipefail
cd /private/tmp/triton-eval-v09-publish-yg17nfbs
git add -- \
  src/triton_eval/{ascend_lowering,backend_capture,backend_evidence,compilation_evidence,coverage_evidence,coverage_generation}.py \
  scripts/{build_coverage_corpus.py,coverage-suite-verify.sh,check_submission_readiness.py,check_coverage_readiness.py,audit_coverage_results.py} \
  tests/{test_backend_capture.py,test_physical_backend_evidence.py,test_coverage_generation.py,test_coverage_npu.py,test_submission_preflight.py,test_ascend_suite.py,test_coverage_regression.py,test_corpus.py} \
  tests/fixtures/ascend_backend \
  docs/{v16-backend-pattern-certification.md,corpus-evolution.md,backend-dump-diagnosis.md,backend-dump-diagnosis-results.md,v15-npu-validation.md} \
  corpus/v0.16-backend-pattern-certification-candidates \
  examples/coverage-v16-ascend-submission
git diff --cached --stat
git commit -m "fix: bind and certify physical Ascend backend patterns in v16"
git push origin HEAD:refs/heads/codex/v16-backend-pattern-certification
SH
```

### 真机显式拉取该分支并定向验证

```bash
bash <<'SH'
set -euo pipefail
cd "$HOME/triton-eval-v07"
BRANCH=codex/v16-backend-pattern-certification
git fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
  git merge --ff-only "origin/$BRANCH"
else
  git switch -c "$BRANCH" --track "origin/$BRANCH"
fi
git log -1 --oneline
bash scripts/coverage-backend-verify.sh
SH
```

入口不接受扩大范围的参数，启动前重新计算静态覆盖、冻结指纹和入口契约。
使用本轮独立编译缓存，保留 CPU/NPU 参考、正确性和同步门禁。
不能复用旧版本缓存来冒充本轮新编译。

成功必须同时满足：
- 40 例完整执行且通过；
- `FA-PAT-VEC-LOAD` 的 compilation、execution_validation 都通过；
- `FA-PAT-TILE-GEMM` 的 compilation、execution_validation 都通过；
- 独立 `backend-pattern-validation.json` 的 `passed=true`。

只有功能 PASS、但上述任一认证仍缺证时，入口返回非零并保存发布日志。
本次仍为两个算子分区回归，不会将 40 例结果写成 1377 例同版本全量验收。
收到结果后，再按原始 case 身份及兼容性审计决定能否进行历史分区叠加。
