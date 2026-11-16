# v0.17：四项缺口的生成、选择与认证闭环

2026-10-27 运行后更新：批次 `20261027-092553-890191` 的 52 例全部通过，
多阶段归约、Split-K、向量广播认证通过；分块扫描因真实 IR 识别兼容性
问题未通过认证。修复发布到独立 v18，见 `v18-scan-certification.md`。
以下状态表保留为 v17 发布前本机记录，不代表运行后的最新认证结论。

## 当前状态：本机已冻结，真机认证待运行

本轮已实现候选生成、有约束选择、四种实际内核、编译数据流识别和独立模式门禁。
版本标识为 `v0.17-pattern-completion-candidates`。2026-10-27 已完成本机构建、
1385 例独立 CPU 参考检查和新版本冻结；尚未提交、推送或执行新的真机测试，
也未切换旧版本默认发布入口。

旧系统解释器 Python 3.9 保持不动。新独立环境位于 `.venv/pattern-build`，
实际使用 Python 3.11.18、NumPy 1.26.4、PyTorch 2.2.2、Pydantic 2.14.1、
PyYAML 6.0.3 和 pytest 9.1.1。未安装替代昇腾 Triton 的编译器，未修改真机环境。
构建器没有跳过参考门禁；第一次真实构建暴露的三例输入域问题也保留在追加式日志中。

| 检查 | 本轮状态 |
| --- | --- |
| 新增定向本机测试 | 28 项通过 |
| 含输入域、静态门禁和独立模式门禁的定向回归 | 117 项通过，16 个 subtest 通过 |
| 完整项目回归 | 1083 项通过，188 项跳过，21 个 subtest 通过；跳过项不作为设备认证 |
| 实际 Triton 内核源码的 NumPy 执行模拟 | 通过；不是编译或设备执行 |
| 旧 v16 40 个真实编译观测的离线识别重放 | 通过；原有 witness 不变 |
| 父版本 manifest、1377 例身份及便携参考资产 | 保持原样，已核查 |
| 新版本完整独立 CPU 参考门禁 | 1385/1385 通过 |
| 正式静态覆盖 | 71/71，100%，四项目标各有两个结构不同的专用载体 |
| 74 个提交入口的源码契约检查 | 通过；实际 Triton 模块加载尚未执行 |
| 新版本真实编译、执行认证 | NOT_RUN |
| 同版本全量、性能及任务书最终验收 | 未完成 |

旧 v16 冻结目录和提交样例未修改。公共框架增加了新版本支持，因此不能用
编辑后的框架冒充旧 v16 指纹；若复跑旧冻结版本，应使用已验证的 v16 代码提交。

## 选择策略

父版本锁定为已验证的 v16 manifest：
`e758eaef22e06c06a0d533bc5dabb1ee2bd35b74ff2af490dd2d718c57eff546`。
保留全部 1377 例，包括原 68×20 例；只补充 8 个目标用例，不按算子重新淘汰旧例。
候选仍通过注册规格和现有结构谓词校验，不在 case 中绑定实现、覆盖容差或手填标签。

`pattern_generation.select_patterns` 与旧策略的区别是：

- 四个目标是显式义务，不能因为全局静态覆盖已超过 90% 就忽略其中一项。
- 每项要求两个不同结构的 witness；单换 seed 不能补足数量。
- 区分“负载可能取得的标签”和“本轮该目标的专门载体”，不会拿 Cummin 的
  大轴静态归约标签替代专门的 ReduceMax 多趟归约实现。
- 总新增预算 16、每算子预算 4、单例输入/输出/显式中间缓冲估算上限 64 MiB。
  估算不是实测设备峰值，不包含编译器或参考实现的额外内存。
- 预算或可达性不足时硬失败，并报告缺少哪些义务；不缩分母、不降低阈值。
- 静态选择日志中的编译、执行状态始终为 NOT_RUN，真机证据另行生成。

正式构建保留 1377 例并新增 8 例，共 1385 例；静态标签并集为 71/71。
这一分母不是每类算子覆盖率的算术平均，也不表示 Triton API/IR 的所有操作已覆盖。
独立编译、执行认证状态仍为 NOT_RUN，不能由静态 100% 推导出动态 100%。

## 本轮本机构建修复

首次完整参考检查中，两例长轴 Cummin 和一例 K=64 的 GroupedMatmul 被旧扩展输入域拒绝。
这不是参考数值不一致：旧校验只接受 rank3、轴长不超过 32 的 Cummin 和 K=32 的分组乘法。
新增 `pattern_contracts.py` 仅在带有 v17 版本标识的 coverage-plan 下，接纳候选计划中
精确列出的 Cummin/GroupedMatmul 形状、dtype 和属性组合；未列出的组合仍走旧校验。
新增域仍检查输入元数、张量类型、连续布局、同 dtype/设备，以及 weight/bias/分组关系。
原输入工厂、Torch/NumPy 参考、容差文件和旧冻结资产均未修改。

失败尝试与之后通过的检查保留在新版本 `generation-log.yaml`；只有最后一次完整检查
通过后才记录 build_end。发布准备检查验证最后一次构建，不能拿更早的通过记录掩盖失败。
本轮尚未发布的 v17 开发中间快照在修正框架后重新完整构建；没有重冻 v16 或更早版本。
独立复核确认父版本 148 个参考文件、1377 例内容哈希及输入工厂、两类参考、规格与容差
保持原样；历史版本在新框架下仍会因指纹不符而拒绝启动，相关回归也保留了拒绝断言。

冻结身份：

- manifest：`95ba03ce7a13bb2e79223e9f710e95c00921499521e0987d036a1c82bc4c3ed1`
- assets：`9aa9135b7055c57c92e4c9f045fce8ea02a8559fa9cb05ee612ff04cdd4cf3ad`

## 四种实现

### 多趟归约

选用已有 ReduceMax 契约，避免把本轮扩大为新增求和容差问题。
两例为 `[1, 1048576]` 和 `[2, 1048833]`，包含不满块尾部及 keepdim 差异。
第一阶段每 1024 元素产生一个 float32 部分最大值，第二阶段归约部分结果到 Out。
认证检查两个阶段的真实缓冲身份、形状、类型、grid、常量参数、存活最大值归约
以及最终输出，拒绝错误归约种类、死结果、伪造中间缓冲和别名。

### 分块扫描

选用已有 Cummin 双输出契约，两例扫描轴长度为 1048576 和 1048833。
不再只用总元素数量判断是否有长扫描轴。
实现为块内 value/index 对扫描、块间携带前缀、最终修正三个阶段。
逻辑上仍是“局部扫描＋跨块修正”，不声称只有两个 kernel launch。
相等最小值采用参考要求的最后出现索引；数值和索引都必须经过参考验证。
识别器检查三个阶段的实际张量链、活跃 scan 区域、最小值/相等时较大索引语义
以及最终 value/index 配对合并，不接受缺少携带阶段或只验证数值输出。

### Split-K

两例主输入为 `[16, 65536]`、`[32, 65600]`，权重继续由原输入工厂生成。
8 个并行 K 分区产生 float32 部分乘积，再通过固定归约写入输出，不使用原子累加。
识别器验证两个矩阵操作数各自来自 X/Weight，且第三个实际 program 轴对
输入地址的系数分别为 SEGMENT、SEGMENT×N，对部分输出地址的系数为 M×N。
普通内核里的 K 循环、仅出现第三维 grid、死 dot 或重叠部分缓冲都不够。
大 K 的数值结果仍必须通过原有容差，不放宽容差来接受实现。

### bias 跨行复用

选用真正具有 bias 输入的 GroupedMatmul，主输入 `[2048, 32]`、`[2064, 64]`。
每个计算 tile 加载一份 16 列 bias，在 16 行计算中广播复用。
静态派生为本轮补充用例增加实际载体约束，普通大 M Matmul 不能冒充带 bias 负载。
认证要求 bias 的真实输入绑定、行不变的一维加载、存活广播及最终输出依赖。
这证明的是编译后的 tile 内向量复用，不证明硬件 cache 命中率、全局驻留或性能收益。

四种模式的识别证据层明确记为 TTIR，结合受控内核源码、实际特化、张量身份及
编译产物链；不把它们描述为已观测到的物理后端指令。
不同于 v16 两项物理访存模式，这四项首先考察跨 launch/并行分区/广播的程序结构。
未知 IR 拼写或结构保守拒绝，需收到真实产物后再决定是否适配。

## 测试和运行入口

以下命令仅生成草案，不冻结、不更新覆盖认证：

```bash
python3 scripts/build_pattern_corpus.py --plan-only
```

完整 Python 3.11+ 开发环境下，正式构建入口为：

```bash
.venv/pattern-build/bin/python scripts/build_pattern_corpus.py
```

构建器必须先通过全部新旧用例的独立 CPU 参考门禁，才写正式 manifest、
build_end 和冻结评测资产。不提供跳过参考门禁的参数。
已冻结时重运行会验证已有身份，不会原地刷新冻结指纹或跳过参考检查记录。

后续真机入口为 `scripts/coverage-patterns-verify.sh`，固定选择
ReduceMax、Cummin、Matmul、GroupedMatmul 四个算子，分别 22、4、22、4 例，
合计 52 例，不自动启动 1385 例全量，也不接受扩大范围的参数。
运行前检查冻结指纹和源码契约，运行中保留 CPU/NPU 参考、完整性、同步和正确性门禁。
独立 `target-pattern-validation.json` 要求四项各有两个结构不同的编译、执行 witness；
仅功能 PASS、仅一次 witness 或仅静态标签都不能通过。

本机定向测试可独立运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_target_patterns.py -v
```

新环境下完整本机回归命令为：

```bash
.venv/pattern-build/bin/python -m pytest -q
.venv/pattern-build/bin/python scripts/check_coverage_readiness.py \
  --corpus corpus/v0.17-pattern-completion-candidates \
  --submission examples/coverage-v17-ascend-submission
```

NumPy 可用时执行实际内核源码模拟；无 NumPy 时相关模拟明确跳过。
本次使用的 NumPy 环境下 28 项全部通过，无跳过。合成 IR 仅用于验证识别器，
绝不作为真机认证证据。

本机产物为 `/private/tmp/triton-v17-pattern-plan.json`、
`/private/tmp/triton-v17-pattern-tests.log` 和 `/private/tmp/triton-v17-v16-replay.json`。
本轮新环境产物为 `/private/tmp/triton-v17-build-py311.log`、
`/private/tmp/triton-v17-focused-pytest.log`、`/private/tmp/triton-v17-full-pytest.log`、
`/private/tmp/triton-v17-static-readiness.json`、`/private/tmp/triton-v17-submission-readiness.json`、
`/private/tmp/triton-v17-local-validation.json` 和 `/private/tmp/triton-v17-frozen-reference-check.log`。
正式静态覆盖已达成；新四项动态认证保持未知。已有 v16 历史叠加 67/71（94.37%）
不因本机参考检查、模拟或合成 IR 测试而提升，也不能自动成为 v17 同版本全量认证。
共同确认状态保持 `AWAITING_JOINT_CONFIRMATION`。
