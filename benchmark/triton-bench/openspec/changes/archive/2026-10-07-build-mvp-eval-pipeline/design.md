# Design

## Context

动机见 `proposal.md`。本节只记录约束设计的前提事实，全部来自在目标执行机上的实测。

**执行环境（CANNLab 昇腾主机，已实测）**

| 项 | 实测值 |
|---|---|
| 架构 / OS / 内核 | `aarch64` / Ubuntu 20.04.5 / `5.10.0-hce2` |
| NPU | `Ascend910_9362` × 2，HBM 64GB/卡，容器可见 `npu:0` / `npu:1` |
| CANN | `9.0.0`，路径 `~/Ascend/cann-9.0.0`（非 root 安装，`set_env.sh` 不在 `/usr/local`） |
| Python / torch / torch_npu | 3.11.4 / 2.7.1+cpu / 2.7.1.post4 |
| 被测编译器 | `triton-ascend` 3.2.1（`import triton` 实测为 3.2.0） |
| 权限 / 网络 / 持久化 | 免密 sudo 可用；华为云昇腾源与 PyPI 均可达；`/home` 196G |
| 容器 | 无 docker |

**五条硬约束（均由实测得出，直接决定设计）**

1. **两卡共享、非独占**：HBM 恒有约 3.1GB 占用但本容器无任何进程，占用来自其他容器；无锁频权限。
2. **`triton.knobs` 在 triton-ascend 3.2.1 上是坏的**：干净 venv 全新安装后仍报 `ImportError: cannot import name 'getenv' from 'triton._C.libtriton'`。这是被测编译器自身的能力缺失，不是环境污染 → **`TRITON_INTERPRET` 在该版本上确证不可用**。
3. **`@triton.jit` 依赖源文件**：`inspect.getsourcelines` 反查源码，从 stdin / `exec` / `eval` 加载的 kernel 一律失败 → kernel 必须是文件系统上的真实文件，不能存进数据库再执行。
4. **`triton` 与 `triton-ascend` 安装目录同名**：装社区 `triton` 会覆盖并静默破坏被测对象 → 评测工具链的依赖清单中禁止出现 `triton`。
5. **无上游 Triton 可装**：昇腾上不存在官方 Triton 上游实现 → 报告 §6.3 所需的 `P_ref_geo`（B0 基线）不可得。

## Goals / Non-Goals

**Goals:**

- 一条命令完成「冻结 corpus → 执行 → 判定 → 落库 → 评分 → 出报告」的完整闭环
- 每份产物可追溯到环境指纹、corpus 版本、评分器版本
- 换开发机、换执行机的成本低且路径明确
- 判定规则无歧义：case 终态互斥且完备，每个字段语义唯一

**Non-Goals:**

- 不产出可上榜的性能数字（环境共享 + 无锁频，结果只作技术验证）
- 不构建生成层（抽取器、变异、覆盖引导、去重门禁）
- 不构建 Web / 榜单 / 多人提交流程
- 不修正技术报告本身的条文冲突（仅在实现时按本设计取舍，并回报需修订处）

## Decisions

### D1 · 冻结现成版本组合，不追新

被测基线锁定为 `triton-ascend` 3.2.1 + `torch_npu` 2.7.1.post4 + CANN 9.0.0，尽管官方最新配套是 3.2.2 / post8 / CANN 9.1.0。

- **理由**：MVP 目标是打通链路而非追新；在无 docker 的 aarch64 环境手动升级 CANN toolkit 是半天到一天的坑，收益为零。且固定快照更符合「冻结」理念。
- **备选**：升级到最新配套 → 否，引入无谓风险；且保留 3.2.1 这一已知快照，将来升级时可顺带验证跨版本回归判定链路。

### D2 · 被测编译器用独立 venv 隔离

`--system-site-packages` 建 venv 继承 torch / torch_npu，配 `PYTHONNOUSERSITE=1` 屏蔽用户级 site-packages，venv 内只装 `triton-ascend`。

- **理由**：约束 4 表明目录同名会互相覆盖；隔离后被测编译器的安装状态可被精确重建与快照（`pip freeze` 哈希入环境指纹）。
- **备选**：直接用系统 Python → 否，无法保证不被后续装包污染，且无法快照。

### D3 · 按 NPU 依赖切分 L0 / L1，契约是 `results.jsonl`

- **L1**（只能在昇腾主机跑）：`npu_exec`（编译执行）、`bench`（`do_bench` 调用）、`envprobe`（`npu-smi` 采集）。全仓库只有 `npu_exec` 一处 `import triton`。
- **L0**（macOS / Linux 通用）：规范加载、corpus 校验、评分、报告、存储、CLI。
- **契约**：L1 产出 `runs/<id>/results.jsonl`（一行一 case），L0 消费它。

- **理由**：这条契约是可迁移性的支点——拿到 jsonl 就能在任何无 NPU 的机器上重算分数、重出报告、改了评分逻辑后批量重算历史结果。
- **备选**：直接写库、不落 jsonl → 否，则评分与报告永久绑定 NPU 环境。

### D4 · 采集 / 解析分离，L1 压薄

L1 只负责「调一下、把原始输出存下来」，所有解析与判定逻辑下沉到 L0：

| 模块 | L1（采集） | L0（解析 / 判定） |
|---|---|---|
| envprobe | 执行 `npu-smi`、存原始输出 | 解析 → 构造 `env_json` → 规范化 → 算 `env_fp` |
| bench | 调 `do_bench`，取回样本数组 | 算 median / MAD / CV / speedup / 不稳定判定 |
| harness | 编译 + 执行，取回张量与耗时 | 与参考比对 → 容差判定 → `status` → 组装 record |

- **理由**：把易错的判定逻辑推到可测试的 L0；L1 收敛为约百行直白调用。配合把 NPU 命令原始输出存为 fixture，L0 可在 macOS 上完整单测。
- **备选**：L1 内部完成判定 → 否，则核心逻辑在无 NPU 时完全无法测试。

### D5 · corpus 规模缩到 20–30，schema 不缩

- 数量：MVP 20–30 个 case（P1 再扩到 1000+）
- 字段：`case` 表的全部字段按最终结构保留（`seed` / `feature_atoms_json` / `launch_json` / `content_hash` / 容差来源等），不因规模小而裁字段
- 来源：以 triton-ascend 自带 `tutorials/` 与 `unittest/pytest_ut/` 算子为 seed，全部打 `source: vendor_selftest`

- **理由**：规模缩可砍掉整个生成层；schema 不缩可保证 P1 灌数据时只改内容不改代码。schema 是固定资产，corpus 只是内容。
- **备选**：直接上真实算子抽取（FlagGems / vLLM）→ 否，多花 1–2 周且与「打通流程」无关。
- **风险对冲**：用被测方自带用例存在「刷分后门」风险，故以 `source` 字段显式标记，正式 corpus 阶段限制该来源占比 ≤20% 且不计主得分。

### D6 · 算子权重 MVP 取等权

`operator.weight` 在 MVP 全部置 1.0。

- **理由**：真实权重需 profiler 采集「调用频次 × 耗时占比」，属 P1 工程。等权最诚实、可解释、无争议。
- **备选**：手工拍权重 → 否，无依据的权重会污染 `OpScore` 的解释性。

### D7 · 单 oracle 判定 + 双参考交叉校验

- **判定层**：仅 torch eager on NPU，PASS / WRONG_RESULT 只由它决定
- **校验层**：corpus 内为每个算子提供第二参考实现（CPU 路径），与主 oracle 交叉；不一致 → `ORACLE_FAIL`
- **信号层**：`TRITON_INTERPRET` 确证不可用（约束 2），不参与判定

- **理由**：双 oracle 仲裁逻辑在单 oracle 已足够时是纯复杂度；交叉校验是纯评测方资产，不依赖被测编译器能力。
- **备选**：按报告 §4.2 做双 oracle + 仲裁 → 否；且 `TRITON_INTERPRET` 是 triton 的一部分、triton-ascend 是其 fork，用它验证 triton-ascend 存在循环论证问题。

### D8 · 容差按 dtype 分级，禁止 case 级覆盖

| dtype | rel | abs |
|---|---|---|
| int8 / int16 / int32 / int64 / uint* / bool | 0 | 0 |
| float32 | 1e-5 | 1e-6 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 5e-3 | 1e-4 |

参考实现统一以 `float64` 精度计算后再 cast 回目标 dtype（已实测昇腾支持 fp64）。

- **理由**：报告 §6.1.1 只给了 `rel 2e-2 / abs 1e-4` 一组数，对 fp32 过松、对整型完全不适用。禁止 case 覆盖可消灭「调容差刷通过率」的后门。
- **备选**：允许 case 覆盖并记原因 → 否，字段与解释成本都高，且开后门。

### D9 · 确定性在冻结阶段解决，不在评测阶段解决

corpus 冻结前每个 case 连续跑 5 次，出现结果不一致即剔除并记录到排除列表。

- **理由**：不给状态机增加 `NONDETERMINISTIC` 分支，评分分母保持干净。不确定性是 corpus 的质量问题，应在冻结前筛掉。
- **备选**：评测时对归约类 case 做多次运行取一致 → 否，增加状态与计时成本。

### D10 · 8 个终态状态机，`severe` / `counted` 由 `status` 派生

```
SPEC_ERROR / COMPILE_FAIL / RUNTIME_FAIL / PASS /
WRONG_RESULT / SHAPE_MISMATCH / ORACLE_FAIL / ENV_ABORT
```

派生规则（写死在代码常量，不落列）：

```
severe  = (status == 'WRONG_RESULT')
counted = (status NOT IN ('SPEC_ERROR','ORACLE_FAIL','ENV_ABORT'))
```

`counted=0` 的三种状态是 corpus 或评测方自身的 bug，被测方不为其背分，但必须落库并触发告警。

- **理由**：互斥且完备（MECE）是「无歧义」的核心；不引入 `UNSUPPORTED` 是因为该机制可被用于声明不支持来剔除难题，是刷分后门，且 MVP 无多提交者场景。
- **备选**：按报告 §4.9 引入 capability manifest + `UNSUPPORTED` → 否，复杂度爆炸。

### D11 · 性能：B1 基线 + 首轮锚点，不出 Total

- 计时：`do_bench(warmup=25, rep=100, return_mode="median")`——**必须显式指定 median**，该函数默认返回 `mean`（已实测签名）
- 基线：同机会话内 torch eager 交错测量（B1）
- 归一化：因约束 5（无 B0），`P_ref_geo` 取**首轮基线冻结值**并定义为 100 分，后续版本相对它计算；`baseline_anchor_ref` 字段显式记录锚点来源
- **不合成 `Total`**：报告 §3.3 禁止两赛道隐式换算，与 §6.4 的加权合成冲突，本设计采用 §3.3，按赛道分列

- **备选**：只报 `P_geo` 裸指标 → 否，无法出分与排序；用厂商库 B2 作锚点 → 否，报告明确 B2 仅作说明性指标。

### D12 · 存储：SQLite + 7 表，`results.jsonl` 为真相源

7 张表：`corpus_version` / `operator` / `case` / `compiler_submission` / `run` / `case_result` / `run_score`。主键全部自增 `INTEGER`，业务唯一性另设 UNIQUE 约束。

- **理由**：MVP 单执行点、无并发写入，SQLite 零运维；schema 不用 SQLite 专有类型，迁 PG 只需改连接串。
- **`results.jsonl` 是 SoT，SQLite 是查询加速层**：数据库可从 jsonl 重建，因此 `store` 层必须提供 `import`。
- **备选**：直接上 PostgreSQL → 否，MVP 无多提交者并发场景；FA 单独建表 → 否，MVP 只存不查，用 JSON 数组足够。

### D13 · 平台：CLI + 本地报告，不做 Web

子命令：`run` / `score` / `report` / `import`。输出 Markdown + HTML 到本地文件。

- **理由**：Web 是展示层，与「流程能否跑通」无关；等 `case_result` 有真实数据后套 Web 是低风险增量。

### D14 · 开发工作流：rsync 同步 + pytest marker 分层

- 本地编辑 → `env/sync-and-run.sh`（rsync，`--exclude runs/`）→ 远端跑 `--profile smoke`（3–5 case，秒级反馈）
- 测试标记：`pytest -m "not npu"` 在 macOS；NPU 上跑全部
- 代码始终以 Git 为准，不在远端直接改（CANNLab 环境可回收）

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|---|---|---|
| 共享环境导致性能数据不可用 | 性能数字不可上榜 | 报告强制标注 `UNVERIFIED`；CV 门禁在 smoke/standard 放宽至 0.05；性能赛道目标降为「技术链路打通」 |
| CANNLab 环境被回收，历史结果丢失 | 唯一不可重建资产灭失 | `runs/` 目录约定 + 定期 rsync 导出；`results.jsonl` 为真相源可单独归档 |
| 用被测方自带用例做 seed，存在刷分后门 | 评测公信力 | `source=vendor_selftest` 显式标记；正式 corpus 限占比 ≤20% 且不计主得分 |
| 被测编译器被评测工具链弄坏（目录同名覆盖） | 大面积假阳性 FAIL | `requirements-eval.txt` 禁 `triton`；venv 隔离；每次 run 前跑健康探针，失败则整轮 `INVALID` |
| corpus 太小，`OpScore` 统计意义弱 | 分数不代表真实完备性 | MVP 明确不出正式分数，只验证链路；P1 扩至 1000+ |
| 首轮锚点定义易被误读为跨编译器对照 | 分数解释争议 | `baseline_anchor_ref` 落库；报告模板中固定写明「相对本体系首轮基线」 |
| 技术报告 §3.3 与 §6.4 冲突未解决 | 规范自相矛盾 | 本设计采用 §3.3（不出 Total）；需回报技术报告修订 |

## Migration Plan

本项目为新建，无存量系统迁移。上线顺序：

1. 环境固化：`env/env.sh` + 两个 requirements + venv，跑通最小 kernel 探针
2. 采集 fixture：`npu-smi` 原始输出、`do_bench` 样本、样例 `results.jsonl` 入库到 `tests/fixtures/`
3. corpus 冻结：seed 算子入库 → 5 次确定性筛查 → 生成 manifest → 冻结
4. L1 实现：envprobe / bench / npu_exec
5. L0 实现：store / scorer / report / CLI
6. 端到端 smoke → standard → 首轮基线冻结为锚点

**回滚**：全新建项目，无回滚需求；任一步失败只需删除对应目录重来，Git 保证代码不丢。

**执行机失效时的恢复路径**：`git clone` → 装 CANN → `source env.sh` → `pip install -r requirements-compiler.txt` → `import` 历史 jsonl 重建数据库。约半天，主要耗时在 CANN。

## Open Questions

- 若后续取得独占昇腾机器，`P_ref_geo` 是否改用 B0（上游对照）而非首轮锚点——不影响 MVP 的任务分解与 spec。
- corpus 权重从等权切换到 profiler 加权时，历史分数是按新权重全量重算还是仅对新 run 生效——P1 决策，MVP 等权不受影响。
- 正式 corpus 中 `vendor_selftest` 的占比上限（暂定 20%）是否需按赛道分别设定——P1 决策。
