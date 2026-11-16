# 面向 Triton 编译器的功能完备性与综合性能评测体系 —— 可行性分析与技术报告

> 文档版本：v0.1（草案）
> 调研时点：2026-09
> 适用范围：Triton 编译器评测体系建设（规范层 / 生成层 / 运行层）
> 读者对象：项目立项评审、架构设计、工程实施与运营团队

---

## 0. 阅读指引

本报告回答四个问题：

| 问题 | 章节 |
| --- | --- |
| 这个规划能不能做、风险在哪 | 第 1、2 章（可行性分析 + 现状调研） |
| 难在哪里、怎么解 | 第 4 章（12 项重难点 + 针对性方案） |
| 用什么技术栈 | 第 5 章 |
| 评测体系到底怎么落地（口径、格式、公式、流程） | 第 3、6、7、8 章 |

---

## 1. 可行性分析

### 1.1 总体判断

**结论：规划整体可行，且时机成熟；但需按"规范先行、测试集分批冻结、平台后建"的顺序分期实施，并对若干指标口径做工程化裁剪。**

判断依据分四条主线展开：技术可行性、工程可行性、度量可行性、运营可行性。

### 1.2 技术可行性：高

支撑条件已经具备，不需要发明新方法：

1. **评测对象边界清晰且可枚举。** Triton 的语言面（language surface）是封闭的 Python DSL，`triton.language` 官方 API 文档可分为 15 个类别（编程模型 / 创建 / 形状操作 / 线性代数 / 内存指针 / 索引 / 数学 / 归约 / Scan-Sort / 原子 / 随机数 / 迭代器 / 内联汇编 / 编译提示 / 调试），合计约 100 个可命名算子与类型。这直接给出了**覆盖维度模型的主干**，也是"核心语言特性清单覆盖率 ≥ 90%"这一指标可计算的前提。
2. **编译器行为可观测。** Triton 编译链 TTIR → TTGIR → LLIR → PTX/cubin（AMD 为 HSACO/ROCm，Ascend 为对应后端产物）的中间表示均可导出，既有成熟的 IR dump 机制，也有 `TRITON_INTERPRET` 这样的解释器模式作为**不依赖硬件的语义 oracle**。这让"功能完备性"可以从"跑得对不对"升级为"编译是否正确 + 语义是否等价 + 代码质量如何"三层次判定。
3. **测试集生成方法学有成熟范式可迁移。** 编译器随机测试领域已有 Csmith（随机 C 程序生成）、YARPGen（面向编译器优化 bug 的随机程序生成）、差分测试 / EMI（Equivalence Modulo Inputs）、覆盖率引导变异等成熟技术。Triton 作为受限 DSL（无指针别名歧义、无 UB 的主要来源、表达式结构受 tile 语义约束），其程序生成难度**显著低于 C/C++**，可复用上述方法且误判率更低。
4. **多硬件生态已形成对比需求。** Triton 已具备 NVIDIA / AMD 官方后端与多个第三方后端；昇腾 Triton-Ascend 自 2025-05 开源起按里程碑推进（2025-06-30 支持 85% Triton Python API 与连续访存；2025-08-15 完善 Atomic 类 API；2025-09-30 完善 Scan/Sort 类 API、支持非连续访存并完成 vLLM / sglang 重点算子适配；后续发布 3.2.0 / 3.2.1 版本并配套 CANN 社区版）。这类"API 支持度按里程碑爬坡"的公开路线图，恰好需要一个外部、中立、可量化的评测体系来度量——本项目具备明确的产业需求侧支撑。
5. **执行与统计工具链现成。** `triton.testing.do_bench` 已提供 warmup / rep / 分位数 / L2 flush（fast_flush）的默认口径（默认 warmup=25ms、rep=100ms，返回中位数及 20/80 分位），可作为本体系"统计口径"的基线与兼容层，不必从零实现计时器。

### 1.3 工程可行性：中等偏高，关键在成本控制

规模估算（用于验证"1000+ case × 多编译器 × 多硬件"是否跑得动）：

| 项目 | 估算 |
| --- | --- |
| 功能 case 数 | 700（单 case 平均：JIT 编译 1–3s + 执行/校验 0.3–1s，含失败快速退出） |
| 性能 case 数 | 300（单 case：编译 1–3s + 5 轮 ×(warmup 25ms + rep 100ms)≈0.6s） |
| 单次全量（1 编译器 × 1 硬件） | 编译约 1000×2s ≈ 33min（可命中 JIT 缓存大幅下降），执行与测量约 15min，合计 **≈ 45–60 min（未并行）** |
| 并行化后 | 8 卡单机多进程并行 + case 级并行，可压到 **10–15 min** |
| 全矩阵（3 编译器 × 3 硬件 × 5 次重复） | 约 45 次全量 → 单机串行约 30h，4 节点并行约 **8h** |

结论：全量评测可做到"每日一轮"，增量评测（仅变更 case / 仅变更编译器）可做到小时级。**工程上可行，但必须把并行执行、JIT 缓存复用、分层抽样（smoke / full）作为一等公民设计进去**（见 §4.10）。

风险点不在"跑不完"，而在"跑出来的数不能复现、不能比较"——这是本项目真正的技术含量所在，见 §4.3、§4.4。

### 1.4 度量可行性：中等，需重点设计

可行性最弱的一环是**度量本身的可辩护性**：

- "功能完备性覆盖率 ≥ 90%" 若定义为"用到了该特性的 case 通过率"，会被"用极简单 case 覆盖大量特性"刷高；若定义为"特性组合覆盖"，组合爆炸。→ 必须引入**分层特性原子 + 覆盖强度分级 + 权重**（§4.1）。
- "综合性能"若直接用绝对时延，跨硬件不可比；若用相对 PyTorch 的加速比，PyTorch 版本一变基线就漂；若跨厂商横向排名，容易被质疑。→ 必须引入**同机同时段基线 + 分组榜单 + 置信区间**（§4.3、§6.4）。
- 数值正确性的"正确"需要可判定标准。→ 必须引入**双 oracle + 分级容差 + 确定性约束**（§4.2）。

这三点由 §4 的前四项重难点逐一给出可落地方案，因此判定为"可行但需重点设计"。

### 1.5 运营可行性：中等，依赖治理机制而非技术

平台原型的开发不难，难的是长期有人提交、有人复跑、有人治理。需要在项目期内就把**版本发布节奏、case 争议仲裁流程、榜单准入与退出、errata（勘误）机制**写成制度文件并试运行一轮，否则交付即停摆（§4.8、§4.12、§8.5）。

### 1.6 规划中需要修正/补充的三处

1. **"覆盖 Triton 核心语言特性清单比例不低于 90%"**：建议改为两条互补指标——(a) *测试集侧*：清单中被至少一个有效 case 覆盖的特性比例 ≥ 95%；(b) *编译器侧*：被测编译器在测试集上的**加权特性通过率**（这是真正的完备性得分）。单列一条 90% 容易被误读为"测试集自嗨"。
2. **"不少于 1000 个 case"**：建议明确**有效 case 的定义**（通过编译 + 语义校验 + 通过去重与平凡性检测），并要求"每个特性原子至少 5 个独立 case"，避免 1000 个 case 挤在少数热点特性上。
3. **"平台原型"**：建议把"可复现性"作为平台的头等公民——每次榜单条目必须能一键追溯到 `环境指纹 + case 快照版本 + 原始日志 + 产物签名`，否则榜单无公信力。
4. **补充：明确一级评测单元为算子**（已在 §3.4 展开）。若把测试集建成"API 逐项用例集"，会出现"覆盖率虚高但真实算子跑不通"的失真；应以**算子为探针**、以**语言特性为诊断坐标**，并把算子池优先对齐 triton-ascend 路线图中的适配目标（FlagGems / vLLM / sglang 重点算子）。

---

## 2. 现状调研（联网）

### 2.1 已有 Triton 相关评测工作

| 工作 | 定位 | 规模/口径 | 与本项目的关系 |
| --- | --- | --- | --- |
| **TritonBench（THUNLP, arXiv:2502.14752, ACL Findings 2025）** | 评测 **LLM 生成 Triton 算子**的能力 | 双通道：TritonBench-G（184 个来自 GitHub 的真实算子）/ TritonBench-T（对齐 PyTorch 接口的算子）；指标：CodeBLEU 相似度、call accuracy、execution accuracy、efficiency（speedup）；环境 triton 3.1.0 + torch ≥ 2.5.1 | **评的是"模型"，不是"编译器"**。其算子集合与执行/性能评测管线可作为本项目的**负载来源与工程参考**，但评测目标不同 |
| **KernelBench（Stanford, 2025）** | 评测 LLM 写 CUDA/Triton kernel | 250 个 PyTorch 任务、3 个难度层级；核心指标 `fast_p`：既正确、相对 PyTorch 参考加速比又超过阈值 p 的任务占比 | `fast_p` 的"正确性 ∧ 性能"联合口径值得直接借鉴到本项目的性能得分设计 |
| **KernelBench-Triton / KernelLLM（Meta）** | KernelBench 的 Triton 变体 + 8B 生成模型 | SFT 训练 + prompt 模板 | 说明"Triton 负载集"的复用价值，也说明缺少统一规范时各家会各自造变体 |
| **meta-pytorch/tritonbench** | PyTorch 自定义算子性能集合，评估 **Triton 及其与 PyTorch 集成**的性能 | `python run.py --op gemm`；依赖 Liger-Kernel、FBGEMM、flash-attention、tilelang、ThunderKittens、AITer 等 submodule | 是**性能负载的重要素材库**，但它不做功能完备性评测、不做规范、不做榜单 |
| **GEAK-Triton（AMD 侧）** | 面向 AMD GPU 的 Triton kernel 优化 Agent | 在 TritonBench-modified 上报告平均加速 3.42×，另有 ROCm-bench | 说明各家在"自建变体"上各自报数，**横向不可比**——正是本项目要解决的问题 |
| **RealisticTritonBench（2026）** | 真实 AI 框架中的 Triton kernel 生成 | 自然语言需求 → Triton 实现，强调可复现评测环境 | 再次印证"可复现环境"是社区共同痛点 |
| **Triton 上游测试体系** | 编译器自身回归测试 | `python/test/unit/language/` 下：test_core、test_standard、test_matmul、test_conversions、test_frontend、test_annotations、test_decorator、test_random、test_pipeliner、test_tensor_descriptor、test_warp_specialization、test_mxfp、test_clc、test_libdevice、test_line_info、test_module、test_reproducer、test_subprocess、test_tuple、test_compile_errors、test_compile_only；另有 MLIR lit 测试与集成测试 | **最贴近本项目"功能完备性"的参考**。但它服务于上游 CI（防回归），不是中立评测集、无冻结版本概念、无覆盖率量化 |
| **FlagGems** | 跨芯片 Triton 通用算子库（智源等联合研发，已加入 PyTorch 基金会生态项目体系），宣称支持 20+ 种芯片 | 注册到 ATen backend 的大模型算子库 | **真实世界算子负载与跨芯片一致性需求的来源**，可作为性能测试集的重要素材与"真实场景覆盖"校验集 |

### 2.2 关键差距分析（Gap → 本项目定位）

| 差距 | 现状 | 本项目补位 |
| --- | --- | --- |
| G1 无中立规范 | 各家自建变体（TritonBench-modified、ROCm-bench、KernelBench-Triton…） | **规范层**：定义评测边界、合法输入、排除规则、结果解释边界 |
| G2 无可复现运行规则 | 论文中常只写 "A100 / H100"，无时钟、无驱动、无 warmup 口径、无重复次数 | **运行层**：环境指纹 + 锁时钟 + 统一统计口径 + 原始日志留存 |
| G3 无功能完备性量化 | 上游测试是"通过/失败"回归，不产出完备性分数 | **生成层 + 评分**：特性原子清单 → 加权覆盖率 |
| G4 无冻结与版本治理 | 数据集随仓库漂移，历史结果无法复现 | **corpus 冻结存档 + seed/参数/版本元数据 + errata 流程** |
| G5 无持续榜单与提交流程 | 一次性论文产物 | **平台层**：提交模板 → 自动复跑 → 评分 → 榜单发布 |

### 2.3 结论

> 现有工作覆盖了"用什么负载"（TritonBench / tritonbench / FlagGems）和"怎么计时"（do_bench / fast_p），但**没有一项工作覆盖"如何中立、可复现、可持续地评价一个 Triton 编译器本身"**。本项目的定位准确，填补的是规范与治理的空白，而非重复造负载。

---

## 3. 评测体系总体设计

### 3.1 三层架构

```
┌──────────────────────────────────────────────────────────────┐
│  规范层 Specification  —— 解决"评什么"                          │
│  · 评测边界与不评测边界      · 语言特性覆盖维度模型                 │
│  · 合法输入 / 排除规则      · 结果解释边界                        │
│  · 统一数据格式 / 运行规则 / 报告口径 / 版本规则                    │
└───────────────┬──────────────────────────────────────────────┘
                │ 产出：spec.yaml（机器可读的规范单）
┌───────────────▼──────────────────────────────────────────────┐
│  生成层 Generator —— 解决"用什么评"                              │
│  · 参数化生成（模板 × dtype × shape × 控制流 × 优化 pattern）      │
│  · 覆盖引导生成（覆盖率反馈 → 变异 → 门禁 → corpus 演化）           │
│  · 双 oracle 校验（PyTorch/NumPy 参考 + TRITON_INTERPRET 解释）   │
│  · 去重 / 平凡性检测 / 冻结存档（seed + 参数 + 版本元数据）          │
└───────────────┬──────────────────────────────────────────────┘
                │ 产出：corpus-vX.Y（冻结测试集 + manifest）
┌───────────────▼──────────────────────────────────────────────┐
│  运行层 Run Rules / Report —— 解决"如何持续运行"                  │
│  · 环境冻结与指纹  · 调度执行（本地/CI/集群）  · 统一计时与统计       │
│  · Coverage / Performance 自动评分  · 波动区间与回归判定           │
│  · 报告生成 · 结果提交 · 榜单发布 · 版本治理与 errata               │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 两条赛道（Track）

| 赛道 | 目标 | 探针 | 主要指标 | case 规模建议 |
| --- | --- | --- | --- | --- |
| **Coverage Track（功能完备性）** | 编译器是否"说得对、编得对、跑得对" | **算子**（ATen 对齐 / 真实框架 / 参数化变体） | 加权算子通过率 `OpScore`、严格通过率 `OpStrictPass`、编译成功率、语义正确率、反投影特性分层通过率（诊断） | ≥ 700 |
| **Performance Track（综合性能）** | 编译器是否"编得快、跑得快" | **热点算子**（真实负载 Top-N + P0 组算子） | 相对基线几何平均加速比 `P`、`fast_p` 通过率、编译时间、代码质量（寄存器/共享内存/指令数） | ≥ 300 |

两条赛道共用**同一批算子探针**与同一套运行基础设施，但**评分口径、重复次数、基线定义不同**，报告中分开呈现、加权合成总分。共用探针的好处是：功能与性能结论可以对齐到同一个算子上（"该算子能跑对但慢 3 倍"），避免出现"功能集和性能集各说各话"。

### 3.3 结果解释边界（规范必须写死的"不做什么"）

1. 评测结果为**特定软硬件组合、特定编译器版本、特定 corpus 版本下的测量值**，不构成对编译器整体优劣的终局判断。
2. **不跨硬件厂商做单一总分排名**；榜单按 (硬件型号, 驱动/运行时版本) 分组展示。
3. Coverage 分数高不代表性能好，反之亦然；两赛道分数不做隐式换算。
4. 不评测 Triton 之外的编译器（除非设立明确的对照赛道）；不评测 LLM 生成能力（那是 TritonBench / KernelBench 的目标）。
5. 性能数字**必须**附带置信区间与波动指标；未达稳定性门禁的结果标注为"测量不稳定"，不进入榜单。

---

### 3.4 评测单元：以算子为探针（面向 triton-ascend）

> 本节是本项目评测模型的**核心定位**：一级评测对象是 **Triton 编译器实现（triton-ascend）**，一级评测单元（probe）是 **算子**。

#### 3.4.1 为什么用算子做探针

1. **算子是编译器能力的天然载体。** 一个真实算子（如 `layer_norm`、`rotary_embedding`、`fused_moe`、`flash_attention`、`softmax`）会同时触发访存路径、归约、dot、控制流、mask 边界、dtype 转换、软件流水与 shared memory 分配等多条编译路径。它对编译器的压力是**组合式**的，远大于孤立的单 API 用例。
2. **算子是需求侧的共同语言。** triton-ascend 的公开路线图本身即以"完成 FlagGems 开源仓重点 Triton 算子适配""完成 vLLM、sglang 开源仓中重点 Triton 算子适配""完善 Atomic / Scan-Sort 类 API、支持非连续访存"作为里程碑。用算子做探针，可与被测方的研发节奏**逐条对齐**，评测结论能直接转化为"还差哪些算子、卡在哪条编译路径"。
3. **算子天然具备双赛道属性。** 同一个算子既能判"能否正确编译并执行正确"（功能完备性），又能测"执行多快"（综合性能），避免功能集与性能集割裂、两套 corpus 互相脱节。
4. **算子具有客观语义锚点。** 以 PyTorch ATen 或真实框架中该算子的既有语义为参考实现，正确性判定有客观依据，不需要人工标注期望输出。
5. **算子失败可直接转化为缺陷。** 算子级失败可自动 minify 成最小复现并提交给编译器团队，评测不止于"打分"，而是直接产生工程价值。

#### 3.4.2 算子与特性原子的关系（关键定位）

```
                    ┌──────────────────────────────┐
    探针（可观测）   │  算子 Operator（一级评测单元）  │
                    └───────────────┬──────────────┘
                                    │ Op-Feature 映射矩阵（静态抽取 + 人工校正）
                    ┌───────────────▼──────────────┐
   诊断（可归因）    │  语言特性原子 FA（二级诊断坐标）│
                    └──────────────────────────────┘
```

- **算子是探针**：直接决定"能不能过、跑多快"，是分数的来源。
- **FA 是诊断坐标**：算子通过/失败经 Op→FA 反投影后，得到特性级能力画像与缺口清单，回答"为什么不过、缺哪块能力"。
- 因此"核心语言特性覆盖 ≥ 90%"这一指标由**算子通过率反投影派生**，而不是对 API 逐项造孤立用例——从根本上避免了 corpus 退化成"语法练习册"。

#### 3.4.3 算子的三层形态

| 形态 | 定义 | 语义锚点 | 用途 |
| --- | --- | --- | --- |
| **T1 ATen 对齐算子** | 算子签名与 PyTorch ATen 对齐（如 `torch.layer_norm`、`torch.softmax`、`torch.addmm`） | PyTorch eager 输出 | 语义无争议，作为**功能完备性主干** |
| **T2 真实框架算子** | 从 FlagGems / vLLM / sglang / Liger-Kernel 等开源仓抽取的真实 Triton 算子 | 该仓库自带的参考实现 / 框架内等价路径 | 保证**真实负载代表性**，是性能赛道主力 |
| **T3 参数化变体算子** | 对 T1/T2 算子施加 dtype / shape / layout / 控制流 / 优化 pattern 变异得到的变体 | 随变体同步变换的参考实现 | 探测**边界与组合**，补足特性覆盖盲区 |

三类算子统一使用同一份 case schema（§6.1.1），仅 `source` 与 `reference` 字段不同。

#### 3.4.4 算子探针矩阵（1000+ case 的构成）

```
case = 算子模板(op) × dtype × shape/layout × 控制流变体 × 优化配置
```

建议配比（总数 ≥ 1000，有效 case ≥ 98%）：

| 来源层 | 算子模板数 | 变体数/算子 | case 数 | 说明 |
| --- | --- | --- | --- | --- |
| T1 ATen 对齐算子（核心） | ≈ 80 | 4–6 | ≈ 380 | 覆盖 ATen 高频算子，按重要性加权 |
| T2 真实框架算子（FlagGems / vLLM / sglang / Liger） | ≈ 120 | 3–5 | ≈ 420 | 真实负载，性能赛道主力 |
| T3 参数化变体与补盲变体 | — | — | ≈ 200 | 由覆盖引导生成，专补 Op→FA 矩阵中的空白格 |
| **合计** | **≈ 200** | — | **≈ 1000** | 每个算子模板至少 3 个独立变体 |

**重要性权重** $\omega_o$（用于加权算子分）：由"真实大模型训推中的调用频次 × 关键路径耗时占比"标定（在若干代表性模型上用 profiler 采集算子调用次数与耗时占比，归一化后作为权重），并叠加算子所属特性层（L1/L2/L3）的系数。权重表随 corpus 版本冻结。

#### 3.4.5 面向 triton-ascend 的算子来源与优先级

| 优先级 | 来源 | 说明 | 建议算子数 |
| --- | --- | --- | --- |
| **P0** | **FlagGems**（跨芯片 Triton 通用算子库，已加入 PyTorch 基金会生态项目体系，支持多类芯片） | triton-ascend 路线图明确以"完成 FlagGems 重点算子适配"为里程碑，与其对齐可让评测结果直接指导研发 | 60–80 |
| **P0** | **vLLM / sglang 中的重点 Triton 算子** | 同上，路线图明确提及；且为推理真实热点（MoE、RMSNorm、RoPE、PagedAttention 相关 kernel） | 30–40 |
| **P1** | **Liger-Kernel / flash-attention 的 Triton 实现** | 训练侧融合算子，覆盖复杂融合与反向 | 15–25 |
| **P1** | **Triton 官方 tutorials（01–06 及性能教程）** | 语义清晰、被广泛认可，适合作**冒烟与基线对照** | 10–15 |
| **P1** | **TritonBench-G 184 个 GitHub 真实算子**（经清洗与规范化后） | 真实世界分布，用于泛化性验证 | 20–30（抽样） |
| **P2** | **ATen 对齐算子（参照 TritonBench-T 思路构造）** | 补齐 ATen 语义覆盖面 | 60–80 |
| **P2** | **参数化合成变体**（覆盖引导生成） | 补盲，特别是边界 shape、非对齐尾块、跨步访存、极端 num_warps/stages | 动态 |
| — | **Held-out 算子集**（不公开内容，只公开分布） | 防特化，占 20% | 见 §4.8 |

> 注：真实算子抽取时需做**规范化处理**——剥离框架特有的运行时依赖（如自定义 autotune 装饰器、缓存、分布式通信），保留纯 kernel 逻辑，并参数化 shape，使其可作为独立 case 运行。抽取过程与规范化 diff 需留痕归档。

#### 3.4.6 昇腾场景下的性能基线定义

对 triton-ascend，基线选择直接决定"性能分数"有无说服力：

| 基线 | 定义 | 用途 |
| --- | --- | --- |
| **B0 跨编译器对照** | 同硬件上**官方 Triton 上游对应版本**（若可用）编译同一算子 | 计算 `Performance` 分数的锚点（§6.3），表达"相对参考编译器" |
| **B1 同会话主基线** | **同机、同软件栈、同一次运行会话内**测得的 PyTorch（昇腾后端）eager/ATen 实现时延 | 计算 `speedup` 与 `fast_p` 的主口径 |
| **B2 厂商库上界参考** | CANN / ACLNN（及 Ascend 算子库）实现的时延 | **仅作"相对硬件可达上限"的说明性指标，不计分**，用于回答"距厂商手写库还有多远" |

关键要求：B1 必须与被测对象**在同一批次内交错测量**（interleaved），禁止使用历史存档值，以消除日间漂移。B2 需在报告中明确标注"仅供参考，不构成评分"。

#### 3.4.7 算子级评分（主口径）

以算子为主口径时，得分从"API 覆盖率"改为"算子通过率 + 反投影特性覆盖"：

- **算子通过率**：$\text{OpPass} = \frac{|\{o \mid \text{all variants}(o) = PASS\}|}{|\{o\}|}$（严格口径），另给宽松口径（≥ 2/3 变体通过）。
- **加权算子分**：$\text{OpScore} = 100 \times \frac{\sum_o \omega_o \cdot r_o}{\sum_o \omega_o}$，其中 $r_o$ 为该算子变体的语义正确率。
- **反投影特性覆盖**：见 §6.2.1，作为**诊断维度**而非主得分。
- **静默错误（severe）**：算子编译通过、运行通过但结果错误——单独高亮，是最高优先级风险信号。

#### 3.4.8 算子失败归因流程（把评测转化为缺陷）

```
算子 FAIL
  → ① 双 oracle 复核（排除参考实现/容差问题）
  → ② 自动最小化（minify）：
       缩小 shape 至最小可复现、固定 dtype、剔除无关分支、
       固定 num_warps/stages、导出最小 kernel 源码
  → ③ 编译链定位：对比 TTIR / TTGIR / LLIR / 最终产物各阶段
       首个出现异常/报错的 pass 即为可疑阶段
  → ④ Op→FA 反投影：列出该算子触及的 FA，取交集定位可疑特性
  → ⑤ 生成 issue 模板：最小复现源码 + IR dump diff + 环境指纹 +
       期望输出 vs 实际输出 + 复现命令
  → ⑥ 归档到 defect 库，标注状态（已提交/已修复/已回归验证）
```

该流程使每轮评测除分数外，额外产出**可直接提交的编译器缺陷清单**——这是本项目对 triton-ascend 最直接的工程价值，建议作为独立 KPI（如"每轮产出 ≥ N 个已确认缺陷"）。

---

## 4. 重难点与针对性方案（核心章节）

### 4.1 重难点一：功能完备性无法直接量化——"覆盖率"定义歧义

**问题**：Triton 没有语言标准文本，"特性覆盖率"缺少权威锚点；若只统计"用了某 API 的 case 通过率"，一个 `tl.load` 满天下的 corpus 会让覆盖率虚高到 95% 以上，却完全没测到 `tl.atomic_cas`、`tl.associative_scan`、`tl.dot_scaled`。

**方案：三层特性原子模型 + 覆盖强度分级 + 权重**

1. **特性原子（Feature Atom, FA）清单**：以 `triton.language` 官方文档的 15 个类别为骨架，把每个 API 拆成**可独立判定的最小特性单元**。拆分原则是"不同实现路径即不同原子"，例如：
   - `tl.load` → FA-LOAD-CONTIG（连续访存）/ FA-LOAD-STRIDED（跨步）/ FA-LOAD-MASKED（带 mask）/ FA-LOAD-BOUNDARY（越界保护）/ FA-LOAD-OTHER（other= 填充）
   - `tl.dot` → FA-DOT-FP16 / FA-DOT-BF16 / FA-DOT-FP8 / FA-DOT-TF32 / FA-DOT-INT8 / FA-DOT-ACC-FP32
   - `tl.atomic_add` → FA-ATOMIC-ADD-SCALAR / FA-ATOMIC-ADD-TENSOR / FA-ATOMIC-ADD-MASKED / FA-ATOMIC-SEM-{ACQ_REL,RELAXED,ACQUIRE,RELEASE} / FA-ATOMIC-SCOPE-{CTA,GPU,SYS}
   - 控制流 → FA-CTRL-IF / FA-CTRL-FOR-STATIC / FA-CTRL-FOR-DYNAMIC / FA-CTRL-WHILE / FA-CTRL-NESTED / FA-CTRL-EARLY-EXIT
   - 跨维度组合 → FA-COMB-{LOAD×REDUCE}、FA-COMB-{DOT×SOFTMAX}、FA-COMB-{ATOMIC×SCATTER} 等"特性对"
2. **分层与权重**：
   - **L1 核心层**（权重 3）：编译器不可用则生态不可用——基础访存、算术、`tl.dot`、归约、`program_id`、常见 dtype、基础控制流。
   - **L2 常用层**（权重 2）：生产算子常见——atomic 常用子集、scan/sort、mask/boundary、layout/permute、随机数、autotune/warp 相关常见 pattern。
   - **L3 扩展层**（权重 1）：前沿/厂商差异大——TMA / tensor_descriptor、warp specialization、mxfp / dot_scaled、inline asm、调试与断言、非连续访存。
   - **L4 厂商专有层**（不计分，单独报告）：后端自定义扩展、私有 intrinsic。
3. **覆盖强度分级（防止"一题覆盖"刷分）**：每个 FA 要求 **≥ 5 个结构不同的独立 case**（以规范化 AST 相似度 < 0.7 判定"结构不同"），强度分三档：
   - `S0` 无覆盖（0 分）
   - `S1` 1–2 个 case（0.5 分）——标注"弱覆盖"
   - `S2` 3–4 个 case（0.8 分）
   - `S3` ≥ 5 个 case 且至少 2 个跨维度组合（1.0 分）
4. **计算公式**（见 §6.2）：`CoverageScore = Σ_a w_a · p_a · s_a / Σ_a w_a · s_a`，其中 `p_a` 为该原子下 case 的语义正确率，`s_a` 为覆盖强度系数。
5. **清单共建与冻结**：FA 清单本身纳入"双方共同确认"流程，v1.0 冻结后只在 errata 流程下增删，任何增删都触发 corpus 版本号 +1 且历史分数不可直接比较（必须重跑）。

**为什么能解决问题**：把"是否用到"变成"用到 × 用得够不够多 × 用得对不对"，同时用分层权重保证核心能力的主导地位，用强度分级杜绝刷分。

---

### 4.2 重难点二：正确性 oracle 的可信性——"谁定义正确"

**问题**：Triton kernel 的"正确输出"由谁给出？用 PyTorch 参考实现，则 PyTorch 语义与 Triton 语义在边界（NaN/Inf、舍入、越界、归约结合序）上可能不一致；用人工标注，1000+ case 不可行；浮点误差使精确比较不可行。

**方案：双 oracle + 确定性约束 + 分级容差 + 争议仲裁**

1. **Oracle A（权威语义参考）**：以 **PyTorch eager（同机、同 dtype）** 为默认参考，另可指定 NumPy float64 高精度参考用于低精度 dtype 的误差标定。参考实现与输入数据均随 case 冻结。
2. **Oracle B（解释器交叉验证）**：使用 **TRITON_INTERPRET 解释器模式**在 CPU 上端到端执行同一 kernel，作为与硬件无关的第二个语义参考。当 A 与 B 不一致时，case 标记为 **ORACLE_DISPUTED**，自动降级为"人工仲裁队列"，**不计入任何得分**（既不判对也不判错）。
3. **确定性约束（写在规范里，作为合法输入规则）**：
   - 禁止依赖未初始化内存；
   - 原子操作只用于**结果可归约到确定值**的场景（如顺序无关的整数求和归约），或使用"排序后比较"的规范化判别；
   - 随机数必须固定 seed 且使用 `tl.random` 系列的可复现路径；
   - 禁止依赖 warp 调度顺序、禁止跨 program 未同步通信；
   - 禁止依赖 `tl.atomic_cas` 竞争结果等本质非确定语义。
4. **分级容差**（按 dtype 与运算类型，规范冻结）：

   | dtype / 场景 | 判据 |
   | --- | --- |
   | fp64 | `rel ≤ 1e-12`，`abs ≤ 1e-12` |
   | fp32 | `rel ≤ 1e-5`，`abs ≤ 1e-6` |
   | tf32（输入标记） | `rel ≤ 1e-3` |
   | bf16 / fp16 | `rel ≤ 2e-2`（相对 fp32 参考），并对归约类放宽至 `5e-2` |
   | fp8 / mxfp | `rel ≤ 1e-1`，且必须附 fp32 参考 |
   | int 全宽度 | 精确相等 |
   | bool / 索引 / argmax-argmin（含并列） | 精确相等；并列索引需 case 显式声明 tie-break 规则，否则判 INVALID |
   | NaN / Inf | 位置与符号严格一致 |
5. **误判审计**：每轮评测随机抽取 3% 的 FAIL case 与 1% 的 PASS case 做人工复核，错误率 > 2% 时触发容差表修订（走 errata）。

**为什么能解决问题**：双 oracle 把"参考实现自身有 bug"的风险暴露出来；确定性约束把"本质不可判定的程序"挡在 corpus 之外；分级容差把浮点问题从"政治问题"变成"参数表"。

---

### 4.3 重难点三：跨硬件 / 跨版本的可比性与公平性

**问题**：同一编译器在 A100 / H100 / MI300 / 昇腾上的性能数字无法直接比较；不同厂商的"参考基线"（cuBLAS / rocBLAS / 厂商库）性能差异巨大；跨版本对比时环境一变结论就翻。

**方案：能力清单协商 + 同机同时段基线 + 分组榜单 + 归一化**

1. **Capability Manifest（能力清单）**：每个参评编译器在注册时提交机器可读的能力清单，声明：支持的目标（sm_90 / gfx942 / Ascend910x 等）、支持的 dtype 子集、支持的 FA 子集（L1–L4）、已知不支持项及其降级行为（编译报错 / 运行时报错 / 静默错误）。
2. **三类结果状态严格区分**（这是公平性的关键）：
   - `UNSUPPORTED`：编译器在 manifest 中声明不支持 → **从分母中剔除**，但**单独计入"能力缺口报告"**，并在报告中显式披露"该编译器有 N 项声明不支持"。
   - `COMPILE_FAIL` / `RUNTIME_FAIL` / `WRONG_RESULT`：声明支持但做不到 → **计入分母并判 0**。
   - `PASS`：通过。
   - 这样"诚实声明不支持"不会被重罚，"假装支持却出错"会被惩罚，杜绝"用 manifest 造假"的反向激励（造假即计入失败）。
3. **性能基线定义（冻结、同机、同时段）**：
   - 主基线 **B1**：**同硬件、同软件栈、同一次运行会话内**测得的 PyTorch eager（ATen）实现时延。
   - 辅基线 **B2**：厂商官方库（cuBLAS / rocBLAS / 厂商算子库）时延，作为"理论上界参考"，**不参与计分**，仅用于报告中的"相对硬件峰值"说明。
   - 关键：**基线必须与被测对象在同一批次内交替测量**（interleaved measurement），而非用历史存档值，以消除日间漂移（温度、驱动状态、机器负载）。
4. **分组榜单**：榜单按 `(硬件型号, 驱动/运行时主版本, corpus 版本)` 三维分组；**组内可排序，组间不可排序**。总榜只呈现"各组第一名"与"各组参与情况"，不做跨组加权。
5. **归一化性能指数（用于跨版本追踪）**：以该编译器在 **v1.0 corpus 上的首轮得分**为 100 分基准，后续版本得分表示为相对变化率，支持"自我进步"叙事而不制造跨厂商对立。

**为什么能解决问题**：把不可比的部分（跨硬件）从"分数"中剥离出去，只保留可辩护的比较（同硬件同基线、同编译器跨版本）；把"不支持"变成一种可声明、可披露、可审计的状态而非隐形失败。

---

### 4.4 重难点四：性能测量噪声与可复现性

**问题**：GPU 计时受 SM 频率动态调节、L2 缓存污染、PCIe / NVLink 状态、邻居进程、温度墙、驱动 JIT 编译缓存、autotune 抖动影响。常见现象：同一 kernel 不同轮次差 10–30%，导致榜单无法自证。

**方案：环境固化 + 测量规程 + 统计门禁 三件套**

1. **环境固化**：
   - GPU：`persistence mode` 开启；`nvidia-smi --lock-gpu-clocks=<base>` 锁定 SM 与显存时钟（AMD 用对应 rocm-smi 手段，Ascend 用对应工具；无法锁频的硬件**必须**在报告中标注 `clock: unlocked` 并放宽门禁）。
   - 隔离：独占节点；禁用 MIG 混跑与共享；关闭无关服务与其他 GPU 进程；固定 CPU 亲和性与 NUMA 绑定。
   - 软件：Docker 镜像以 **digest** 固定；`pip freeze --all` 哈希入指纹；驱动、CUDA/ROCm、torch、triton commit 全部记录。
   - **环境指纹** `env_fp = sha256(canonical_json(硬件/驱动/库/环境变量/镜像 digest))`，写入每条结果记录。指纹不一致的结果**不得**与历史结果比较。
2. **测量规程**（对齐并扩展 `do_bench` 口径）：
   - 冷启动与热态分离：先做 1 次编译 + 3 次 warmup 执行（JIT、指令 cache、TLB 预热），再进入计时阶段。
   - 计时阶段：`warmup = 25ms`、`rep = 100ms`（默认），**每 case 独立重复 R = 5 轮**，轮间执行 **L2 cache flush**（`fast_flush` 语义）+ `torch.cuda.synchronize()`。
   - 建议对短 kernel（< 10μs）启用 **CUDA Graph 采集**以消除 launch 开销与 CPU 抖动，并记录采集方式（`cudagraph: true/false`）。
   - 编译时间单独测量：`compile_time = JIT 编译 + lowering 到最终产物` 的墙钟时间，冷缓存下测 3 次取中位数。
3. **统计门禁**：
   - 单 case 统计量：取 5 轮的中位数 `t_med`，并记录 MAD 与变异系数 `CV = std/mean`。
   - **门禁阈值**：`CV ≤ 3%`（锁频环境）/ `CV ≤ 5%`（无法锁频）。超阈值 → 自动加倍重复至 R=15；仍超 → 标记 `UNSTABLE`，该 case 不计入性能得分（计入覆盖率得分，因为语义是正确的）。
   - 离群处理：使用 **bootstrap（10k 次重采样）** 给出几何平均加速比的 95% 置信区间，而不是简单剔除离群点（剔除会掩盖真实抖动）。
   - **复现性校验**：同一 (编译器, 硬件) 组合在 48h 内跑两轮，几何平均加速比的相对偏差应 ≤ 2%；超标则整轮结果标记为 `REPRO_CHECK_FAILED`。

**为什么能解决问题**：把"噪声"从"靠人自觉"变成"机器可判定的门禁 + 不可通过时的显式降级路径"，同时用环境指纹保证"可比"的前提被机器校验。

---

### 4.5 重难点五：autotune / JIT 缓存带来的系统性偏差

**问题**：同一个 kernel 是否使用 `triton.autotune`、autotune 的 config 搜索空间大小、是否命中磁盘 JIT 缓存、是否预编译（AOT），都会让"编译器性能"变成"调参器性能"或"缓存命中运气"，严重污染评分。

**方案：把 autotune 预算显式化、三态分离、缓存策略固化**

1. **三态分离测量**：每个性能 case 测量三种状态并分别记录：
   - `T_cold`：清空 JIT/磁盘缓存后的首次编译 + 首次执行时间（反映**编译开销**，用于 Compile-Time 指标）。
   - `T_warm_fixed`：使用 case 指定的**固定 config**（由 corpus 冻结，禁止搜索）的执行时间（**主性能口径**）。
   - `T_autotuned`：允许 `triton.autotune` 在**固定预算**内搜索后的最优执行时间（**辅助口径**，搜索时间不计入）。
2. **Autotune 预算（Budget）规范化**：若被测方要求使用 autotune，必须在提交时声明 config 空间与 `warmup/rep` 预算；预算上限由规范规定（如 ≤ 32 configs、单 config ≤ 50ms 测量窗口），超出部分不计分。
3. **默认口径**：**榜单主口径 = `T_warm_fixed`**（固定 config，禁 autotune），理由是"编译器能力"应体现在**给定 tiling 下生成代码的质量**，而非搜索能力；`T_autotuned` 单列为"调优后口径"，供参考。
4. **缓存策略固化**：JIT 缓存目录、AOT 编译开关、`TRITON_CACHE_DIR`、是否启用 `triton-dejavu` 类缓存复用机制——全部写入 run profile 并进入环境指纹；不同缓存策略的结果不可互相比较。
5. **随机化与交错**：多编译器/多 config 的执行顺序**随机交错**（round-robin with random start），避免"先跑的吃亏（冷）后跑的占便宜（热）"。

**为什么能解决问题**：把隐性变量（搜索、缓存、顺序）全部暴露为显式维度，主口径剥离调参器因素，辅助口径保留现实可用性。

---

### 4.6 重难点六：自动生成 1000+ case 的质量控制（无效 / 重复 / 平凡）

**问题**：随机生成很容易产出大量"编译不过的垃圾""结构雷同的孪生 case""毫无意义的平凡 case（如全零 kernel）"，导致 1000 这个数字达标而评测能力不达标。

**方案：四道门禁 + 覆盖引导闭环**

1. **门禁 G1 语法/静态合法**：AST 解析通过；`tl.constexpr` 使用合法；BLOCK 大小为 2 的幂且 ≤ 硬件上限；无 Python 副作用进入 kernel 体。
2. **门禁 G2 编译可执行**：在**参考编译器（基线 Triton 版本）**上必须能编译并通过语义校验。这一条保证 case 是"合法 Triton 程序"而非"生成器臆想"——**参考实现由 Oracle A（PyTorch）+ Oracle B（解释器）双确认**。
3. **门禁 G3 非平凡性**：
   - 输出张量不是常量 / 不恒为零；
   - 至少包含一个非平凡数据流（load → compute → store 全链路）；
   - 归一化 AST 的**结构指纹**不与已有 case 重复（相似度 < 0.7）；
   - 有效 tile 元素数 ≥ 64，避免被优化成空操作。
4. **门禁 G4 覆盖增益**：新 case 至少为某个"覆盖强度未满"的 FA 带来增益，否则拒绝入库（除非进入"真实负载白名单"，如 FlagGems / 真实框架抽取的算子，这类 case 可豁免 G4 但需标注来源）。
5. **覆盖引导闭环**：
   ```
   corpus → 静态抽取 FA 标签 → 计算 (覆盖率, 强度分布)
        → 采样权重 ∝ (目标权重 × 未覆盖缺口 / 当前覆盖度)
        → 生成器按权重采样模板 + 变异（dtype / shape / 控制流 / layout / 优化 hint）
        → 四道门禁 → 新 case → 去重 → 入 corpus → 回到抽取
   ```
   每轮迭代输出覆盖率曲线与 corpus 演化日志（新增/拒绝/拒绝原因分布），可审计。
6. **人工抽样审计**：每 200 个自动生成 case 抽取 20 个由两名工程师独立评审"是否有意义、标签是否正确"，一致率 < 90% 时回炉调整生成器。

**为什么能解决问题**：用门禁保证"每个 case 都合法、非平凡、有覆盖贡献"，用闭环保证 1000 个 case 覆盖在特性空间上是**均匀铺开**而非扎堆。

---

### 4.7 重难点七：测试集"冻结"与"演进"的矛盾

**问题**：规范要求冻结存档以保证可复现，但 corpus 一定会有 bug（错误 oracle、非确定 case、非法 case）。冻结后修 bug 会让历史分数不可比；不修则错误永久化。

**方案：语义化版本 + 冻结快照 + errata 通道 + 双轨榜单**

1. **版本号**：`corpus-v{MAJOR}.{MINOR}[-frozen]`。
   - `MAJOR+1`：FA 清单变更、case 删除、oracle 语义变更、判定规则变更 → **历史分数全部作废，必须重跑**。
   - `MINOR+1`：新增 case（不改动已有 case）→ 历史分数保留但标注"基于 vX.(k-1) 子集"，可与新版本子集对比。
2. **冻结快照**：每个发布版本生成不可变快照（内容寻址存储，SHA-256 manifest），包含：case 文件、输入生成 seed、参考实现、容差表版本、FA 标签表、生成日志、生成器 commit。
3. **Errata 通道**：发现 case 缺陷 → 提交 issue → 技术委员会仲裁 → 三种处置之一：
   - `RETRACT`（撤销该 case，MAJOR+1）
   - `AMEND`（修正 oracle/容差，MAJOR+1，并注明受影响的 case 列表）
   - `ANNOTATE`（仅加注说明，不改判定，MINOR）
4. **双轨榜单**：
   - **冻结轨（Frozen Track）**：只接受 `*-frozen` 版本，结果永久可比，是"官方成绩"。
   - **活跃轨（Active Track）**：跟随 main 分支，用于发现新问题和预览，结果标注 `PRELIMINARY`，不上总榜。

**为什么能解决问题**：把"冻结"与"演进"从二选一变成两条并行轨道，历史可比性与纠错能力同时保留。

---

### 4.8 重难点八：评分被针对性优化 / 榜单作弊

**问题**：一旦有榜单，就会出现"针对测试集特化编译器"（hard-code pattern、针对 case 加 fast path），导致分数虚高而真实能力未提升。

**方案：held-out 集 + 可复现提交 + 抽检复跑 + 反特化规则**

1. **公开集 / 保留集二分**：1000+ case 中约 80% 公开（含完整内容与标签），**约 20% 为 held-out**（只公开覆盖标签分布、不公开内容），held-out 结果由运营方在可信环境跑，单独出"泛化得分"。公开集与 held-out 集得分差距过大（如公开 > held-out 20%）→ 触发"疑似特化"标注。
2. **提交必须可复现**：提交物不是"分数"，而是 `Dockerfile/镜像 digest + 编译器构建脚本 + run profile + 提交者签名`，由 CI 自动复跑后**自动生成分数**，人工不得填写分数。
3. **抽检复跑**：每月随机抽取 ≥ 10% 上榜条目在独立硬件上复跑，偏差 > 3% 的条目进入复核，两次复核失败则下榜并公示原因。
4. **反特化规则**（规范中明确禁止并声明检测手段）：
   - 禁止按 case id / case 名称 / 输入具体数值做分支；
   - 禁止在编译器中内置 corpus 指纹查表；
   - 检测手段：对 case 做**语义等价变异**（改 shape 数值、改变量名、等价表达式重写、等价指令替换）后重跑，分数显著下降者判定为特化。
5. **透明化**：所有原始日志、环境指纹、产物签名公开可下载，接受社区复核。

**为什么能解决问题**：作弊成本被抬高到"必须同时骗过 held-out 集、语义变异集和独立复跑"，实际不可行。

---

### 4.9 重难点九：多后端支持度差异导致 case 大面积失败

**问题**：如 Triton-Ascend 的公开路线图显示，Atomic、Scan/Sort、非连续访存等能力是分阶段补齐的。若 corpus 直接以最新上游 Triton 为准，早期后端会大面积失败，评测失去指导意义；若以最小公倍数为准，评测失去区分度。

**方案：分层 FA + 能力协商 + 分层报告 + 能力缺口清单**

1. FA 分层（§4.1 的 L1–L4）直接对应"能力阶梯"：L1 是准入线（任何参评编译器应达到），L2 是成熟度线，L3 是先进性线，L4 是厂商专有。
2. 报告按层分别给分，并生成**能力缺口清单（Capability Gap List）**：明确列出"该编译器在 L2 层未通过的原子"，这比一个总分对开发更有指导价值。
3. **降级规则**：对声明不支持的 FA，编译器可选择"编译期明确报错（推荐）"或"运行时报错"，二者计 0 分但**不额外惩罚**；若"静默产生错误结果"，则触发**严重缺陷标记（SEVERE）**，在报告中单独高亮（静默错误比显式失败危险得多）。
4. 允许"部分通过"：`COMPILE_OK_NUM_WARN`（编译通过但输出触发 NaN/Inf 异常检测）等中间状态单独统计，作为质量信号。

**为什么能解决问题**：评测从"总分排名"变成"能力画像 + 缺口清单"，对不同成熟度的后端都具有指导意义，且对危险行为（静默错误）有强信号。

---

### 4.10 重难点十：执行规模与成本控制

**问题**：1000 case × 多编译器 × 多硬件 × 多次重复，执行成本高，容易陷入"跑一次几天"的困境。

**方案：分层执行 + 缓存复用 + 增量评测 + 预算模型**

1. **三级执行档位**：
   - `smoke`：随机抽样 5% case + 全部 L1 原子 case，< 5 min，用于 PR 门禁。
   - `standard`：全量功能 case（1 次执行）+ 性能 case 3 轮，约 20–30 min，用于 nightly。
   - `official`：全量 + 5 轮 + 双基线 + 48h 复跑校验，用于榜单发布（weekly/monthly）。
2. **JIT 缓存复用**：同 (编译器版本, case, config) 的编译产物按内容哈希缓存于 artifact 存储，跨轮次复用；缓存命中率作为 CI 指标监控。
3. **增量评测**：corpus 变更 → 只重跑受影响 case；编译器版本变更 → 全量重跑但可复用基线测量（基线在无变更时可缓存，但需每周强制刷新一次并做漂移校验）。
4. **调度**：case 级并行 + GPU 级并行 + 节点级并行（Ray / Slurm / K8s Job），单 case 设置硬超时（默认 60s）与内存上限，防止 kernel hang 拖垮整轮。
5. **预算模型**（写进运营计划）：以"每周一次 official × 3 硬件 × 5 编译器"为基准，估算机时与成本，并在平台中展示"本轮消耗机时"。

---

### 4.11 重难点十一：结果可追溯（Provenance）

**问题**：榜单上一个数字，如何证明它是"那台机器、那个版本、那个 corpus、那次运行"产出的？

**方案：全链路签名链**

1. 每次运行生成 `run_id`，并记录：
   - `env_fp`（环境指纹）、`corpus_manifest_hash`、`spec_version`、`scorer_version`、调度器版本；
   - 每个 case 的：输入 seed、编译产物哈希、原始 stdout/stderr、计时原始样本数组（不只要中位数）、退出状态、错误信息。
2. 结果记录整体做 **canonical JSON → SHA-256 → 运营方私钥签名**；产物（PTX/cubin/HSACO、日志）上传对象存储并记录 hash。
3. 榜单条目展示 `run_id`，可一键下钻到：环境详情 → case 列表 → 单 case 日志与计时样本 → 产物下载。
4. 所有 hash 与签名写入 append-only 的审计日志（可用 git 仓库 + signed tag，或数据库 + 只追加表）。

---

### 4.12 重难点十二：持续运营（避免交付即停摆）

**问题**：平台建好后无人提交、无人治理，一年后成为死链。

**方案：制度化 + 自动化 + 激励**

1. **发布节奏**：corpus 季度发布 MINOR、年度发布 MAJOR；榜单月度更新；报告季度发布；规范年度复审。
2. **组织机制**：技术委员会（规范与争议仲裁）+ 运营组（复跑与发布）+ 贡献者社区（case 提交与缺陷报告）。争议仲裁有明确 SLA（如 10 个工作日）。
3. **贡献激励**：case 采纳计入贡献榜、缺陷报告致谢、编译器改进与分数提升联动展示（"某 issue 提出后，X 编译器在 FA-Y 上由 FAIL 转 PASS"）。
4. **降低参与门槛**：提供 `docker-compose up` 一键本地跑通 + `submit.py` 一键打包提交 + CI 模板，让外部团队 30 分钟内能跑出第一份结果。
5. **KPI**：首年目标建议为 ≥ 3 家编译器 / ≥ 2 类硬件 / ≥ 4 次榜单更新 / ≥ 20 个社区提交的 case 缺陷报告。

### 4.13 重难点十三：算子集合的选择偏差与代表性

**问题**：用算子做探针后，首要难题变成"选哪些算子"。选偏了，覆盖率数字好看却不能代表编译器真实能力（例如全是 elementwise，测不到 dot/atomic/scan）；选全了，算子数量与维护成本失控；选了冷门算子，被测方会质疑评测不公平。

**方案：三层抽样 + 频次加权 + 社区提名 + 保留集校验**

1. **三层抽样构造算子池**：
   - **ATen 语义层**（T1）：保证算子空间的**正交覆盖**——按 ATen 算子分类（elementwise / reduction / normalization / matmul / attention / 索引与 scatter-gather / 随机数 / 类型转换 / 激活与融合）分层抽样，每层保证最低算子数。
   - **真实负载频率层**（T2）：在代表性大模型（如若干主流 LLM 的训推）上用 profiler 采集 Triton/ATen 算子的**调用次数与耗时占比**，取 Top-N 热点算子入库。这一层保证"评的是真正影响业务的东西"。
   - **特性补盲层**（T3）：扫描 Op→FA 矩阵，对尚未被任何算子覆盖的 FA 格子，定向构造或变异出算子变体补盲。
2. **频次加权**：算子权重 $\omega_o$ 由"调用频次 × 耗时占比"标定并归一化（§3.4.4），冷门算子仍入库但权重低，既保证覆盖又不主导分数。权重表随 corpus 冻结，且**在报告中完整公开**。
3. **社区提名与异议通道**：被测方与社区可提名算子入库，也可对"某算子不具代表性"提出异议，由技术委员会裁决。提名记录与裁决结果公开。
4. **保留集（held-out 算子）校验**：公开集分数与保留集分数差距 > 20% 判定为"疑似特化"（§4.8），这是检验算子集合是否被针对性优化的手段。
5. **定期复盘**：每半年用新的真实负载分布重新校验算子池的频率代表性，偏差大的算子走 MINOR 版本调整。

**为什么能解决问题**：把"选哪些算子"从主观决策，变成"正交覆盖 + 真实频次 + 特性补盲"三重客观约束下的可审计过程。

---

### 4.14 重难点十四：算子语义契约（Operator Contract）与判定争议

**问题**：算子天然带语义，但 Triton 实现与 ATen 参考在细节上常有分歧，且这些分歧**往往不是编译器 bug 而是约定差异**：
- `layer_norm`：eps 加在方差内还是外？归一化轴的 dtype 提升规则？返回 mean/rstd 是否？
- `softmax` / `log_softmax`：是否先减最大值？fp16 输入的累加 dtype？
- `rotary_embedding`（RoPE）：不同框架有不同旋转约定（相邻对 vs 半分 vs GPT-NeoX 风格）、不同位置编码基数。
- `dropout` / 随机算子：掩码生成方式不可比。
- `scatter_add` / `index_add`：重复索引处的累加顺序（本质非确定）。
- 广播规则、in-place 与别名、stride 输出布局、NaN/Inf 传播、整数除法的舍入方向。
若不定义契约，正确性判定会陷入无休止争议，且容易把"约定差异"误判为"编译器缺陷"。

**方案：契约先行、契约与本体分离、争议走 errata**

1. **为每个算子编写 Operator Contract（机器可读 + 人读双份）**，字段包括：
   - 输入/输出签名（shape、dtype、stride 期望、是否允许 in-place、是否允许别名）
   - 广播与类型提升规则
   - 归约轴与 keepdim 语义
   - 数值约定（eps 位置、累加 dtype、舍入模式、减最大值等稳定化策略）
   - NaN / Inf / 0×∞ 的传播规则
   - 随机性处理（固定 seed、掩码由输入 seed 决定、或改用确定性等价算子）
   - 并列索引的 tie-break 规则
   - 参考实现来源（ATen 具体版本 / 某个开源仓的具体 commit）
2. **契约一致性测试（Contract Conformance Test）与算子本体分离**：先跑一组"契约探测用例"确认被测实现与契约一致（例如显式检查 eps 位置），不一致则判定为 `CONTRACT_MISMATCH` 而非 `WRONG_RESULT`，并在报告中单列——**这一项极其重要**：它能区分"编译器算错了"与"两边约定不同"，避免误伤被测方、也避免掩盖真实缺陷。
3. **随机性算子处理规范**：一律固定 seed 并要求掩码可由 seed 复现；无法确定的（如竞争性 atomic）按 §4.2 的确定性约束排除，或改用顺序无关的归约判据。
4. **争议通道**：`CONTRACT_MISMATCH` 与 `WRONG_RESULT` 的界定由技术委员会仲裁，裁决结果写入 contract 版本号（`contract-vX.Y`）并触发 corpus 对应条目更新，历史结果按 §4.7 的双轨规则处理。
5. **契约版本入环境指纹**：每条结果记录 `contract_version`，不同契约版本的结果不可比较。

**为什么能解决问题**：把"正确性"从"和 PyTorch 一模一样"升级为"符合一份公开、可版本化、可仲裁的契约"，既保住客观性，又消除了约定差异带来的误判。

---

### 4.15 重难点十五：算子级失败的可归因性与缺陷转化

**问题**：如果评测只输出"算子 X FAIL"，对 triton-ascend 团队的价值非常有限——他们需要的是"为什么 FAIL、最小复现是什么、卡在哪个 pass"。缺乏归因能力是本类评测最常见的失败模式（被测方无法据此改进，最终放弃参与）。

**方案：自动 minify + 编译链定位 + 反投影 + 一键 issue**（流程见 §3.4.8）

1. **自动最小化（minify）**：
   - shape 二分收缩（保持失败可复现的前提下最小化每个维度）；
   - 分支/语句删除（借鉴 C-Reduce 的 delta debugging 与 Hypothesis 的 shrinking）；
   - 固定 `num_warps` / `num_stages` / BLOCK 配置，消除调优维度；
   - 输出：**最小可复现 kernel 源码 + 最小输入 + 固定配置**，通常能从数百行真实算子缩到 20–40 行。
2. **编译链阶段定位**：对每个阶段（TTIR / TTGIR / LLIR / 最终二进制/PTX 类产物）做 dump 与 diff：
   - 编译报错 → 记录报错 pass 名与报错位置；
   - 结果错误 → 逐阶段 dump 与参考编译器的同阶段 IR 做结构化 diff，首个出现语义分歧的 pass 为可疑阶段；
   - 这一步对跨编译器对照尤其有效（有 B0 上游 Triton 作为参照）。
3. **Op→FA 反投影**：算子触及的 FA 集合 ∩ 同阶段失败的其它算子所触及的 FA 集合 → 交集即为最可疑特性，可直接给出"疑似 `tl.atomic_*` 在非连续访存下 lowering 错误"这类结论。
4. **一键 issue 生成**：产出含"最小复现源码 + 复现命令 + IR diff + 环境指纹 + 期望/实际输出"的 Markdown，可直接粘贴到 triton-ascend 仓库。
5. **缺陷生命周期管理**：defect 库记录 `已提交 → 已确认 → 已修复 → 回归验证通过` 的状态流转；修复后自动在新一轮评测中回归验证，并在报告中展示"本轮由评测发现并已修复的缺陷数"。
6. **KPI 化**：建议把"每轮评测产出 ≥ N 个经被测方确认的缺陷"作为项目验收的加分项，确保评测不停留在打分层面。

**为什么能解决问题**：把评测从"裁判"扩展为"诊断工具"，被测方有强动机持续参与，运营可持续性（§4.12）也随之解决。

---

## 5. 技术栈选型

### 5.1 选型总表

| 层次 | 用途 | 选型 | 理由 | 备选 |
| --- | --- | --- | --- | --- |
| **语言/运行时** | 全部工具链 | **Python 3.11+**（`pyproject.toml` + `uv` 或 `pip` + hash-pinned lockfile） | Triton 生态原生语言；生成器需直接操作 Triton AST/IR | — |
| **规范表达** | 机器可读规范 | **YAML + JSON Schema + Pydantic v2** | 规范要能被代码直接校验与消费，而不是躺在 Word 里 | Protobuf（过重） |
| **测试集生成** | 模板化 kernel 生成 | **Jinja2 模板 + `libcst` 源码变换** | Jinja2 控制结构，`libcst` 做保格式 AST 改写与规范化（用于去重/变异） | 纯字符串拼接（不可靠）、`ast`（丢格式） |
| **参数化采样** | dtype/shape/layout 组合 | **Hypothesis**（策略化生成 +  shrinking） | 成熟的 property-based 生成与失败用例最小化能力 | 自研随机（无 shrinking） |
| **覆盖引导** | 覆盖率反馈与调度 | **自研 `coverage-engine`（Python）+ NetworkX（特性依赖图）** | 需自定义"特性原子覆盖增益"目标函数，通用 fuzzer 不适用；依赖图用于保证组合可行性 | AFL++/libFuzzer（面向二进制，不匹配 DSL） |
| **去重** | case 相似度 | **MinHash + LSH（`datasketch`）+ 规范化 AST 哈希** | 1000+ 规模下近似去重足够，成本可控 | 精确图同构（太慢） |
| **算子抽取** | 从真实仓构造探针 | **`libcst`/`ast` 静态分析 + Jinja2 规范化模板 + Git commit 锚定**；配合 **PyTorch Profiler / Kineto** 采集真实模型中的算子调用频次与耗时占比 | 保证探针来自真实负载且可回溯到源仓；频次用于加权 | 手工抄算子（不可维护） |
| **算子契约** | 语义判定依据 | **Operator Contract（YAML + 人读 MD）+ 契约一致性探测用例** | 分离"约定差异"与"编译器缺陷" | 直接比 ATen（争议不断） |
| **失败归因** | 缺陷转化 | **Delta debugging 自研 minify（shape 二分 + 语句删除）+ IR 逐阶段 dump/diff + Op→FA 反投影** | 产出可直接提交的 issue | 仅报告 FAIL（无工程价值） |
| **语义 Oracle** | 参考实现 | **PyTorch eager（主）+ NumPy float64（标定）+ `TRITON_INTERPRET`（交叉）** | 权威 + 高精度 + 硬件无关三方交叉 | 人工标注（不可扩展） |
| **执行调度** | 单机/集群执行 | **pytest 作为 case 执行器 + Ray（本地集群）/ Slurm 或 K8s Job（生产）** | pytest 天然支持参数化、超时、报告；Ray 上手快，Slurm/K8s 对接现有算力平台 | 自研 runner（重复造轮子） |
| **并行隔离** | GPU 资源分配 | **Docker / Podman + NVIDIA Container Toolkit（或 ROCm / Ascend 对应运行时）** | 环境可固化、可分发；镜像 digest 进入环境指纹 | Conda env（隔离不彻底） |
| **环境固化** | 可复现环境 | **Dockerfile 多阶段构建 + `pip install --require-hashes` + 镜像 digest 固定**；进阶用 **Nix flake** | digest 是最强锚点；Nix 适合追求极致复现时 | 仅 requirements.txt（不可复现） |
| **性能测量** | 计时核心 | **`triton.testing.do_bench` 增强封装**（warmup/rep/quantiles/fast_flush）+ CUDA Events + **CUDA Graph**（短 kernel）+ `torch.cuda.synchronize` | 与社区口径兼容，便于对照；增强部分自研 | 自研计时器（口径不被认可） |
| **环境控制** | 降噪 | **`nvidia-smi` 锁频（persistence mode）**；AMD/Ascend 对应工具；DCGM 采集温度/功耗/时钟 | 锁频是降噪最有效手段 | 只做多次重复（治标） |
| **深度剖析** | 性能归因（可选） | **NVIDIA Nsight Compute（NCU）** / ROCm rocProf / **Triton Proton**；IR dump（TTIR/TTGIR/LLIR/PTX/cubin） | 归因到指令、寄存器、共享内存、occupancy | 只看端到端时间 |
| **指标存储** | 结果数据 | **Parquet（原始样本）+ DuckDB（分析）+ PostgreSQL（元数据/榜单）** | Parquet 存原始样本成本低；DuckDB 免运维做分析；PG 承担事务 | 全套 PG（原始样本太占空间） |
| **产物存储** | 日志/编译产物 | **S3 兼容对象存储（MinIO 自建或云 S3）**，内容寻址（hash 作为 key） | 天然去重 + 防篡改 | 本地文件系统（不可扩展） |
| **版本/审计** | corpus 与结果追溯 | **Git + signed tag（corpus/spec）+ 数据库 append-only 审计表 + SHA-256 签名** | Git 天然适合文本 corpus 的版本与签名 | 仅数据库（无 diff 能力） |
| **后端服务** | 平台 API | **FastAPI（Python）+ SQLAlchemy + Alembic + Celery/Ray 异步任务** | 与工具链同语言，减少上下文切换 | Django（过重）、Go（团队切换成本） |
| **前端** | 榜单/报告站 | **Next.js（React + TypeScript）+ Tailwind + ECharts**；轻量替代 **Gradio / Streamlit** | 榜单需要 SEO、路由、复杂交互；内部看板可用 Gradio 快速出 | 纯静态 HTML（难维护） |
| **可视化** | 图表 | **ECharts**（雷达图用于能力画像、箱线图用于波动、折线用于版本趋势） | 雷达图/箱线图支持好 | Matplotlib 静态图（无交互） |
| **CI/CD** | 自动化 | **GitHub Actions（或 GitLab CI）+ pre-commit + 自建 GPU Runner** | 与贡献流程天然集成 | Jenkins（维护成本高） |
| **实验追踪** | 运行记录 | **MLflow** 或 **Weights & Biases**（离线模式可选） | 记录参数、指标、产物，UI 现成 | 自研数据库表 |
| **可观测性** | 平台运维 | **OpenTelemetry + Prometheus + Grafana** | 标准方案 | — |
| **文档** | 规范与报告 | **Markdown + MkDocs Material**（或 Docusaurus），规范页由 YAML 自动生成 | 文档即代码，规范与机器可读版本同源 | Word（不可版本化） |
| **报告生成** | 标准化报告 | **Jinja2 模板 + Pandoc（PDF/Word）+ Quarto（可选）** | 一键产出 PDF/HTML | 人工编写（不可持续） |

### 5.2 推荐的最小可行技术栈（MVP，3 个月内可跑通）

若需控制首期投入，建议 MVP 只保留：

```
Python 3.11 + uv           # 工具链
YAML/JSON Schema/Pydantic  # 规范 + 算子契约 + 数据格式
libcst + Jinja2            # 真实算子抽取与规范化（探针来源）
Jinja2 + libcst + 自研覆盖引擎   # 变体生成与 Op→FA 补盲
PyTorch + TRITON_INTERPRET # 双 oracle
pytest + Ray + Docker      # 执行
do_bench 增强封装           # 计时
Parquet + DuckDB + MinIO   # 存储
FastAPI + Next.js 或 MkDocs # 展示（首期可用静态站点 + DuckDB WASM 直连 Parquet）
GitHub Actions             # CI
```

**首期可省略**：PostgreSQL、Celery、Prometheus/Grafana、NCU 深度剖析、Nix。待榜单进入常态化运营后再补齐。

### 5.3 仓库结构建议

```
triton-eval/
├── spec/                       # 规范层
│   ├── spec.yaml               # 机器可读规范（版本、边界、门禁、容差、权重）
│   ├── feature_atoms.yaml      # FA 清单（分层 + 权重 + 强度要求）
│   ├── exclusion_rules.md      # 合法输入与排除规则（人读）
│   ├── run_rules.md            # 运行规则（人读）
│   └── templates/              # 报告/提交模板
├── generator/                  # 生成层
│   ├── templates/              # kernel 模板库（Jinja2）
│   ├── mutate/                 # libcst 变异算子
│   ├── coverage_engine.py      # 覆盖引导闭环
│   ├── gates.py                # G1–G4 门禁
│   ├── dedup.py                # MinHash/AST 去重
│   └── oracle.py               # 双 oracle 与容差判定
├── corpus/                     # 冻结测试集（算子探针）
│   ├── operators/              # 算子目录，一算子一目录
│   │   └── rms_norm/
│   │       ├── op.yaml         # 算子契约 + 权重 + FA 标签
│   │       ├── kernel.py       # Triton 实现
│   │       ├── ref.py          # 参考实现（ATen / 源仓）
│   │       └── variants/       # 变体（dtype × shape × layout × config）
│   │           └── fp16_4096_strided.yaml
│   ├── op_feature_matrix.yaml  # Op→FA 映射矩阵
│   ├── v1.0-frozen/            # 快照（内容寻址）
│   ├── manifest.json           # case 索引 + 标签 + hash
│   └── CHANGELOG.md            # corpus 演化记录
├── runner/                     # 运行层
│   ├── harness.py              # 单 case 执行器（pytest 插件）
│   ├── bench.py                # do_bench 增强封装
│   ├── envprobe.py             # 环境指纹采集
│   ├── scheduler.py            # Ray/Slurm 调度
│   └── profiles/               # run profile（smoke/standard/official）
├── scorer/                     # 评分
│   ├── coverage_score.py
│   ├── perf_score.py
│   └── report.py
├── platform/                   # 平台
│   ├── api/                    # FastAPI
│   ├── web/                    # Next.js
│   └── submit/                 # 提交模板与 CLI
└── docs/
```

---

## 6. 评测体系技术细节

### 6.1 统一数据格式

#### 6.1.1 Case Schema（每个 case 一个目录）

```yaml
# case.yaml
id: FA-LOAD-STRIDED-0042
version: "1.0"
track: coverage            # coverage | performance | both
source: generated          # generated | realworld(FlagGems/vLLM/sglang) | manual
seed: 20260929             # 输入数据生成种子
feature_atoms:             # 覆盖标签（生成时自动抽取 + 人工抽检校正）
  - FA-LOAD-STRIDED
  - FA-COMB-LOAD×REDUCE
  - FA-CTRL-FOR-DYNAMIC
  - FA-DTYPE-FP16
layers: [L1, L2]           # 所属特性层
kernel: kernel.py          # Triton kernel 源码
entry: add_strided         # 入口函数名
inputs:                    # 输入规格（生成器据此造数据）
  - name: x
    shape: [1024, 2048]
    dtype: float16
    layout: contiguous     # contiguous | strided | transposed | non_contiguous
    dist: normal
    seed: 20260929
  - name: out
    shape: [1024]
    dtype: float32
    init: zeros
reference:                 # Oracle A
  kind: torch
  code: ref.py
  dtype_cast: float32      # 参考计算精度
tolerance:                 # 覆盖默认容差表
  rel: 2.0e-2
  abs: 1.0e-4
  nan_inf: strict
determinism:
  atomic_order_independent: true
  tie_break: lowest_index
launch:                    # 固定 config（性能主口径）
  grid: ["cdiv(1024, 128)"]
  block_config: {BLOCK_M: 128, BLOCK_N: 128, num_warps: 4, num_stages: 3}
timeout_s: 60
tags: [memory, strided, reduction]
meta:
  generator_commit: a1b2c3d
  generated_at: "2026-09-01T10:00:00Z"
  reviewed_by: null        # 人工抽检后填写
```

#### 6.1.2 Run Profile（运行规则快照）

```yaml
# profiles/official.yaml
profile: official
spec_version: 1.0
corpus: v1.0-frozen
repeats: 5                      # 每 case 计时轮数
warmup_ms: 25
rep_ms: 100
l2_flush: true
use_cuda_graph: auto            # kernel < 10us 时启用
interleave: random              # 多对象交错顺序
clock_lock: required            # required | best_effort | off
cv_gate: 0.03
unstable_action: escalate       # escalate(加重复) | drop
baseline:
  primary: torch_eager_same_session
  secondary: vendor_lib         # 仅报告，不计分
autotune: disabled              # 主口径禁 autotune
jit_cache: {enabled: true, dir: /cache/triton, key: content_hash}
timeout_s: 60
max_parallel: 8
```

#### 6.1.3 Result Record（每条结果）

```json
{
  "run_id": "20260929-official-ascend910b-triton-ascend3.2.1-0001",
  "submitted_at": "2026-09-29T02:11:03Z",
  "env_fp": "sha256:9f2c...",
  "env": {
    "gpu": "Ascend 910B", "driver": "24.1.0", "runtime": "CANN 8.x",
    "python": "3.11.9", "torch": "2.6.0+xxx", "triton": "3.2.1+ascend(commit abc1234)",
    "image_digest": "sha256:1a2b...", "clock_locked": true,
    "sm_clock_mhz": 1400, "mem_clock_mhz": 1600, "temperature_c": 41
  },
  "compiler": {"name": "triton-ascend", "version": "3.2.1", "commit": "abc1234",
               "capability_manifest_hash": "sha256:...", "build_script_hash": "sha256:..."},
  "corpus": {"version": "v1.0-frozen", "manifest_hash": "sha256:..."},
  "spec_version": "1.0", "scorer_version": "1.0.3",
  "case_id": "FA-LOAD-STRIDED-0042",
  "status": "PASS",
  "compile": {"status": "OK", "cold_ms": 1843.2, "warm_ms": 12.5,
              "artifacts": {"ttgir": "sha256:...", "bin": "sha256:..."}},
  "correctness": {"oracle_a": "PASS", "oracle_b": "PASS",
                  "max_rel_err": 3.1e-3, "max_abs_err": 8.2e-6, "nan_inf_match": true},
  "perf": {
    "samples_ms": [0.412, 0.409, 0.415, 0.410, 0.411],
    "median_ms": 0.411, "cv": 0.0055, "mad": 0.0015,
    "baseline_median_ms": 1.204,
    "speedup": 2.928,
    "ci95": [2.90, 2.95],
    "cudagraph": true
  },
  "log_uri": "s3://triton-eval/runs/.../FA-LOAD-STRIDED-0042.log",
  "signature": "ed25519:..."
}
```

### 6.2 Coverage Score（功能完备性得分）

本项目采用**双口径**：**算子口径为主得分**（对应"以算子为探针"的定位），**特性反投影口径为诊断维度**。

#### 6.2.1 算子口径（主得分）

设算子集合 $O$，算子 $o$ 的重要性权重 $\omega_o$（由真实负载调用频次 × 耗时占比标定，随 corpus 冻结，§3.4.4），算子 $o$ 的变体集合 $V_o$，$V_o^{eff}$ 为其中未被判为 `UNSUPPORTED` 的变体（依据 capability manifest）。

变体通过率与算子通过率：

$$
r_o = \frac{|\{v \in V_o^{eff} \mid \text{status}(v) = \text{PASS}\}|}{|V_o^{eff}|},
\qquad
q_o = \mathbb{1}\big[\, \forall v \in V_o^{eff},\ \text{status}(v) = \text{PASS} \,\big]
$$

$$
\boxed{\text{OpScore} = 100 \times \frac{\sum_{o \in O} \omega_o \, r_o}{\sum_{o \in O} \omega_o}}
\qquad
\boxed{\text{OpStrictPass} = \frac{\sum_{o \in O} \omega_o \, q_o}{\sum_{o \in O} \omega_o}}
$$

- `OpScore`（加权变体通过率）→ **作为 Coverage 主得分输入**（宽容口径，反映整体能力）。
- `OpStrictPass`（加权全变体通过率）→ 作为**严格口径**单独披露，反映能力稳定性（"能不能在所有变体下都对"）。
- 二者差距过大说明编译器存在**边界敏感**问题，应在报告中提示。

#### 6.2.2 特性反投影口径（诊断维度）

**符号**：特性原子集合 $A$，权重 $w_a \in \{3,2,1\}$，覆盖强度系数 $s_a \in \{0,0.5,0.8,1.0\}$。

对原子 $a$，令 $C_a$ 为标签包含 $a$ 的 case 集合，$C_a^{eff}$ 为其中**未被判为 UNSUPPORTED** 的 case（依据被测编译器的 capability manifest）：

$$
p_a = \frac{|\{c \in C_a^{eff} \mid \text{status}(c) = \text{PASS}\}|}{|C_a^{eff}|}
$$

$$
\boxed{\text{Coverage} = 100 \times \frac{\sum_{a \in A} w_a \, s_a \, p_a}{\sum_{a \in A} w_a \, s_a}}
$$

**辅助指标（必须同时披露）**：

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| `compile_rate` | 编译成功 case 占比 | 区分"编不过"与"编错" |
| `semantic_rate` | 编译成功者中语义正确占比 | 衡量正确性而非可用性 |
| `severe_count` | 静默错误结果（编译通过、运行通过、结果错误）的 case 数 | **最高优先级风险信号** |
| `unsupported_rate` | 声明不支持而被剔除的 case 占比 | 诚实披露 |
| `L1_pass` / `L2_pass` / `L3_pass` | 分层通过率 | 能力画像雷达图 |
| `coverage_breath` | 被 ≥1 个有效 case 覆盖的 FA 占 FA 总数的比例 | 对应"≥90% 清单覆盖"的原始诉求 |

**算子级辅助指标（必须同时披露）**：

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| `OpScore` / `OpStrictPass` | 加权变体通过率 / 加权全变体通过率 | 主得分与稳定性对照 |
| `op_pass_top_hot` | 真实负载 Top-20 热点算子的通过率 | 直击业务影响面 |
| `contract_mismatch_count` | 语义契约不一致（非编译缺陷）的算子数 | 区分"约定差异"与"编译器 bug"（§4.14） |
| `severe_op_list` | 静默错误（编译运行均通过、结果错误）的算子清单 | 最高优先级风险 |
| `defect_yield` | 本轮产出并已被确认的编译器缺陷数 | 评测的工程价值（§4.15） |

**建议口径调整**：把原规划的"覆盖比例 ≥ 90%"落地为三个可验收指标：
- **测试集侧**：`corpus_breath ≥ 95%`（FA 被 ≥1 个有效算子变体覆盖的比例）
- **编译器侧（主）**：`OpScore` 算子加权通过率（这才是完备性评价）
- **编译器侧（诊断）**：反投影得到的 `L1/L2/L3` 分层特性通过率与能力缺口清单

### 6.3 Performance Score（综合性能得分）

**主口径**：固定 config、禁 autotune、同机会话内基线。

单 case 加速比：$S_i = t^{baseline}_i / t^{impl}_i$（取各自 5 轮中位数）。

$$
P_{\text{geo}} = \exp\left(\frac{1}{N}\sum_{i=1}^{N}\ln S_i\right) \quad(\text{几何平均，对比例型指标无偏})
$$

$$
\text{fast}_p = \frac{1}{N}\left|\{i \mid \text{correct}(i) \land S_i \ge p\}\right|,\quad p \in \{1.0, 1.2, 2.0\}
$$

$$
\boxed{\text{Performance} = 100 \times \frac{\ln P_{\text{geo}}}{\ln P^{\text{ref}}_{\text{geo}}}}
$$

其中 $P^{\text{ref}}_{\text{geo}}$ 为**基准锚点**：建议取"同一硬件上官方 Triton 上游对应版本"的几何平均加速比作为 100 分锚点（首轮标定后冻结）。这样分数表达为"相对参考编译器的对数加速比比例"，避免绝对数值在不同硬件上失去意义。

**辅助指标**：

| 指标 | 说明 |
| --- | --- |
| `compile_time_p50` / `p95` | 冷编译时延，反映编译开销 |
| `code_quality` | 从产物提取：寄存器数、共享内存用量、指令数、occupancy、PTX/SASS 级别向量化率（如 128-bit 访存占比） |
| `unstable_rate` | CV 超标被剔除的性能 case 占比 |
| `autotuned_geo` | 允许的 autotune 预算内的加速比（辅助口径） |

### 6.4 总分合成与波动处理

$$
\text{Total} = 0.40 \times \text{Coverage} + 0.45 \times \text{Performance} + 0.15 \times \text{Stability}

其中 $\text{Coverage}$ 取 **§6.2.1 的算子口径 $\text{OpScore}$**（算子探针主口径）；$\text{OpStrictPass}$ 与特性反投影的 `L1/L2/L3` 分层通过率作为诊断维度**单独披露，不参与总分合成**，避免"覆盖率"与"算子通过率"双重计分。
$$

- `Stability`（稳定性/可复现性）= $100 \times (1 - \text{unstable\_rate}) \times \mathbb{1}[\text{复跑偏差} \le 2\%]$，并扣减环境指纹不一致、复跑失败等事件分。
- **任一前置门禁不通过（环境指纹缺失、复跑失败、severe_count 超阈值）则总分标记为 `INVALID`，不上榜。**

**波动与回归判定**：
- 每个 case 报告 5 轮中位数 + MAD + CV；每个聚合指标用 **bootstrap 10k 次**给出 95% CI。
- 版本间对比采用**配对 bootstrap**（case 级配对），$\Delta \ln P_{\text{geo}}$ 的 95% CI 不跨 0 **且** $|\Delta| \ge 2\%$（死区，dead-band）才判定为"显著提升/显著回归"，否则标 `NO_SIGNIFICANT_CHANGE`。
- 聚合层面另声明"波动区间"：报告 $P_{\text{geo}}$ 的 CI95，而不是一个裸数。

### 6.5 运行规则（写入规范的硬约束）

| 项 | 规则 |
| --- | --- |
| 时钟 | 必须锁频（锁频失败则标注并放宽 CV 门禁至 5%） |
| 隔离 | 独占 GPU 与节点；无其他计算进程 |
| warmup | 25ms（可在 profile 覆盖）；另加 3 次预热执行 |
| rep | 100ms/轮；轮数 R=5，不稳定时升至 15 |
| L2 flush | 每轮前执行（fast_flush 语义） |
| 顺序 | 多被测对象随机交错，起点随机 |
| JIT 缓存 | 目录与 key 策略固定并入库；冷编译单独计时（清空缓存路径） |
| 超时 | 单 case 60s 硬超时，超时记 TIMEOUT 判失败 |
| 复跑 | official 档位 48h 内双跑，偏差 ≤ 2% |
| 日志 | 原始计时样本（非仅中位数）+ stdout/stderr + 编译产物全部留存 |

### 6.6 报告模板（标准化）

一份标准报告至少包含：

1. **元信息**：报告版本、生成时间、spec 版本、corpus 版本、scorer 版本、run_id 列表
2. **环境声明**：硬件型号与数量、驱动/运行时、镜像 digest、锁频状态、环境指纹
3. **被测对象**：编译器名称/版本/commit、capability manifest 摘要
4. **Coverage 结果**：总分、分层通过率、能力画像雷达图、能力缺口清单（FAIL 的 FA 列表）、severe（静默错误）清单
5. **Performance 结果**：几何平均加速比 + CI95、fast_1/fast_1.2/fast_2、编译时间分位、代码质量摘要、unstable 清单
6. **稳定性结果**：CV 分布、复跑偏差、环境指纹一致性
7. **版本对比**：与上一版本的配对 bootstrap 差异表（提升/回归/无显著变化）
8. **可追溯附录**：日志与产物索引（hash + URI）、签名、复现命令
9. **解释边界声明**：引用 §3.3 的边界条款

### 6.7 榜单规则

- **分组**：`(硬件型号, 运行时主版本, corpus 版本)` 三维分组；组内按 `Total` 排序。
- **展示字段**：编译器 + 版本 + Coverage + Performance + Total + CI95 + 更新时间 + `run_id`（可下钻）。
- **状态标签**：`VERIFIED`（已复跑）、`PRELIMINARY`（未复跑/活跃轨）、`UNSTABLE`、`INVALID`。
- **不展示**：跨组加权总分、未附环境指纹的条目、来源不可复现的条目。
- **更新频率**：月度；重大版本发布可申请临时更新。

### 6.8 提交与准入流程

```
1. 提交者 fork 平台仓库 → 填写 submission.yaml（编译器名称/版本/commit/构建脚本/镜像 digest）
2. 提交 Dockerfile/构建脚本 + capability manifest（声明支持/不支持的 FA）
3. CI 自动构建 → 在指定硬件上按 official profile 全量执行 → 自动生成 scores.json
4. 自动校验：环境指纹完整、复跑偏差 ≤ 2%、无 severe 静默错误超阈值
5. 运营组抽检复跑（≥10% 条目，独立硬件）
6. 通过后进入 VERIFIED 榜单；失败则公示失败原因与日志
7. 异议通道：10 个工作日内可提交仲裁申请，技术委员会裁决
```

---

## 7. 测试集生成方法细节

### 7.1 覆盖维度模型

五维笛卡尔空间（做**约束采样**而非全组合）：

| 维度 | 取值来源 | 示例 |
| --- | --- | --- |
| D1 语言特性 | FA 清单（L1–L3） | `tl.atomic_add` / `tl.associative_scan` / `tl.dot_scaled` |
| D2 dtype | fp64/fp32/tf32/bf16/fp16/fp8-e4m3/fp8-e5m2/int8/int16/int32/int64/uint*/bool | — |
| D3 shape/layout | 1D–4D；contiguous / strided / transposed / broadcast / 非对齐 / 尾块不对齐 / 超大尾数 | `[1,2048]`, `[4096,4096]`, stride=3 |
| D4 控制流 | 无 / if / static for / dynamic for / while / 嵌套 / 早退 / 循环内归约 | — |
| D5 优化 pattern | tiling 形状、num_warps、num_stages、向量化宽度、mask 策略、shared memory 复用、双缓冲、持久化 kernel、swizzle、split-K、epilogue 融合、grid 映射（1D/2D/3D） | `BLOCK=128,warps=8,stages=4` |

**约束规则（避免无效组合）**：如 `tl.dot` 要求 2D tile 且 K 维对齐；fp8 仅对特定 shape 有效；`tl.atomic_cas` 仅用于整数；`num_stages ≥ 2` 才启用软件流水等。约束以规则表形式维护，生成前做可行性检查。

### 7.2 算子探针的构造方式与配比

统一口径：**所有 case 都是"某个算子的某个变体"**，不存在脱离算子的裸 API 用例。

| 构造方式 | 说明 | 建议占比 |
| --- | --- | --- |
| **真实算子抽取 + 规范化**（T2） | 从 FlagGems / vLLM / sglang / Liger-Kernel / flash-attention 抽取真实 Triton 算子，剥离框架依赖、参数化 shape、绑定参考实现 | **40%** |
| **ATen 对齐算子构造**（T1） | 按 ATen 语义分层抽样构造 Triton 实现，参考实现直接取 PyTorch eager | **35%** |
| **算子变体参数化生成**（T3-a） | 对已有算子施加 dtype / shape / layout / 控制流 / 优化配置采样 | 15% |
| **覆盖引导变异生成**（T3-b） | libcst 变异（换 dtype、加 mask、改控制流、引入 atomic / scan），专补 Op→FA 矩阵空格 | 10% |

**真实算子占比提高到 40% 的原因**：本项目以算子为探针，真实算子是唯一能保证"评的是真实编译压力"的来源，也直接对应 triton-ascend 路线图中的适配目标（FlagGems / vLLM / sglang 重点算子）。

**算子抽取规范化规程**（必须留痕）：

```
源仓定位(仓库 + commit + 文件路径 + 算子名)
  → 依赖剥离（去掉自定义 autotune / 缓存 / 分布式 / 框架调度逻辑）
  → shape 参数化（硬编码维度 → 参数，给出合法取值范围）
  → 参考实现绑定（ATen 等价实现 或 源仓自带 reference）
  → 语义契约编写（§4.14）
  → FA 标签抽取（静态分析 kernel AST + 人工抽检校正）
  → 四道门禁（§4.6）→ 入库
```

抽取 diff 与源仓 commit 一并归档，保证任何 case 都能回溯到它的真实来源，也便于源仓升级时同步更新。

### 7.3 生成与冻结流水线

```
seed(固定) → 采样器(Hypothesis 策略)
   → 模板渲染(Jinja2) → 变异(libcst)
   → G1 静态合法 → 参考编译器编译(基线 Triton)
   → Oracle A(torch) + Oracle B(TRITON_INTERPRET) 双校验
   → G2 可执行且语义一致 → G3 非平凡/去重 → G4 覆盖增益
   → 标签抽取(FA) → 入库 → 覆盖率重算 → 反馈采样权重 → 下一轮
   → 达到目标(corpus_breath ≥ 95%, 每 FA ≥ 5 case, N ≥ 1000)
   → 冻结快照(manifest + hash + 种子 + 生成日志 + 生成器 commit)
```

### 7.4 覆盖率达标验证（对应"≥90%"）

以 FA 清单版本 v1.0（建议 120–160 个原子）为准：

- `corpus_breath = |{a : |C_a| ≥ 5}| / |A| ≥ 0.95`
- 每个 L1 原子：`|C_a| ≥ 10`（核心能力需更强覆盖）
- 每个 L2 原子：`|C_a| ≥ 5`
- 每个 L3 原子：`|C_a| ≥ 3`
- 总 case 数 ≥ 1000（其中有效 case 占比 ≥ 98%）
- 至少 30% 的 case 带"跨维度组合标签"（FA-COMB-*）

### 7.5 审计留痕

每次生成输出 `generation-log-{timestamp}.jsonl`，逐条记录：`case_id / 采样参数 / 模板 / 变异算子 / 门禁通过情况 / 拒绝原因 / 覆盖增益 / generator_commit / seed`。corpus 版本发布时随快照一并归档，支持事后回答"这个 case 是怎么来的"。

---

## 8. 实施计划与治理

### 8.1 分期建议

| 阶段 | 周期 | 交付物 | 关键里程碑 |
| --- | --- | --- | --- |
| **P0 规范先行** | 1–1.5 月 | spec v0.9、FA 清单 v0.9、容差表、运行规则 | FA 清单双方确认（这是后续一切的地基） |
| **P1 算子探针 corpus v1.0** | 2–2.5 月 | 算子抽取器 + 四门禁 + Op→FA 矩阵 + 覆盖引擎 + 1000+ 算子变体冻结快照 | 算子模板 ≥ 200、corpus_breath ≥ 95%、每 FA ≥ 5 case、P0 组算子 100% 入库 |
| **P2 执行与评分** | 1.5–2 月 | harness + bench 封装 + 环境指纹 + 评分器 + 报告模板 | 同一环境两轮偏差 ≤ 2% |
| **P3 平台原型** | 1.5 月 | 提交流程 + 自动复跑 + 榜单站 + 报告自动生成 | 首个 VERIFIED 条目上榜 |
| **P4 试运营** | 持续 | 季度报告、errata 流程、社区贡献 | ≥3 编译器 / ≥2 硬件 / ≥4 次榜单更新 |

总计约 **6–8 个月**（含试运营前的建设期），与规划的三项成果一一对应。

### 8.2 资源需求

- 人力：规范/编译器 1–2 人、生成器工程 2 人、执行与平台 2 人、运营 1 人（可兼任）。
- 硬件：至少 2 类硬件各 1 个独占节点（榜单发布用），另需 1 个开发/CI 节点。
- 存储：corpus 快照（小）+ 原始日志与产物（每轮 GB 级，建议按轮次保留 6 个月，聚合结果永久保留）。

### 8.3 验收指标映射

| 规划目标 | 可验收指标 |
| --- | --- |
| 形成一套评测体系 | spec v1.0 + 机器可读规范 + 报告模板 + 评分脚本（可重复执行） |
| 测试集 ≥ 1000 case，特性覆盖 ≥ 90% | corpus v1.0-frozen：N ≥ 1000，有效 case ≥ 98%，corpus_breath ≥ 95%，每 FA ≥ 5 case |
| 平台原型 | 提交→复跑→评分→榜单全链路自动化；≥1 条 VERIFIED 记录可一键追溯 |

### 8.4 风险登记册

| 风险 | 概率 | 影响 | 应对 |
| --- | --- | --- | --- |
| FA 清单双方无法按期确认 | 中 | 高 | P0 设硬截止，超时则以上游官方文档为默认基线，争议项降级到 L3 试评 |
| 生成 case 质量不达标（大量平凡/重复） | 中 | 高 | 四门禁 + 每 200 case 人工抽检 + 覆盖率曲线监控 |
| 多硬件环境不可得 | 中 | 中 | 首期单硬件试点，能力协商机制先落地，后续扩展分组 |
| 锁频在部分硬件不可行 | 中 | 中 | 标注 `clock: unlocked` + 放宽 CV 门禁 + 增加重复次数 |
| 静默错误（severe）判定争议 | 低 | 高 | 双 oracle + 人工仲裁 + 单独高亮披露 |
| 社区参与度不足 | 中 | 高 | P4 前即启动贡献者招募与 30 分钟上手体验 |
| 评分被特化 | 中 | 中 | held-out 集 + 语义变异检测 + 抽检复跑 |
| 上游 Triton API 快速演进导致清单过期 | 高 | 中 | 年度 MAJOR 复审 + L3 层容纳新特性 |

### 8.5 治理机制要点（写入运营计划）

- 技术委员会：规范解释、争议仲裁、FA 清单变更审批。
- 运营组：复跑、发布、抽检、errata 执行。
- 贡献者：case 提交、缺陷报告、编译器提交；贡献计入致谢与贡献榜。
- 透明度：所有规范、corpus、日志、产物公开；所有争议与裁决记录公开。
- 中立性：运营方不参与编译器开发，或参与时必须回避相关条目的评审（利益冲突声明制度）。

---

## 9. 附录 A：Triton 语言特性原子清单草案（骨架）

以下以 `triton.language` 官方 API 的 15 个类别为骨架，给出 FA 分组建议（实际条目需 P0 阶段细化并与各方确认）：

| # | 类别 | 代表 API | 建议层 | 建议原子拆分方向 |
| --- | --- | --- | --- | --- |
| A01 | 编程模型 | `program_id`, `num_programs`, `tensor`, `tensor_descriptor`, `map_elementwise` | L1 | 1D/2D/3D grid 映射；descriptor 基础 |
| A02 | 创建 | `arange`, `full`, `zeros`, `zeros_like`, `cat`, `cast`, `to_tensor` | L1 | 各构造函数 × dtype |
| A03 | 形状操作 | `broadcast`, `broadcast_to`, `expand_dims`, `permute`, `reshape`, `trans`, `view`, `split`, `join`, `interleave`, `ravel`, `squeeze`, `unsqueeze` | L2 | 静态/动态 shape；非 2 的幂 reshape |
| A04 | 线性代数 | `dot`, `dot_scaled` | L1(dot) / L3(dot_scaled) | 按 dtype 与 accumulate dtype 拆分；非对齐 K |
| A05 | 内存/指针 | `load`, `store`, `make/load/store_tensor_descriptor` | L1(load/store) / L3(TMA) | 连续/跨步/mask/越界/other/向量化宽度；TMA 描述符 |
| A06 | 索引 | `where`, `flip`, `swizzle2d` | L1(where) / L3(swizzle) | 标量/张量条件；swizzle 模式 |
| A07 | 数学 | `abs, add, cdiv, ceil, clamp, cos, div_rn, erf, exp, exp2, fdiv, floor, fma, log, log2, maximum, minimum, mul, rsqrt, sigmoid, sin, softmax, sqrt, sqrt_rn, sub, umulhi` | L1 | 逐算子 × dtype；`div_rn`/`sqrt_rn` 舍入模式单列 |
| A08 | 归约 | `argmax, argmin, max, min, reduce, reduce_or, sum, xor_sum` | L1 | axis / keep_dims；自定义 combine_fn；跨维度 |
| A09 | Scan/Sort | `associative_scan, cumprod, cumsum, histogram, sort, topk, gather` | L2 | 各算子 × 方向 × dtype；稳定性要求 |
| A10 | 原子 | `atomic_add/and/cas/load/max/min/or/poll/store/xchg/xor` | L2 | 算子 × 标量/张量 × mask × sem(scope) |
| A11 | 随机数 | `rand`, `randn`, `randint`, `rand4x`, `randn4x`, `randint4x` | L2 | seed 可复现性；分布 |
| A12 | 迭代器 | `range`, `static_range` | L1 | 静态/动态边界；嵌套；循环内归约 |
| A13 | 内联汇编 | `inline_asm_elementwise` | L4 | 厂商专有，不计分 |
| A14 | 编译提示 | `assume`, `max_constancy`, `max_contiguous`, `multiple_of`, `debug_barrier` | L2(max_*/assume) / L3(debug_barrier) | 提示生效性（可通过产物质量间接验证） |
| A15 | 调试 | `static_print`, `static_assert`, `device_print`, `device_assert`, `expect_zero` | L3 | 编译期/运行期断言行为；错误诊断质量 |

**补充维度（正交标签，不占 FA 编号）**：
- `D-DTYPE-*`：fp64/fp32/tf32/bf16/fp16/fp8e4m3/fp8e5m2/int8/int16/int32/int64/uint*/bool
- `D-LAYOUT-*`：contiguous / strided / transposed / broadcast / unaligned / tail-partial
- `D-OPT-*`：num_warps / num_stages / persistent / split-K / swizzle / epilogue-fusion / 双缓冲
- `D-DIAG-*`：编译错误信息可读性（可选人工评分维度）

---

## 9.5 附录 C：算子探针清单草案（面向 triton-ascend）

> 说明：以下为算子池的**分组骨架**，P0 组应优先建设并与 triton-ascend 路线图对齐；具体算子名与数量在 P0 阶段由双方共同确认后冻结。

| 组 | 代表算子 | 主要触发的编译路径（用于 Op→FA 反投影） | 优先级 |
| --- | --- | --- | --- |
| **G1 Elementwise / 激活** | add, mul, silu, gelu, relu, sigmoid, tanh, exp, rsqrt, clamp, dropout(确定性) | 向量化访存、连续/跨步 load-store、mask 边界、数学库映射 | P0 |
| **G2 Reduction / Norm** | sum/max/min/mean, softmax, log_softmax, layer_norm, rms_norm, batch_norm, group_norm, cross_entropy | 块内归约、多阶段归约、warp shuffle、layout 转换、eps 数值约定 | **P0** |
| **G3 Matmul / Linear** | addmm, matmul, bmm, gemm(含 bias/epilogue), linear, fp16/bf16/fp8/tf32 变体 | `tl.dot` 各 dtype、accumulator 选择、shared memory 双缓冲、软件流水、split-K、swizzle | **P0** |
| **G4 Attention / 融合** | flash_attention(前向/反向), fused_softmax, rotary_embedding(RoPE), alibi, paged-attention 相关 kernel | dot × softmax 融合、在线 softmax、mask、跨 block 归约、复杂控制流 | **P0** |
| **G5 MoE / 稀疏** | fused_moe, topk_gating, scatter_add, index_add, gather, moe_align | topk/sort、atomic 累加、scatter/gather、非连续访存、动态 shape | **P0** |
| **G6 索引 / 数据搬移** | gather, scatter, index_select, embedding, repeat_interleave, flip, cat/split, transpose/permute, slice | 非连续访存、stride 计算、permute/layout、边界保护 | P1 |
| **G7 Scan / Sort / 累积** | cumsum, cumprod, associative_scan, sort, histogram, unique 类 | associative_scan、sort 网络、跨 block 累积、shared memory 复用 | P1 |
| **G8 随机数 / 采样** | randn, randint, multinomial, top-k/top-p 采样 kernel | `tl.rand*`、`rand4x`、seed 可复现性、原子竞态规避 | P1 |
| **G9 类型转换 / 量化** | cast, fp32↔bf16/fp16/fp8, mxfp 相关, dequant/quant, per-token/per-channel 量化 | dtype 转换链、舍入模式、饱和截断、向量化宽度变化 | P1 |
| **G10 损失 / 优化器** | mse, bce, adamw(单步), sgd, clip_grad_norm | 多张量融合、atomic、归约 + 更新、in-place 语义 | P2 |
| **G11 卷积 / 特殊算子** | conv(implicit gemm), pool, upsample, interpolate | 复杂 index 计算、掩码、非方形 tile | P2 |
| **G12 边界与压力算子**（T3） | 尾块不对齐、非 2 的幂 shape、超长序列、极端 num_warps/stages、深嵌套控制流、非对齐指针 | 边界保护、mask、编译期常量传播、寄存器压力与 spill | P2 |

**Op→FA 映射矩阵片段示例**（用于反投影与补盲）：

| 算子 | FA-LOAD | FA-REDUCE | FA-DOT | FA-ATOMIC | FA-SCAN | FA-CTRL | FA-DTYPE | FA-LAYOUT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rms_norm` | 连续 + 尾块 mask | 行归约 | — | — | — | 无 | fp16/bf16/fp32 | contiguous |
| `fused_moe` | 跨步 + 非连续 | topk | 可能 | 累加 | sort/topk | 动态 for | fp16/bf16 | 非连续 |
| `rotary_embedding` | 跨步 | — | — | — | — | 动态 for | fp16/bf16 | strided |
| `flash_attention` | TMA/连续 | 在线归约 | 是 | — | — | 嵌套 for | fp16/bf16 | 多头 stride |
| `cross_entropy` | 连续 | 两级归约 | — | — | — | if | fp32 | contiguous |

**建设顺序建议**：G2/G3/G4/G5（P0，直接对应 LLM 训推热点与 triton-ascend 适配目标）→ G1/G6/G7/G8/G9（P1，补语言特性面）→ G10/G11/G12（P2，边界与压力）。

---

## 10. 附录 B：参考资料

- TritonBench（THUNLP），arXiv:2502.14752 / ACL Findings 2025：<https://arxiv.org/abs/2502.14752>，代码 <https://github.com/thunlp/TritonBench>
- KernelBench（Stanford）：<https://github.com/ScalingIntelligence/KernelBench>（`fast_p` 指标）
- meta-pytorch/tritonbench：<https://github.com/meta-pytorch/tritonbench>
- Triton 上游测试体系：`python/test/unit/language/`（test_core / test_standard / test_matmul / test_pipeliner / test_tensor_descriptor / test_warp_specialization / test_mxfp / test_random 等）
- `triton.language` API 文档（15 类）：<https://triton-lang.org/main/python-api/triton.language.html>
- `triton.testing.do_bench` 口径：<https://triton.hyper.ai/docs/python-api/triton.testing/triton_testing.do_bench/>
- Triton-Ascend（昇腾 Triton 编译框架）：<https://gitee.com/ascend/triton-ascend>，文档 <https://triton-ascend.readthedocs.io/>（含版本兼容矩阵与里程碑）
- FlagGems（跨芯片 Triton 算子库）：<https://github.com/FlagOpen/FlagGems>
- 编译器随机测试：Csmith <https://github.com/csmith-project/csmith>；YARPGen <https://github.com/intel/yarpgen>
- GEAK-Triton（AMD 侧 Triton kernel 优化）：公开报告在 TritonBench-modified 上取得平均 3.42× 加速
