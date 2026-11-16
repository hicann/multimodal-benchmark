# 官方来源审计与 v0.8 候选集

## 状态与口径

v0.7 的 11 个接入项目、220 个用例在记录 `20261023-081041-845799` 全部 PASS。
v0.8 的首轮 68 个候选、1360 例运行结果现已取回，记录 `20261023-120902-701382`，
对应代码 `87f1066`：63 项整算子通过，排除 4 项语言原语后有 59 项可计规模，
已选择 50 项/1000 例。原全量仍有五项未通过，冻结标签覆盖仅 33/71（46.48%）。
发布脚本曾破坏远程冻结目录的来源 JSON，本轮已按原 SHA256 恢复并修复发布逻辑。
逐例统计、失败分析、恢复与复验指令见
[v0.8 真机结果审计与 v0.8.1 修复](v08-npu-failure-analysis.md)。

当前运行入口使用独立修复版本 **v0.8.1**：仍为 68 个候选、每算子 20 例、共 1360 例。
CPU 双参考通过不等于 NPU 通过；当前候选状态仍为 `CANDIDATE_NOT_NPU_QUALIFIED`，
必须以同一新资产、提交和环境的真机运行重新确定资格。旧 v0.8 证据保留，不重写指纹。
历史 25 个退出新版正式集，保留旧代码和证据用于追溯。

## 来源层级

| 来源 | 可据此确定的事实 | 不能据此推定 |
|---|---|---|
| `Ascend/triton-ascend-ops` | 真实 Triton 教程及优化实例 | 15 个主题等于 15 个独立算子 |
| `triton-lang/triton-ascend` | 迁移后的官方编译器仓、教程、API 示例、正确性测试 | 当前执行机的旧 Triton/CANN 也支持全部接口 |
| `cann/ops-math`、`cann/ops-nn`、`cann/ops-transformer` | CANN 官方算子定义与实现 | AscendC/ACLNN 实现可直接作为 Triton 提交 |
| 本地 `reference/cann-bench` | 任务定义、golden、参数；少量真实 Triton 例子 | 每个任务都有现成 Triton 内核 |

核查日期：2026-10-23。旧 `Ascend/triton-ascend` 已公告迁移。
编译器来源固定到 `5f1d99a87432c09815b6ded57624c04c00bbe0aa`。
逐项路径、提交、SHA256、实际 Triton 调用、上游 skip 标记及适配说明见
`reference/ascend-official/sources-v08.json`。
67 项来自官方编译器仓；Mish 使用 CANN Bench 定义、golden 与本地真实 Triton 示例。

同时读取 CANN `ops-math`、`ops-nn`、`ops-transformer` 的固定版本目录树及相关定义，
64/68 项找到数学对应；Exp2、Log2、Log10、XorSum 仅有本次官方编译器原语来源，
不冒充已找到独立 CANN 算子。逐项 CANN 路径、提交与定义 SHA256 见
`reference/ascend-official/cann-correspondence-v08.json`，该映射也纳入冻结资产。
源码语言和产品支持以各仓定义及真机证据分别判断，CANN 的支持表不能证明 Triton 后端支持。

特别地，SortValues/TopKValues 只映射 values；RmsNorm/LayerNorm 只映射前向 y，
不宣称覆盖 CANN 的 indices、rstd、mean/variance 等完整输出。
CANN Gelu 文档定义 tanh 近似，本批 exact erf 模式来自官方 Triton，明确区分来源。
最终 50 个只从这 64 个有直接 CANN 对应的候选中选取；其余 4 个保留语言能力验证，
不计入算子规模，避免把 Exp 的固定底数变体等重复计数。因此规模候选仍有 14 个冗余。

严格映射比较数学函数、输入输出、轴、dtype、舍入、边界与副作用，允许明确声明的
接口适配和受测子域。提交主机侧只分配输出、设置 launch，核心计算均为 `@triton.jit`。
本批合同域为连续非空二维张量、按末轴操作，不声称所有 rank、广播、非连续布局、
动态 shape、NaN/Inf 和全部 dtype 均完成。

## 为什么 15 项只接入 11 项

此前称“四个无效算子”不准确，实际是四个未接入教学主题：

| 项目 | 实际语义 | 处理原则 |
|---|---|---|
| `003-ub_overflow.py` | `alloc_extend_kernel` 为新增 token 分配 KV 槽位，演示 UB 分块 | 有独立功能。需补长度、页分配、输出偏移契约后接入，不能说没有算子 |
| `006-tiling.py` | 二维 `Gather(dim=1)` 的分块实现 | 有语义；归为 Gather 实现/规模变体，不能另加算子数 |
| `005-load_order.py` | BA/AB 加载顺序，计算输出并更新 B 状态 | 需明确共享状态、写入并发及多输出契约，不能直接当成无副作用单输出 |
| `003-load_care_padding.zh.md` | `tl.load(care_padding=...)` 的 API 文档，无对应完整算子 Python 文件 | 放入能力探针/访存边界；mask 外未定义数据不应参与数值比较，不独立计算子数 |

“15”是主题数，混合了完整算子、优化和 API 文档。统计 Python 文件时还要排除
best_practice 的 `utils.py`。这四项不是因为本轮编译失败才被剔除。

当前 11 项还存在命名与语义重叠：

- VectorAdd 就是 Add；VectorCmp 实际是无 affine 的 LayerNorm 前向及统计输出。
- DiscreteMemoryAccess 是索引选取；InsertSlice 的参考为 `out[i]=x[index[i]]`，
  ExtractSlice 为 `out[index[i]]=x[i]`。这是置换的正向/逆向，不是通用 StridedSlice。
- GatherScatter 当前只接 gather 前向，weights 不参与输出，不代表 scatter/加权反向已测。
- Binned/Padded 版本增加容量截断、桶布局或 padding；合同有差异，但不能仅因布局或
  优化不同就自动当成三种独立核心算子。

因此新版不将旧 11 个与新 68 个直接相加，也不重复累计 Add/VectorAdd 等别名。

## 历史 25 个的处置

| 分组 | 名称 | 新版处理 |
|---|---|---|
| 9 个重映射名称 | Exp、Gelu、Maximum、Softmax、RmsNorm、Gather、Transpose、ArgMax、SwiGlu | 重新给出来源、契约、参考、Triton 实现和用例，不继承历史资格 |
| 5 个 CANN Bench 基础任务 | Cummin、MaskedScale、Scatter、GroupedMatmul、CrossEntropyLoss | 保留历史，核对完整官方接口后重新接入 |
| 7 个组合任务 | LinearGelu、RmsNormLinear、GroupedMatmulSwiGlu、TransposeLinear、MaskedSoftmax、MaskedGelu、GatherExp | 退出新正式集；组合可成立，但不能冒充完整官方目标任务 |
| 4 个后续池映射 | AddRmsNorm、ApplyRotaryPosEmb、MoeGatingTopKSoftmax、UnsortedSegmentSum | 不整体继承，逐项重新审计完整定义与输出 |

实质问题：AddRmsNorm 仅保留 AddRmsNormDynamicQuant 的加法/归一化主干，缺量化及输出；
GroupedMatmulSwiGlu 也不是完整 GroupedMatmulSwiGluQuant。
没有直接复用量化任务名称，不等于可凭原任务来源证明裁剪后的接口严格一致。
反过来，不能仅因教程首页没列出就认定算子不存在：很多数学函数位于官方 API/测试中。

上一轮 14 个是初筛候选，不是实现清单。Mish/Sigmoid 本次落地；Gcd、GroupNorm、
StridedSlice、完整带 indices 的 TopK、Conv2D、DepthwiseConv2D、AdaptiveAvgPool3D、
ResizeBilinear、DynamicQuant、QuantMatmul、WeightQuantBatchMatmul、ApplyAdamW
仍属后续队列，不虚假计入本次实现数量。量化舍入/饱和/scale 和 AdamW 状态/原地契约
须先核清，再实现和入场。

## 本次 68 个实际候选

| 分类 | 数量 | 候选 |
|---|---:|---|
| Elementwise | 38 | Abs、Neg、Exp、Exp2、Expm1、Log、Log2、Log10、Log1p、Sqrt、Rsqrt、Sin、Cos、Tan、Tanh、Sinh、Cosh、Floor、Ceil、Trunc、Relu、Sigmoid、Add、Sub、Mul、Div、FloorDiv、Mod、Pow、Maximum、Minimum、BitwiseAnd、BitwiseOr、BitwiseXor、BitwiseNot、Clamp、Mish、Gelu |
| MaskPredicate | 10 | Signbit、Equal、NotEqual、Greater、GreaterEqual、Less、LessEqual、LogicalAnd、LogicalOr、Where |
| Reduction | 7 | ReduceSum、ReduceMean、ReduceMax、ReduceMin、XorSum、CumSum、CumProd |
| SortSelect | 4 | ArgMax、ArgMin、SortValues、TopKValues |
| LayoutTransform | 3 | Transpose、Concat、Flip |
| Normalization | 3 | Softmax、RmsNorm、LayerNorm |
| IndexGather | 1 | Gather |
| Contraction | 1 | Matmul |
| FusedComposite | 1 | SwiGlu |

本批偏基础数学和编译器原语，不能描述成 68 个复杂模型算子，也不覆盖全部 13 类。
分类数不是算子数上限。后续应补索引更新、量化、优化器、卷积和注意力改善分布，
不能凭空造分类或把参数变化当成独立算子。

已修正的具体映射陷阱：

- 整数 BitwiseNot 对应 `test_invert.py`，不是 bool 非的 `test_not.py`。
- Mod 是截断余数 fmod，不是 Python 向下取整余数；FloorDiv 要求向下取整。
- `test_18_gather.py` 是行 embedding gather；元素 Gather 使用官方 API 示例。
- `test_log1p.py` 是 add+log1p 融合；纯 Log1p 使用官方 API 示例。
- 有序 Concat 参照 `test_cat_dim.py`；`test_cat.py` 允许 reorder。
- SortValues/TopKValues 只有 values，与官方示例一致，不冒充 CANN Bench 带 indices TopK。
- 官方 SwiGlu 有两个数据输入；本批按 CANN Bench 采用单输入两半，明确适配，不重复计数。

## 生成、隔离与冻结

流程：来源锁定 → 合同 → 独立参考 → 候选池 → 覆盖引导/强制边界 → NPU 验证 → 整算子资格 → 50/1000 冻结。

复用已有 label_case、guided_selection、正式 CLI、比较器、监督进程、完整性审计、
评分/报告及 GitCode 上传。没有绕过正式评测另造 PASS。
每算子从 22 种 shape 与 dtype 的组合池选 20 例，不靠仅换 seed 重复计数。
强制保留单元素、并列/符号边界、零/退化与较大尾块；池中有整块对齐、非对齐、
2 幂及相邻长度。输入构造有随机与边界模式，合同检查 dtype、轴及张量关系。
主参考用 Torch，独立参考用 NumPy/标量数学，提交用 Triton；核心计算不互相调用。
整数/布尔仍精确比较，没有放宽容差。
通用规格新增独立 case_dtypes，避免将 ArgMin 的 int64 索引输出或 Where 的 bool 条件
错误用作主数据类型；生成器和契约校验共用这一声明。

参考门禁可以逐算子隔离：某个 NPU Torch 参考不支持，记 REFERENCE_GATE_FAILURE 并继续，
不能归罪于 Triton，也不能当 PASS。编译、运行、结果失败保留真实终态。
全局设备健康失败仍停止，避免污染归因。

qualify_official_results.py 独立重读原始结果，检查：

1. manifest、冻结资产及提交源哈希一致，不混代码版本。
2. 每算子恰好 20 例，无缺漏/重复/额外，单一 run_id，全部 PASS。
3. CPU/NPU 参考门禁及 run/score/report 成功。
4. integrity=OK、子进程正常退出，无环境错配、无运行中改源。
5. 真机 NPU 健康；按硬件/CANN/编译器/Torch 身份分组，只从同组选择。
6. 至少 50 个有直接 CANN 对应的候选通过后，按分类/语义族轮转及固定来源顺序选择，
   保留全部 68 项的失败台账；仅语言原语通过不能补足规模。

满足时自动在结果目录生成 frozen-official50：50 算子、1000 用例，以及参考、manifest、
评测资产和资格证明。新资产采用 @corpus/ 相对命名，下载/移动目录不改变身份。
不足 50 时只输出资格报告，不生成冒充完成的冻结件。

target_met 仅指功能规模 50/1000。覆盖报告使用完整 71 个 active 原子作分母，
**不表示语言核心特性达到任务书 90%**。复杂融合、多输出、多轴和布局仍有缺口，
必须单独补齐，不能缩小分母宣布任务书全部验收。

## 真机运行与回传

```bash
cd "$HOME/triton-eval-v07" &&
git pull --ff-only origin main &&
bash scripts/official-suite-verify.sh
```

环境激活、终端日志、独立目录、异常收尾与回传沿用 ascend-suite-verify.sh。
可追加 `--operators LayerNorm` 定位单项；完整资格需要全量候选证据。
真机不需重新联网采集来源或重复生成候选。当前开发机无 NPU，不能将 CPU 验证说成真机通过。

结果仍上传 npu-results，远端和已上传本地记录保留最新 3 轮，未上传记录保留。
冻结件随结果上传；验收后应晋升主仓，不能长期只依赖滚动分支。
退出码 1 可能仅表示部分冗余候选失败，应同时看 qualification.json 的 eligible_count、
target_met 及失败台账；不能删失败或只凭总退出码判断是否已达到 50/1000。

本次是新资产版本。v0.7 指纹绑定旧框架，复现 220 PASS 需用记录的 2583ed7 代码或
独立旧 checkout，不能篡改旧指纹让新版框架冒充原验证版本。

开发侧来源与生成命令：

```bash
python scripts/audit_official_sources.py
python scripts/build_official_corpus.py
```

来源锁不能被不同内容覆盖；评测资产或用例变化须发布新修订。

## v0.9 生成机制扩展（2026-10-24）

当前官方验证入口默认使用 `v0.9-official-candidates`，新集仍为 68 算子、1360 用例。生成机制新增显式且受合法域约束的数值分布、逐元素 1～4 维与 stride=2 主输入、较大性能负载，以及严格固定预算的覆盖选择。静态标签并集覆盖为 49/71（69.01%），未达到任务书的 90% 目标；原有 NPU 资格不能继承到新集。具体范围、数据和复现方式见 [v0.9 改进说明](v09-generation-improvements.md)。
