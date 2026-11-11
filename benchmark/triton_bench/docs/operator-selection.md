# 算子选型清单

来源：`reference/cann-bench/tasks/`（53 个算子，13 个 category，4 个 level）。
选型原则：**优先选 triton 能写出来的**，不选 triton 极难或不现实的。

## 一、category 与 level 的关系

两者正交：

| 维度 | 回答 | 取值 |
|---|---|---|
| `level` | 写起来多难 | L1–L4（纵向难度） |
| `category` | 算子干什么 | 13 个语义族（横向类别） |

cann-bench 的真实分布：

| category | L1 | L2 | L3 | L4 | 算子 |
|---|---|---|---|---|---|
| Broadcast | 0 | 2 | 0 | 0 | Gcd, Maximum |
| Contraction | 0 | 0 | 7 | 0 | Conv2D, Conv3DBackpropFilter, DepthwiseConv2D, Dilation2D, GroupedMatmul, QuantMatmul, WeightQuantBatchMatmul |
| Elementwise | 5 | 0 | 0 | 0 | Exp, Gelu, Mish, Sigmoid, SwiGlu |
| FusedComposite | 1 | 5 | 4 | 8 | ForeachAddcdivScalar, ApplyAdamW, ApplyRotaryPosEmb, DynamicQuant, GridSampler3D, ResizeBilinear, AddRmsNormDynamicQuant, DequantSwigluQuant, MoeFinalizeRouting, ROIAlign, GQA, GRU, LSTM, MHA, MLA, ... |
| IndexGather | 0 | 1 | 0 | 0 | Gather |
| LayoutTransform | 0 | 0 | 4 | 0 | MoeGatingTopKSoftmax, MoeReRouting, StridedSlice, Transpose |
| MaskPredicate | 1 | 0 | 0 | 0 | MaskedScale |
| Normalization | 0 | 3 | 0 | 0 | GroupNorm, RmsNorm, Softmax |
| NumericalStable | 0 | 1 | 0 | 0 | CrossEntropyLoss |
| Reduction | 1 | 1 | 1 | 0 | ForeachNorm, Cummin, AdaptiveAvgPool3D |
| ScatterUpdate | 0 | 2 | 0 | 0 | Scatter, UnsortedSegmentSum |
| SortSelect | 0 | 1 | 3 | 0 | ArgMax, NMS, TopK, Unique |
| VVFusion | 0 | 0 | 2 | 0 | EngramGateFusion, MhcSinkhorn |

## 二、选型结果（13 category，每类 5 个 case）

### 第 1 批 · triton 擅长（7 个 category）

| category | 选用算子 | level | 目录 | 说明 |
|---|---|---|---|---|
| Elementwise | `exp` | L1 | `level1/exp` | 含 attrs（base/scale/shift），可测 attrs 变化 |
| Elementwise | `gelu` | L1 | `level1/gelu` | approximate=none/tanh 两模式 |
| Reduction | `cummin` | L2 | `level2/cummin` | 前缀归约，比全量归约更能暴露差异 |
| Broadcast | `maximum` | L2 | `level2/maximum` | 双输入 + 广播 |
| MaskPredicate | `masked_scale` | L1 | `level1/masked_scale` | `tl.where` 掩码语义 |
| Normalization | `softmax` | L2 | `level2/softmax` | 沿 dim 归约，triton 经典 |
| Normalization | `rms_norm` | L2 | `level2/rms_norm` | 真实模型负载 |
| IndexGather | `gather` | L2 | `level2/gather` | 索引访存 |
| ScatterUpdate | `scatter` | L2 | `level2/scatter` | 写回 + 可能涉及 atomic |

> 说明：Elementwise / Normalization 各选 2 个算子凑 5 case 的多样性；
> 其余 category 各 1 个算子 × 5 case。

### 第 2 批 · 可做但需挑算子（5 个 category）

| category | 选用算子 | level | 目录 | 说明 |
|---|---|---|---|---|
| Contraction | `grouped_matmul` | L3 | `level3/grouped_matmul` | **只选 matmul 类**；Conv2D/Conv3DBackpropFilter/DepthwiseConv2D 需显式 im2col，triton 极难，不选 |
| LayoutTransform | `transpose` | L3 | `level3/transpose` | 布局变换，本类核心语义需放开白名单 |
| NumericalStable | `cross_entropy_loss` | L2 | `level2/cross_entropy_loss` | 数值稳定性 |
| SortSelect | `arg_max` | L2 | `level2/arg_max` | **只选 ArgMax**；TopK/Unique/NMS 需多 pass，暂不选 |
| FusedComposite | `swi_glu` | L1 | `level1/swi_glu` | **只选 L1 简单融合**；L4 的 MHA/MLA/GRU/LSTM 暂不选 |

### 第 3 批 · 暂缓（1 个 category）

| category | 状态 | 原因 |
|---|---|---|
| VVFusion | 留占位，不生成 case | EngramGateFusion / MhcSinkhorn 为向量-向量融合，triton 无可行实现 |

## 三、汇总

| 批次 | category 数 | case 数 |
|---|---|---|
| 第 1 批 | 7 | 35 |
| 第 2 批 | 5 | 25 |
| 第 3 批 | 1 | 0（占位） |
| **合计** | **13** | **60** |

## 四、case 设计维度

每个算子的 5 个 case 从以下维度组合派生（参考 cann-bench `cases.yaml`）：

| 维度 | 取值来源 |
|---|---|
| `shape` | 从算子 `shape_support` 约束内选，覆盖 1D / 2D / 3D+、边界小 shape、大 shape |
| `dtype` | 从算子 `inputs[].dtype` 中选（fp32 / fp16 / bf16 / 整型按算子支持） |
| `attrs` | 从 `attrs` 定义中取非默认值与默认值组合 |
| `value_range` | 常规分布 + 边界值（0、负值、极大/极小、含 NaN 风险的输入） |
| 布局 | 连续为主，LayoutTransform 类含非连续 |

## 五、与 MVP 既有 20 case 的关系

既有 5 个手写算子（add / mul / sum / softmax / layer_norm）属 Elementwise + Reduction，
会被本变更的选型**替换**：

- `add` / `mul` → 归入 Elementwise（保留为候补算子）
- `sum` → 归入 Reduction（`cummin` 覆盖度更高）
- `softmax` → 与 `level2/softmax` 重复，直接采用 cann-bench 版本
- `layer_norm` → 归入 Normalization（候补）

迁移后，corpus 从「5 算子 × 20 case」变为「14 算子 × 60 case」。
