# 语言能力点清单 —— 双方共同确认版（依据 1.1-fusion，待定稿）

> 本文档由 `scripts/build_capability_checklist.py` 从机器清单（`spec/feature_atoms.yaml`）与题库覆盖证据（`corpus/v0.5-cov20-frozen`，冻结于 2026-10-13T09:34:27.876149+00:00）自动生成；改清单或题库后重跑脚本即可，对账测试保证文档不脱节。

**这份文档怎么用**：任务书要求「覆盖**双方共同确认**的 Triton 核心语言特性清单比例不低于 90%」。请逐条审阅：认可的条目在「确认」栏打勾；有异议的条目在「意见」栏写明修改 / 删除 / 合并建议。全部过完后，以确认版清单为分母重算覆盖率即可——标签是「算子规格 + 用例六维」的纯函数，**无需重新生成任何用例**。

**分母原则（诚实性原则）**：清单分「生效」（当前算子够得着，计入覆盖率分母）与「预留」（当前够不着，不计入分母——把够不着的条目放进分母，覆盖率就是自欺）两部分。算子扩容后，够得着的预留条目逐条迁入生效。

**当前状态**：生效 **71** 条，题库全部有题盯（覆盖 71/71，我方口径）；预留 **5** 条。

## 一、生效条目（计入分母，71 条）

### A01 编程模型（grid 调度）

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-PROG-GRID1D` | L1 | 一维 grid 映射（rank-1 输入） | AddRmsNorm、ArgMax、Cummin、Exp、Gather、GatherExp、Gelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、Scatter、Softmax、SwiGlu、UnsortedSegmentSum | 66 | ☐ | |
| `FA-PROG-GRID2D` | L1 | 二维 grid 映射（rank-2 输入） | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 114 | ☐ | |
| `FA-PROG-GRID3D` | L2 | 三维 grid 映射（rank-3 输入） | AddRmsNorm、ArgMax、Cummin、Exp、Gather、GatherExp、Gelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、Scatter、Softmax、Transpose | 15 | ☐ | |
| `FA-PROG-GRID4D` | L2 | 四维及以上 grid 映射（rank≥4 输入） | ArgMax、Cummin、Exp、Gather、Gelu、MaskedScale、Maximum、RmsNorm、Scatter、Softmax、SwiGlu、Transpose | 94 | ☐ | |

### A04 线性代数

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-LINALG-DOT` | L1 | tl.dot 矩阵乘（GroupedMatmul） | GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、RmsNormLinear、TransposeLinear | 44 | ☐ | |
| `FA-LINALG-BIAS` | L1 | 带偏置的线性组合（GroupedMatmul 的 bias 输入） | GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、RmsNormLinear、TransposeLinear | 44 | ☐ | |

### A05 内存 / 指针

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-LOAD-CONTIG` | L1 | 连续访存 tl.load | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 265 | ☐ | |
| `FA-LOAD-STRIDED` | L2 | 非连续（跨步）输入访存——实现必须处理 stride 或显式聚合 | ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、Gelu、GroupedMatmul、MaskedScale、Maximum、RmsNorm、Scatter、Softmax、Transpose、UnsortedSegmentSum | 24 | ☐ | |
| `FA-LOAD-MASKED-EDGE` | L1 | 元素总数非块对齐（非 1024 倍数），块尾必须带 mask | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、Scatter、Softmax、SwiGlu、Transpose、UnsortedSegmentSum | 156 | ☐ | |
| `FA-LOAD-ALIGNED` | L1 | 元素总数块对齐（1024 倍数），可走无掩码快路径 | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 133 | ☐ | |

### A06 索引与算子属性

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-IDX-WHERE` | L1 | tl.where 条件选择（MaskedScale 的掩码语义） | MaskedGelu、MaskedScale、MaskedSoftmax | 36 | ☐ | |
| `FA-IDX-GATHER` | L1 | 按索引聚集读取（Gather） | Gather、GatherExp | 26 | ☐ | |
| `FA-IDX-SCATTER` | L2 | 按索引散射写入（Scatter） | Scatter、UnsortedSegmentSum | 20 | ☐ | |
| `FA-IDX-PERMUTE` | L2 | 维度置换访存（Transpose 的 perm） | Transpose、TransposeLinear | 26 | ☐ | |
| `FA-AXIS-LAST` | L1 | 沿最后一维操作（最常见快路径） | ArgMax、Cummin、Gather、GatherExp、MaskedSoftmax、Scatter、Softmax、SwiGlu | 40 | ☐ | |
| `FA-AXIS-FIRST` | L1 | 沿第 0 维操作（跨行扫描） | ArgMax、Cummin、Gather、GatherExp、MaskedSoftmax、Scatter、Softmax、SwiGlu | 35 | ☐ | |
| `FA-AXIS-INNER` | L2 | 沿中间维操作（rank≥3 且 0<dim<rank-1）——只认末维的实现必错 | ArgMax、Cummin、Gather、Scatter、Softmax、SwiGlu | 16 | ☐ | |
| `FA-ATTR-NONDEFAULT` | L1 | 非默认属性组合（tanh 近似 / keepdim 等） | AddRmsNorm、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 208 | ☐ | |
| `FA-ATTR-LIST` | L2 | 列表型属性（perm 置换 / group_list 分组） | GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、RmsNormLinear、Transpose、TransposeLinear | 64 | ☐ | |
| `FA-MULTI-INPUT` | L1 | 多输入算子（≥2 张量） | AddRmsNorm、ApplyRotaryPosEmb、CrossEntropyLoss、Gather、GatherExp、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、RmsNormLinear、Scatter、TransposeLinear、UnsortedSegmentSum | 187 | ☐ | |
| `FA-MULTI-OUTPUT` | L2 | 多输出算子（值 + 索引） | Cummin、MoeGatingTopKSoftmax | 25 | ☐ | |

### A07 数学函数

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-MATH-ADD` | L1 | 逐元素加法 | AddRmsNorm、ApplyRotaryPosEmb、Exp、GatherExp | 28 | ☐ | |
| `FA-MATH-MUL` | L1 | 逐元素乘法（缩放类算子） | ApplyRotaryPosEmb、Exp、GatherExp、GroupedMatmulSwiGlu、MaskedScale、SwiGlu | 54 | ☐ | |
| `FA-MATH-EXP` | L1 | tl.exp 指数（Exp / Softmax / CEL 的 logsumexp） | CrossEntropyLoss、Exp、GatherExp、MaskedSoftmax、MoeGatingTopKSoftmax、Softmax | 60 | ☐ | |
| `FA-MATH-SQRT` | L1 | tl.sqrt / rsqrt（RmsNorm） | AddRmsNorm、RmsNorm、RmsNormLinear | 21 | ☐ | |
| `FA-MATH-GELU` | L1 | GELU 非线性（含 tanh 近似变体） | Gelu、LinearGelu、MaskedGelu | 24 | ☐ | |
| `FA-MATH-SIGMOID` | L1 | Sigmoid 非线性（SwiGlu 的 swish 分支） | GroupedMatmulSwiGlu、SwiGlu | 12 | ☐ | |
| `FA-MATH-MAXMIN` | L1 | 逐元素取大/取小（Maximum） | Maximum | 20 | ☐ | |
| `FA-MATH-SINCOS` | L1 | sin/cos 三角函数（旋转位置编码的角度对） | ApplyRotaryPosEmb | 6 | ☐ | |

### A08 归约

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-REDUCE-SUM` | L1 | tl.sum 沿轴归约（Softmax 分母 / RmsNorm / CEL） | AddRmsNorm、CrossEntropyLoss、MaskedSoftmax、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Softmax、UnsortedSegmentSum | 75 | ☐ | |
| `FA-REDUCE-MAX` | L1 | tl.max 沿轴归约（Softmax 最大值减 / ArgMax） | ArgMax、MaskedSoftmax、MoeGatingTopKSoftmax、Softmax | 44 | ☐ | |
| `FA-REDUCE-MIN` | L1 | tl.min 前缀扫描（Cummin） | Cummin | 20 | ☐ | |
| `FA-REDUCE-ARG` | L2 | 归约并输出取值位置（ArgMax / Cummin 的 indices） | ArgMax、Cummin、MoeGatingTopKSoftmax | 45 | ☐ | |

### A09 归一化

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-NORM-SOFTMAX` | L1 | Softmax 归一化 | MaskedSoftmax、MoeGatingTopKSoftmax、Softmax | 24 | ☐ | |
| `FA-NORM-RMS` | L2 | RMSNorm 归一化 | AddRmsNorm、RmsNorm、RmsNormLinear | 21 | ☐ | |

### A10 原子操作

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-CONCUR-ACCUM` | L2 | 原子加法 / 顺序无关累加（重复索引下「和」与写入顺序无关是语义要求） | Scatter、UnsortedSegmentSum | 12 | ☐ | |

### A12 迭代器

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-CTRL-STATIC-UNROLL` | L1 | 小循环次数的静态展开（group_list ≤ 8 的分组矩阵乘） | GroupedMatmul | 20 | ☐ | |

### A13 数值稳定性

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-STABLE-LOGSUMEXP` | L2 | log-sumexp 稳定实现（CrossEntropyLoss） | CrossEntropyLoss | 20 | ☐ | |
| `FA-STABLE-MAXSUB` | L1 | 先减最大值再 exp 的稳定 softmax | MaskedSoftmax、MoeGatingTopKSoftmax、Softmax | 24 | ☐ | |
| `FA-NUM-ZEROS` | L1 | 全零输入（退化分布下的边界行为） | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 52 | ☐ | |
| `FA-NUM-MIXED` | L1 | 常规与零混合输入（稀疏化边界） | ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、Scatter、Softmax、Transpose、TransposeLinear、UnsortedSegmentSum | 47 | ☐ | |
| `FA-NUM-EXTREME` | L2 | 大幅值输入（±20σ，考验溢出与稳定减） | ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、Gelu、GroupedMatmul、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、Scatter、Softmax、Transpose、UnsortedSegmentSum | 21 | ☐ | |

### A14 精度

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-DT-FP32` | L1 | float32 路径 | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 115 | ☐ | |
| `FA-DT-FP16` | L1 | float16 路径（窄尾数） | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 73 | ☐ | |
| `FA-DT-BF16` | L1 | bfloat16 路径（窄尾数宽指数） | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 47 | ☐ | |
| `FA-DT-INT8` | L1 | int8 路径 | Gather、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、Transpose | 11 | ☐ | |
| `FA-DT-UINT8` | L1 | uint8 路径（无符号语义） | MaskedGelu、MaskedScale、MaskedSoftmax | 5 | ☐ | |
| `FA-DT-INT16` | L1 | int16 路径 | Transpose | 2 | ☐ | |
| `FA-DT-INT32` | L1 | int32 路径 | ArgMax、CrossEntropyLoss、Cummin、Gather、Maximum、Scatter、Transpose、UnsortedSegmentSum | 17 | ☐ | |
| `FA-DT-INT64` | L1 | int64 路径（宽索引） | ArgMax、CrossEntropyLoss、Cummin、Gather、Maximum、Scatter、Transpose、UnsortedSegmentSum | 19 | ☐ | |

### A15 形态与规模

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-SHAPE-FLAT` | L1 | 一维扁平形态 | AddRmsNorm、ArgMax、Cummin、Exp、Gather、GatherExp、Gelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、Scatter、Softmax、SwiGlu、UnsortedSegmentSum | 56 | ☐ | |
| `FA-SHAPE-SQUARE` | L1 | 二维方阵形态 | ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedScale、Maximum、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear | 39 | ☐ | |
| `FA-SHAPE-SKINNY` | L2 | 长宽比 ≥8 的细长形态（行/列分块策略差异） | ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedScale、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Softmax、Transpose、TransposeLinear、UnsortedSegmentSum | 31 | ☐ | |
| `FA-SHAPE-CUBE` | L2 | 三维立体形态 | AddRmsNorm、ArgMax、Cummin、Exp、Gather、GatherExp、Gelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、RmsNorm、Scatter、Softmax、Transpose | 15 | ☐ | |
| `FA-LOAD-SMALL` | L1 | 小负载（≤4K 元素）——启动开销主导区 | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 179 | ☐ | |
| `FA-LOAD-MEDIUM` | L1 | 中负载（16K–256K 元素） | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 58 | ☐ | |
| `FA-LOAD-LARGE` | L2 | 大负载（256K–2M 元素） | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 26 | ☐ | |
| `FA-LOAD-XLARGE` | L3 | 超大负载（≥2M 元素）——性能赛道主战场 | AddRmsNorm、ApplyRotaryPosEmb、ArgMax、CrossEntropyLoss、Cummin、Exp、Gather、GatherExp、Gelu、GroupedMatmul、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedScale、MaskedSoftmax、Maximum、MoeGatingTopKSoftmax、RmsNorm、RmsNormLinear、Scatter、Softmax、SwiGlu、Transpose、TransposeLinear、UnsortedSegmentSum | 26 | ☐ | |

### A16 算子组合（融合）

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-FUSE-2STAGE` | L2 | 两段融合（组合表声明两个语义段一次执行） | AddRmsNorm、ApplyRotaryPosEmb、GatherExp、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedSoftmax、MoeGatingTopKSoftmax、RmsNormLinear、TransposeLinear、UnsortedSegmentSum | 73 | ☐ | |
| `FA-FUSE-INTERM-ELIDE` | L2 | 融合省去中间结果读写（段 1 输出直接喂段 2，不落显存） | AddRmsNorm、ApplyRotaryPosEmb、GatherExp、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedSoftmax、MoeGatingTopKSoftmax、RmsNormLinear、TransposeLinear、UnsortedSegmentSum | 73 | ☐ | |
| `FA-FUSE-CROSS-CATEGORY` | L3 | 跨语义族融合（如收缩 × 逐元素、掩码 × 归一化） | AddRmsNorm、GatherExp、GroupedMatmulSwiGlu、LinearGelu、MaskedGelu、MaskedSoftmax、MoeGatingTopKSoftmax、RmsNormLinear、TransposeLinear、UnsortedSegmentSum | 67 | ☐ | |
| `FA-FUSE-ELEM-INTRA` | L2 | 逐元素族内融合（两段均为逐元素语义，如三角函数 × 旋转） | ApplyRotaryPosEmb | 6 | ☐ | |
| `FA-FUSE-RED-INTRA` | L3 | 含归约/归一化段的融合（掩码/归一化喂收缩类前缀） | AddRmsNorm、MaskedSoftmax、MoeGatingTopKSoftmax、RmsNormLinear、UnsortedSegmentSum | 35 | ☐ | |

### A17 优化手法（性能负载型）

| 能力点 | 层级 | 白话描述 | 盯它的算子 | 题库用例数 | 确认 | 意见 |
|---|---|---|---|---|---|---|
| `FA-PAT-TILE-GEMM` | L2 | 共享内存分块（收缩类且参与元素量 ≥ 2^18——朴素全局访存慢一个量级） | GroupedMatmul | 3 | ☐ | |
| `FA-PAT-SPLIT-K` | L3 | 沿归约维切分并行（M ≤ 64 且 K ≥ 2^16 的高瘦矩阵——M/N 并行度不足） | GroupedMatmul | 1 | ☐ | |
| `FA-PAT-VEC-BROADCAST` | L2 | 偏置向量跨行广播驻留（带 bias 的收缩且 M ≥ 2048） | GroupedMatmul | 2 | ☐ | |
| `FA-PAT-VEC-LOAD` | L2 | 向量化访存（逐元素/掩码类、连续布局、最内维 ≥ 4096 且 64 整除对齐） | Exp、Gelu、MaskedScale | 4 | ☐ | |
| `FA-PAT-MASK-TAIL` | L2 | 边界掩码访存（最内维 ≥ 2048 且不整除向量宽度——尾部不满块） | Exp、Gelu、MaskedScale | 3 | ☐ | |
| `FA-PAT-MULTIPASS-REDUCE` | L3 | 多趟归约（归约维 ≥ 2^20——单趟寄存器放不下） | ArgMax、Cummin | 3 | ☐ | |
| `FA-PAT-COALESCE-GATHER` | L2 | 聚集访存/内存局部性（索引类且主输入 ≥ 2^16 的大表查访） | Gather | 4 | ☐ | |
| `FA-PAT-SCAN-BLOCKED` | L3 | 分块两趟扫描（扫描类且规模 ≥ 2^20——块内扫描 + 块间携带） | Cummin | 2 | ☐ | |

## 二、预留条目（不计入分母，5 条）

| 能力点 | 为什么暂不进分母 | 确认 | 意见 |
|---|---|---|---|
| `FA-CREATE-ARANGE` | tl.arange（所有用例隐含，无法差异化定向） | ☐ | |
| `FA-CREATE-FULL` | tl.full / tl.zeros（同上） | ☐ | |
| `FA-CTRL-DYNAMIC-FOR` | range 动态循环（同上） | ☐ | |
| `FA-STORE-CONTIG` | 连续写回 tl.store（输出连续性是执行期契约 SUB-EXE-002 强制的， 所有用例平凡满足，无法用参数差异化定向） | ☐ | |
| `FA-SHAPE-BROADCAST` | 二元广播（maximum 的参考输入当前同形；需输入构造层支持 广播形状信号，随下批输入构造扩展转正） | ☐ | |

---

*重新生成：`python scripts/build_capability_checklist.py`（输出与数据逐字对账，见 tests/core/test_docs.py）*
