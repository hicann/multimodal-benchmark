# 测试集演化记录（自动生成，审计用）

> 由 `scripts/corpus_evolution.py` 从 `corpus/<版本>/manifest.json` 推导生成，
> 重新生成：`python scripts/corpus_evolution.py --all --out docs/corpus-evolution.md`。
> 对账测试钉住本文档与再生成结果一致，清单变了不重出文档即测试红。

本表只含清单可推导的事实。与它互补的三类留痕：

1. **门禁剔除记录**：`manifest.json` 的 `excluded` 字段，逐条含原因，随清单
   哈希冻结——事后抹掉或改写会破坏清单指纹，评测启动时即被检出；
2. **生成日志**：`corpus/<版本>/generation-log.yaml`，构建过程每步一行
   （2026-10-14 起启用；此前构建的终端输出未留存）；
3. **人工叙述**：版本内变更的背景与理由（为什么重冻结、为什么再分配）
   在方案文档的落地实况章节，如
   `docs/composition-and-pattern-dimensions.md` §6。

## 迁移总览

| 迁移 | 用例数 | 算子 | 用例（增/删/改） |
|---|---|---|---|
| v0.3-multi-input-frozen → v0.4-cov20-frozen | 60 → 240 | +0 / −0 | +240 / −60 / 改 0 |
| v0.4-cov20-frozen → v0.5-cov20-frozen | 240 → 289 | +11 / −0 | +92 / −43 / 改 25 |
| v0.5-cov20-frozen → v0.6-ascend-ops | 289 → 220 | +11 / −25 | +220 / −289 / 改 0 |
| v0.6-ascend-ops → v0.7-ascend-contract | 220 → 220 | +0 / −0 | +17 / −17 / 改 85 |
| v0.7-ascend-contract → v0.8-official-candidates | 220 → 1360 | +68 / −11 | +1360 / −220 / 改 0 |
| v0.8-official-candidates → v0.8.1-official-candidates | 1360 → 1360 | +0 / −0 | +0 / −0 / 改 0 |
| v0.8.1-official-candidates → v0.9-official-candidates | 1360 → 1360 | +0 / −0 | +1360 / −1360 / 改 0 |
| v0.9-official-candidates → v0.10-coverage-candidates | 1360 → 1377 | +6 / −0 | +17 / −0 / 改 0 |
| v0.10-coverage-candidates → v0.11-implementation-candidates | 1377 → 1377 | +0 / −0 | +0 / −0 / 改 0 |
| v0.11-implementation-candidates → v0.11.1-contract-candidates | 1377 → 1377 | +0 / −0 | +0 / −0 / 改 0 |
| v0.11.1-contract-candidates → v0.12-compat-candidates | 1377 → 1377 | +0 / −0 | +0 / −0 / 改 0 |
| v0.12-compat-candidates → v0.13-evidence-candidates | 1377 → 1377 | +0 / −0 | +0 / −0 / 改 0 |
| v0.13-evidence-candidates → v0.14-backend-evidence-candidates | 1377 → 1377 | +0 / −0 | +0 / −0 / 改 0 |
| v0.14-backend-evidence-candidates → v0.15-backend-capture-fix-candidates | 1377 → 1377 | +0 / −0 | +0 / −0 / 改 0 |
| v0.15-backend-capture-fix-candidates → v0.16-backend-pattern-certification-candidates | 1377 → 1377 | +0 / −0 | +0 / −0 / 改 0 |
| v0.16-backend-pattern-certification-candidates → v0.17-pattern-completion-candidates | 1377 → 1385 | +0 / −0 | +8 / −0 / 改 0 |
| v0.17-pattern-completion-candidates → v0.18-scan-certification-candidates | 1385 → 1385 | +0 / −0 | +0 / −0 / 改 0 |
| v0.18-scan-certification-candidates → v0.19-workload-completion-candidates | 1385 → 1486 | +0 / −0 | +101 / −0 / 改 0 |

---
## v0.3-multi-input-frozen → v0.4-cov20-frozen

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-08T03:26:29.121180+00:00 | 2026-10-09T11:31:21.296463+00:00 |
| 清单哈希 | `7114c9f73a25…` | `020f8d62b573…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 0.9-mvp | 1.0-corpus |
| 算子数 | 14 | 14 |
| 用例数 | 60 | 240 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

| 算子 | 旧 | 新 | Δ |
|---|---|---|---|
| ArgMax | 5 | 20 | +15 |
| CrossEntropyLoss | 5 | 20 | +15 |
| Cummin | 5 | 20 | +15 |
| Exp | 3 | 11 | +8 |
| Gather | 5 | 20 | +15 |
| Gelu | 2 | 9 | +7 |
| GroupedMatmul | 5 | 20 | +15 |
| MaskedScale | 5 | 20 | +15 |
| Maximum | 5 | 20 | +15 |
| RmsNorm | 2 | 9 | +7 |
| Scatter | 5 | 20 | +15 |
| Softmax | 3 | 11 | +8 |
| SwiGlu | 5 | 20 | +15 |
| Transpose | 5 | 20 | +15 |
（其余 0 个算子用例数不变）

### 用例级变化

- 新增用例：240 个（超过 20，按算子折叠）
  - arg_max：20
  - cross_entropy_loss：20
  - cummin：20
  - exp：11
  - gather：20
  - gelu：9
  - grouped_matmul：20
  - masked_scale：20
  - maximum：20
  - rms_norm：9
  - scatter：20
  - softmax：11
  - swi_glu：20
  - transpose：20
- 移除用例：60 个（超过 20，按算子折叠）
  - arg_max：5
  - cross_entropy_loss：5
  - cummin：5
  - exp：3
  - gather：5
  - gelu：2
  - grouped_matmul：5
  - masked_scale：5
  - maximum：5
  - rms_norm：2
  - scatter：5
  - softmax：3
  - swi_glu：5
  - transpose：5
- 同编号但内容变更：无
- 内容未变的用例：0 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| ArgMax/LARGE | 0 | 1 | +1 |
| ArgMax/MEDIUM | 0 | 2 | +2 |
| ArgMax/SMALL | 5 | 16 | +11 |
| ArgMax/XLARGE | 0 | 1 | +1 |
| CrossEntropyLoss/LARGE | 0 | 2 | +2 |
| CrossEntropyLoss/MEDIUM | 0 | 7 | +7 |
| CrossEntropyLoss/SMALL | 5 | 10 | +5 |
| CrossEntropyLoss/XLARGE | 0 | 1 | +1 |
| Cummin/LARGE | 0 | 1 | +1 |
| Cummin/MEDIUM | 0 | 2 | +2 |
| Cummin/SMALL | 5 | 16 | +11 |
| Cummin/XLARGE | 0 | 1 | +1 |
| Exp/LARGE | 0 | 1 | +1 |
| Exp/MEDIUM | 0 | 2 | +2 |
| Exp/SMALL | 3 | 7 | +4 |
| Exp/XLARGE | 0 | 1 | +1 |
| Gather/LARGE | 0 | 1 | +1 |
| Gather/MEDIUM | 0 | 2 | +2 |
| Gather/SMALL | 5 | 16 | +11 |
| Gather/XLARGE | 0 | 1 | +1 |
| Gelu/LARGE | 0 | 1 | +1 |
| Gelu/MEDIUM | 0 | 2 | +2 |
| Gelu/SMALL | 2 | 5 | +3 |
| Gelu/XLARGE | 0 | 1 | +1 |
| GroupedMatmul/LARGE | 0 | 1 | +1 |
| GroupedMatmul/MEDIUM | 0 | 12 | +12 |
| GroupedMatmul/SMALL | 5 | 6 | +1 |
| GroupedMatmul/XLARGE | 0 | 1 | +1 |
| MaskedScale/LARGE | 0 | 1 | +1 |
| MaskedScale/MEDIUM | 0 | 2 | +2 |
| MaskedScale/SMALL | 5 | 16 | +11 |
| MaskedScale/XLARGE | 0 | 1 | +1 |
| Maximum/LARGE | 0 | 1 | +1 |
| Maximum/MEDIUM | 0 | 3 | +3 |
| Maximum/SMALL | 5 | 15 | +10 |
| Maximum/XLARGE | 0 | 1 | +1 |
| RmsNorm/LARGE | 0 | 1 | +1 |
| RmsNorm/MEDIUM | 0 | 2 | +2 |
| RmsNorm/SMALL | 2 | 5 | +3 |
| RmsNorm/XLARGE | 0 | 1 | +1 |
| Scatter/LARGE | 0 | 1 | +1 |
| Scatter/MEDIUM | 0 | 3 | +3 |
| Scatter/SMALL | 5 | 15 | +10 |
| Scatter/XLARGE | 0 | 1 | +1 |
| Softmax/LARGE | 0 | 1 | +1 |
| Softmax/MEDIUM | 0 | 3 | +3 |
| Softmax/SMALL | 3 | 6 | +3 |
| Softmax/XLARGE | 0 | 1 | +1 |
| SwiGlu/LARGE | 0 | 1 | +1 |
| SwiGlu/MEDIUM | 0 | 3 | +3 |
| SwiGlu/SMALL | 5 | 15 | +10 |
| SwiGlu/XLARGE | 0 | 1 | +1 |
| Transpose/LARGE | 0 | 1 | +1 |
| Transpose/MEDIUM | 0 | 2 | +2 |
| Transpose/SMALL | 5 | 16 | +11 |
| Transpose/XLARGE | 0 | 1 | +1 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| ArgMax/bfloat16 | 1 | 4 | +3 |
| ArgMax/float16 | 1 | 5 | +4 |
| ArgMax/float32 | 1 | 6 | +5 |
| ArgMax/int32 | 1 | 3 | +2 |
| ArgMax/int64 | 1 | 2 | +1 |
| CrossEntropyLoss/bfloat16 | 1 | 4 | +3 |
| CrossEntropyLoss/float16 | 1 | 5 | +4 |
| CrossEntropyLoss/float32 | 1 | 5 | +4 |
| CrossEntropyLoss/int32 | 1 | 3 | +2 |
| CrossEntropyLoss/int64 | 1 | 3 | +2 |
| Cummin/bfloat16 | 1 | 3 | +2 |
| Cummin/float16 | 1 | 4 | +3 |
| Cummin/float32 | 1 | 8 | +7 |
| Cummin/int32 | 1 | 2 | +1 |
| Cummin/int64 | 1 | 3 | +2 |
| Exp/float16 | 1 | 4 | +3 |
| Exp/float32 | 1 | 6 | +5 |
| Gather/bfloat16 | 1 | 2 | +1 |
| Gather/float16 | 1 | 3 | +2 |
| Gather/float32 | 1 | 8 | +7 |
| Gather/int32 | 1 | 3 | +2 |
| Gather/int64 | 0 | 2 | +2 |
| Gather/int8 | 1 | 2 | +1 |
| Gelu/bfloat16 | 0 | 1 | +1 |
| Gelu/float16 | 1 | 3 | +2 |
| Gelu/float32 | 1 | 5 | +4 |
| GroupedMatmul/bfloat16 | 1 | 5 | +4 |
| GroupedMatmul/float16 | 2 | 6 | +4 |
| GroupedMatmul/float32 | 2 | 9 | +7 |
| MaskedScale/bfloat16 | 1 | 3 | +2 |
| MaskedScale/float16 | 1 | 5 | +4 |
| MaskedScale/float32 | 1 | 7 | +6 |
| MaskedScale/int8 | 1 | 2 | +1 |
| MaskedScale/uint8 | 1 | 3 | +2 |
| Maximum/bfloat16 | 1 | 3 | +2 |
| Maximum/float16 | 1 | 3 | +2 |
| Maximum/float32 | 1 | 5 | +4 |
| Maximum/int32 | 1 | 2 | +1 |
| Maximum/int64 | 0 | 4 | +4 |
| Maximum/int8 | 1 | 3 | +2 |
| RmsNorm/bfloat16 | 0 | 1 | +1 |
| RmsNorm/float16 | 1 | 3 | +2 |
| RmsNorm/float32 | 1 | 5 | +4 |
| Scatter/bfloat16 | 1 | 4 | +3 |
| Scatter/float16 | 1 | 4 | +3 |
| Scatter/float32 | 1 | 7 | +6 |
| Scatter/int32 | 1 | 2 | +1 |
| Scatter/int64 | 1 | 3 | +2 |
| Softmax/float16 | 1 | 3 | +2 |
| Softmax/float32 | 1 | 7 | +6 |
| SwiGlu/bfloat16 | 1 | 6 | +5 |
| SwiGlu/float16 | 2 | 6 | +4 |
| SwiGlu/float32 | 2 | 8 | +6 |
| Transpose/bfloat16 | 1 | 3 | +2 |
| Transpose/float16 | 1 | 2 | +1 |
| Transpose/float32 | 1 | 6 | +5 |
| Transpose/int16 | 1 | 2 | +1 |
| Transpose/int32 | 0 | 2 | +2 |
| Transpose/int64 | 0 | 3 | +3 |
| Transpose/int8 | 1 | 2 | +1 |

### 覆盖原子得失

- 原子覆盖数：0 → 55
- 新覆盖：FA-ATTR-LIST、FA-ATTR-NONDEFAULT、FA-AXIS-FIRST、FA-AXIS-INNER、FA-AXIS-LAST、FA-DT-BF16、FA-DT-FP16、FA-DT-FP32、FA-DT-INT16、FA-DT-INT32、FA-DT-INT64、FA-DT-INT8、FA-DT-UINT8、FA-IDX-GATHER、FA-IDX-PERMUTE、FA-IDX-SCATTER、FA-IDX-WHERE、FA-LINALG-BIAS、FA-LINALG-DOT、FA-LOAD-ALIGNED、FA-LOAD-CONTIG、FA-LOAD-LARGE、FA-LOAD-MASKED-EDGE、FA-LOAD-MEDIUM、FA-LOAD-SMALL、FA-LOAD-STRIDED、FA-LOAD-XLARGE、FA-MATH-ADD、FA-MATH-EXP、FA-MATH-GELU、FA-MATH-MAXMIN、FA-MATH-MUL、FA-MATH-SIGMOID、FA-MATH-SQRT、FA-MULTI-INPUT、FA-MULTI-OUTPUT、FA-NORM-RMS、FA-NORM-SOFTMAX、FA-NUM-EXTREME、FA-NUM-MIXED、FA-NUM-ZEROS、FA-PROG-GRID1D、FA-PROG-GRID2D、FA-PROG-GRID3D、FA-PROG-GRID4D、FA-REDUCE-ARG、FA-REDUCE-MAX、FA-REDUCE-MIN、FA-REDUCE-SUM、FA-SHAPE-CUBE、FA-SHAPE-FLAT、FA-SHAPE-SKINNY、FA-SHAPE-SQUARE、FA-STABLE-LOGSUMEXP、FA-STABLE-MAXSUB
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.3-multi-input-frozen：无 generation-log.yaml（该版构建早于日志机制上线）
- v0.4-cov20-frozen：无 generation-log.yaml（该版构建早于日志机制上线）


## v0.4-cov20-frozen → v0.5-cov20-frozen

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-09T11:31:21.296463+00:00 | 2026-10-13T09:34:27.876149+00:00 |
| 清单哈希 | `020f8d62b573…` | `2771f8b88970…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.0-corpus | 1.1-fusion |
| 算子数 | 14 | 25 |
| 用例数 | 240 | 289 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子 11 个：AddRmsNorm（FusedComposite / L3）、ApplyRotaryPosEmb（FusedComposite / L2）、GatherExp（FusedComposite / L2）、GroupedMatmulSwiGlu（FusedComposite / L3）、LinearGelu（FusedComposite / L3）、MaskedGelu（FusedComposite / L2）、MaskedSoftmax（FusedComposite / L2）、MoeGatingTopKSoftmax（FusedComposite / L3）、RmsNormLinear（FusedComposite / L3）、TransposeLinear（FusedComposite / L3）、UnsortedSegmentSum（ScatterUpdate / L2）
- 移除算子：无

### 逐算子用例数变化

| 算子 | 旧 | 新 | Δ |
|---|---|---|---|
| AddRmsNorm | 0 | 6 | +6 |
| ApplyRotaryPosEmb | 0 | 6 | +6 |
| Exp | 11 | 10 | -1 |
| GatherExp | 0 | 6 | +6 |
| Gelu | 9 | 10 | +1 |
| GroupedMatmulSwiGlu | 0 | 6 | +6 |
| LinearGelu | 0 | 6 | +6 |
| MaskedGelu | 0 | 8 | +8 |
| MaskedSoftmax | 0 | 8 | +8 |
| MoeGatingTopKSoftmax | 0 | 5 | +5 |
| RmsNormLinear | 0 | 6 | +6 |
| Scatter | 20 | 10 | -10 |
| SwiGlu | 20 | 6 | -14 |
| TransposeLinear | 0 | 6 | +6 |
| UnsortedSegmentSum | 0 | 10 | +10 |
（其余 10 个算子用例数不变）

### 用例级变化

- 新增用例：92 个（超过 20，按算子折叠）
  - add_rms_norm：6
  - apply_rotary_pos_emb：6
  - cummin：1
  - exp：2
  - gather：2
  - gather_exp：6
  - gelu：1
  - grouped_matmul：2
  - grouped_matmul_swiglu：6
  - linear_gelu：6
  - masked_gelu：8
  - masked_scale：1
  - masked_softmax：8
  - moe_gating_top_k_softmax：5
  - rms_norm_linear：6
  - scatter：10
  - transpose_linear：6
  - unsorted_segment_sum：10
- 移除用例：43 个（超过 20，按算子折叠）
  - cummin：1
  - exp：3
  - gather：2
  - grouped_matmul：2
  - masked_scale：1
  - scatter：20
  - swi_glu：14
- **同编号但内容变更** 25 个（编号是弱键：不编码形状/数值，跨版本同编号不等于同题；字段变更频次：feature_atoms×25）
  - 超过 20 个，按算子折叠：ArgMax×1、Cummin×1、Exp×1、Gather×2、Gelu×1、GroupedMatmul×18、MaskedScale×1
- 内容未变的用例：172 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| AddRmsNorm/LARGE | 0 | 1 | +1 |
| AddRmsNorm/MEDIUM | 0 | 1 | +1 |
| AddRmsNorm/SMALL | 0 | 3 | +3 |
| AddRmsNorm/XLARGE | 0 | 1 | +1 |
| ApplyRotaryPosEmb/LARGE | 0 | 1 | +1 |
| ApplyRotaryPosEmb/MEDIUM | 0 | 1 | +1 |
| ApplyRotaryPosEmb/SMALL | 0 | 3 | +3 |
| ApplyRotaryPosEmb/XLARGE | 0 | 1 | +1 |
| Exp/MEDIUM | 2 | 3 | +1 |
| Exp/SMALL | 7 | 5 | -2 |
| GatherExp/LARGE | 0 | 1 | +1 |
| GatherExp/MEDIUM | 0 | 1 | +1 |
| GatherExp/SMALL | 0 | 3 | +3 |
| GatherExp/XLARGE | 0 | 1 | +1 |
| Gelu/MEDIUM | 2 | 3 | +1 |
| GroupedMatmul/MEDIUM | 12 | 11 | -1 |
| GroupedMatmul/XLARGE | 1 | 2 | +1 |
| GroupedMatmulSwiGlu/LARGE | 0 | 1 | +1 |
| GroupedMatmulSwiGlu/MEDIUM | 0 | 1 | +1 |
| GroupedMatmulSwiGlu/SMALL | 0 | 3 | +3 |
| GroupedMatmulSwiGlu/XLARGE | 0 | 1 | +1 |
| LinearGelu/LARGE | 0 | 1 | +1 |
| LinearGelu/MEDIUM | 0 | 1 | +1 |
| LinearGelu/SMALL | 0 | 3 | +3 |
| LinearGelu/XLARGE | 0 | 1 | +1 |
| MaskedGelu/LARGE | 0 | 1 | +1 |
| MaskedGelu/MEDIUM | 0 | 1 | +1 |
| MaskedGelu/SMALL | 0 | 5 | +5 |
| MaskedGelu/XLARGE | 0 | 1 | +1 |
| MaskedScale/MEDIUM | 2 | 3 | +1 |
| MaskedScale/SMALL | 16 | 15 | -1 |
| MaskedSoftmax/LARGE | 0 | 1 | +1 |
| MaskedSoftmax/MEDIUM | 0 | 1 | +1 |
| MaskedSoftmax/SMALL | 0 | 5 | +5 |
| MaskedSoftmax/XLARGE | 0 | 1 | +1 |
| MoeGatingTopKSoftmax/LARGE | 0 | 1 | +1 |
| MoeGatingTopKSoftmax/MEDIUM | 0 | 1 | +1 |
| MoeGatingTopKSoftmax/SMALL | 0 | 2 | +2 |
| MoeGatingTopKSoftmax/XLARGE | 0 | 1 | +1 |
| RmsNormLinear/LARGE | 0 | 1 | +1 |
| RmsNormLinear/MEDIUM | 0 | 1 | +1 |
| RmsNormLinear/SMALL | 0 | 3 | +3 |
| RmsNormLinear/XLARGE | 0 | 1 | +1 |
| Scatter/MEDIUM | 3 | 2 | -1 |
| Scatter/SMALL | 15 | 6 | -9 |
| SwiGlu/MEDIUM | 3 | 1 | -2 |
| SwiGlu/SMALL | 15 | 3 | -12 |
| TransposeLinear/LARGE | 0 | 1 | +1 |
| TransposeLinear/MEDIUM | 0 | 1 | +1 |
| TransposeLinear/SMALL | 0 | 3 | +3 |
| TransposeLinear/XLARGE | 0 | 1 | +1 |
| UnsortedSegmentSum/LARGE | 0 | 1 | +1 |
| UnsortedSegmentSum/MEDIUM | 0 | 2 | +2 |
| UnsortedSegmentSum/SMALL | 0 | 6 | +6 |
| UnsortedSegmentSum/XLARGE | 0 | 1 | +1 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| AddRmsNorm/bfloat16 | 0 | 1 | +1 |
| AddRmsNorm/float16 | 0 | 2 | +2 |
| AddRmsNorm/float32 | 0 | 3 | +3 |
| ApplyRotaryPosEmb/bfloat16 | 0 | 2 | +2 |
| ApplyRotaryPosEmb/float16 | 0 | 3 | +3 |
| ApplyRotaryPosEmb/float32 | 0 | 1 | +1 |
| Exp/float32 | 6 | 5 | -1 |
| Gather/float16 | 3 | 4 | +1 |
| Gather/float32 | 8 | 7 | -1 |
| GatherExp/bfloat16 | 0 | 1 | +1 |
| GatherExp/float16 | 0 | 2 | +2 |
| GatherExp/float32 | 0 | 3 | +3 |
| Gelu/float32 | 5 | 6 | +1 |
| GroupedMatmul/bfloat16 | 5 | 6 | +1 |
| GroupedMatmul/float16 | 6 | 5 | -1 |
| GroupedMatmulSwiGlu/bfloat16 | 0 | 1 | +1 |
| GroupedMatmulSwiGlu/float16 | 0 | 2 | +2 |
| GroupedMatmulSwiGlu/float32 | 0 | 3 | +3 |
| LinearGelu/bfloat16 | 0 | 1 | +1 |
| LinearGelu/float16 | 0 | 2 | +2 |
| LinearGelu/float32 | 0 | 3 | +3 |
| MaskedGelu/bfloat16 | 0 | 1 | +1 |
| MaskedGelu/float16 | 0 | 2 | +2 |
| MaskedGelu/float32 | 0 | 3 | +3 |
| MaskedGelu/int8 | 0 | 1 | +1 |
| MaskedGelu/uint8 | 0 | 1 | +1 |
| MaskedSoftmax/bfloat16 | 0 | 1 | +1 |
| MaskedSoftmax/float16 | 0 | 2 | +2 |
| MaskedSoftmax/float32 | 0 | 3 | +3 |
| MaskedSoftmax/int8 | 0 | 1 | +1 |
| MaskedSoftmax/uint8 | 0 | 1 | +1 |
| MoeGatingTopKSoftmax/bfloat16 | 0 | 2 | +2 |
| MoeGatingTopKSoftmax/float32 | 0 | 3 | +3 |
| RmsNormLinear/bfloat16 | 0 | 1 | +1 |
| RmsNormLinear/float16 | 0 | 2 | +2 |
| RmsNormLinear/float32 | 0 | 3 | +3 |
| Scatter/bfloat16 | 4 | 1 | -3 |
| Scatter/float16 | 4 | 3 | -1 |
| Scatter/float32 | 7 | 4 | -3 |
| Scatter/int32 | 2 | 1 | -1 |
| Scatter/int64 | 3 | 1 | -2 |
| SwiGlu/bfloat16 | 6 | 1 | -5 |
| SwiGlu/float16 | 6 | 3 | -3 |
| SwiGlu/float32 | 8 | 2 | -6 |
| TransposeLinear/bfloat16 | 0 | 1 | +1 |
| TransposeLinear/float16 | 0 | 2 | +2 |
| TransposeLinear/float32 | 0 | 3 | +3 |
| UnsortedSegmentSum/bfloat16 | 0 | 1 | +1 |
| UnsortedSegmentSum/float16 | 0 | 2 | +2 |
| UnsortedSegmentSum/float32 | 0 | 5 | +5 |
| UnsortedSegmentSum/int32 | 0 | 1 | +1 |
| UnsortedSegmentSum/int64 | 0 | 1 | +1 |

### 覆盖原子得失

- 原子覆盖数：55 → 71
- 新覆盖：FA-CONCUR-ACCUM、FA-CTRL-STATIC-UNROLL、FA-FUSE-2STAGE、FA-FUSE-CROSS-CATEGORY、FA-FUSE-ELEM-INTRA、FA-FUSE-INTERM-ELIDE、FA-FUSE-RED-INTRA、FA-MATH-SINCOS、FA-PAT-COALESCE-GATHER、FA-PAT-MASK-TAIL、FA-PAT-MULTIPASS-REDUCE、FA-PAT-SCAN-BLOCKED、FA-PAT-SPLIT-K、FA-PAT-TILE-GEMM、FA-PAT-VEC-BROADCAST、FA-PAT-VEC-LOAD
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.4-cov20-frozen：无 generation-log.yaml（该版构建早于日志机制上线）
- v0.5-cov20-frozen：generation-log.yaml 共 9 条记录（日志起点 2026-10-14T07:58:58.348781+00:00，此前构建无日志）


## v0.5-cov20-frozen → v0.6-ascend-ops

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-13T09:34:27.876149+00:00 | 2026-10-22T17:05:59.140708+00:00 |
| 清单哈希 | `2771f8b88970…` | `e6ef84915692…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | v1.1-fusion |
| 算子数 | 25 | 11 |
| 用例数 | 289 | 220 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子 11 个：AssignReqToTokenPool（ScatterUpdate / L3）、BinnedGatherScatter（IndexGather / L3）、CausalConv1d（FusedComposite / L3）、DecodeGroupedAttention（Contraction / L3）、DiscreteMemoryAccess（IndexGather / L2）、ExtractSlice（LayoutTransform / L2）、GatherScatter（IndexGather / L3）、InsertSlice（LayoutTransform / L2）、PaddedGatherScatter（IndexGather / L3）、VectorAdd（Elementwise / L1）、VectorCmp（Normalization / L2）
- 移除算子 25 个：AddRmsNorm（FusedComposite / L3）、ApplyRotaryPosEmb（FusedComposite / L2）、ArgMax（SortSelect / L2）、CrossEntropyLoss（NumericalStable / L2）、Cummin（Reduction / L2）、Exp（Elementwise / L1）、Gather（IndexGather / L2）、GatherExp（FusedComposite / L2）、Gelu（Elementwise / L1）、GroupedMatmul（Contraction / L3）、GroupedMatmulSwiGlu（FusedComposite / L3）、LinearGelu（FusedComposite / L3）、MaskedGelu（FusedComposite / L2）、MaskedScale（MaskPredicate / L1）、MaskedSoftmax（FusedComposite / L2）、Maximum（Broadcast / L2）、MoeGatingTopKSoftmax（FusedComposite / L3）、RmsNorm（Normalization / L2）、RmsNormLinear（FusedComposite / L3）、Scatter（ScatterUpdate / L2）、Softmax（Normalization / L2）、SwiGlu（FusedComposite / L1）、Transpose（LayoutTransform / L3）、TransposeLinear（FusedComposite / L3）、UnsortedSegmentSum（ScatterUpdate / L2）

### 逐算子用例数变化

| 算子 | 旧 | 新 | Δ |
|---|---|---|---|
| AddRmsNorm | 6 | 0 | -6 |
| ApplyRotaryPosEmb | 6 | 0 | -6 |
| ArgMax | 20 | 0 | -20 |
| AssignReqToTokenPool | 0 | 20 | +20 |
| BinnedGatherScatter | 0 | 20 | +20 |
| CausalConv1d | 0 | 20 | +20 |
| CrossEntropyLoss | 20 | 0 | -20 |
| Cummin | 20 | 0 | -20 |
| DecodeGroupedAttention | 0 | 20 | +20 |
| DiscreteMemoryAccess | 0 | 20 | +20 |
| Exp | 10 | 0 | -10 |
| ExtractSlice | 0 | 20 | +20 |
| Gather | 20 | 0 | -20 |
| GatherExp | 6 | 0 | -6 |
| GatherScatter | 0 | 20 | +20 |
| Gelu | 10 | 0 | -10 |
| GroupedMatmul | 20 | 0 | -20 |
| GroupedMatmulSwiGlu | 6 | 0 | -6 |
| InsertSlice | 0 | 20 | +20 |
| LinearGelu | 6 | 0 | -6 |
| MaskedGelu | 8 | 0 | -8 |
| MaskedScale | 20 | 0 | -20 |
| MaskedSoftmax | 8 | 0 | -8 |
| Maximum | 20 | 0 | -20 |
| MoeGatingTopKSoftmax | 5 | 0 | -5 |
| PaddedGatherScatter | 0 | 20 | +20 |
| RmsNorm | 9 | 0 | -9 |
| RmsNormLinear | 6 | 0 | -6 |
| Scatter | 10 | 0 | -10 |
| Softmax | 11 | 0 | -11 |
| SwiGlu | 6 | 0 | -6 |
| Transpose | 20 | 0 | -20 |
| TransposeLinear | 6 | 0 | -6 |
| UnsortedSegmentSum | 10 | 0 | -10 |
| VectorAdd | 0 | 20 | +20 |
| VectorCmp | 0 | 20 | +20 |
（其余 0 个算子用例数不变）

### 用例级变化

- 新增用例：220 个（超过 20，按算子折叠）
  - assignreqtotokenpool：20
  - binnedgatherscatter：20
  - causalconv1d：20
  - decodegroupedattention：20
  - discretememoryaccess：20
  - extractslice：20
  - gatherscatter：20
  - insertslice：20
  - paddedgatherscatter：20
  - vectoradd：20
  - vectorcmp：20
- 移除用例：289 个（超过 20，按算子折叠）
  - add_rms_norm：6
  - apply_rotary_pos_emb：6
  - arg_max：20
  - cross_entropy_loss：20
  - cummin：20
  - exp：10
  - gather：20
  - gather_exp：6
  - gelu：10
  - grouped_matmul：20
  - grouped_matmul_swiglu：6
  - linear_gelu：6
  - masked_gelu：8
  - masked_scale：20
  - masked_softmax：8
  - maximum：20
  - moe_gating_top_k_softmax：5
  - rms_norm：9
  - rms_norm_linear：6
  - scatter：10
  - softmax：11
  - swi_glu：6
  - transpose：20
  - transpose_linear：6
  - unsorted_segment_sum：10
- 同编号但内容变更：无
- 内容未变的用例：0 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| AddRmsNorm/LARGE | 1 | 0 | -1 |
| AddRmsNorm/MEDIUM | 1 | 0 | -1 |
| AddRmsNorm/SMALL | 3 | 0 | -3 |
| AddRmsNorm/XLARGE | 1 | 0 | -1 |
| ApplyRotaryPosEmb/LARGE | 1 | 0 | -1 |
| ApplyRotaryPosEmb/MEDIUM | 1 | 0 | -1 |
| ApplyRotaryPosEmb/SMALL | 3 | 0 | -3 |
| ApplyRotaryPosEmb/XLARGE | 1 | 0 | -1 |
| ArgMax/LARGE | 1 | 0 | -1 |
| ArgMax/MEDIUM | 2 | 0 | -2 |
| ArgMax/SMALL | 16 | 0 | -16 |
| ArgMax/XLARGE | 1 | 0 | -1 |
| AssignReqToTokenPool/SMALL | 0 | 20 | +20 |
| BinnedGatherScatter/LARGE | 0 | 6 | +6 |
| BinnedGatherScatter/MEDIUM | 0 | 2 | +2 |
| BinnedGatherScatter/XLARGE | 0 | 12 | +12 |
| CausalConv1d/MEDIUM | 0 | 10 | +10 |
| CausalConv1d/SMALL | 0 | 10 | +10 |
| CrossEntropyLoss/LARGE | 2 | 0 | -2 |
| CrossEntropyLoss/MEDIUM | 7 | 0 | -7 |
| CrossEntropyLoss/SMALL | 10 | 0 | -10 |
| CrossEntropyLoss/XLARGE | 1 | 0 | -1 |
| Cummin/LARGE | 1 | 0 | -1 |
| Cummin/MEDIUM | 2 | 0 | -2 |
| Cummin/SMALL | 16 | 0 | -16 |
| Cummin/XLARGE | 1 | 0 | -1 |
| DecodeGroupedAttention/MEDIUM | 0 | 10 | +10 |
| DecodeGroupedAttention/SMALL | 0 | 10 | +10 |
| DiscreteMemoryAccess/SMALL | 0 | 20 | +20 |
| Exp/LARGE | 1 | 0 | -1 |
| Exp/MEDIUM | 3 | 0 | -3 |
| Exp/SMALL | 5 | 0 | -5 |
| Exp/XLARGE | 1 | 0 | -1 |
| ExtractSlice/LARGE | 0 | 5 | +5 |
| ExtractSlice/MEDIUM | 0 | 13 | +13 |
| ExtractSlice/SMALL | 0 | 2 | +2 |
| Gather/LARGE | 1 | 0 | -1 |
| Gather/MEDIUM | 2 | 0 | -2 |
| Gather/SMALL | 16 | 0 | -16 |
| Gather/XLARGE | 1 | 0 | -1 |
| GatherExp/LARGE | 1 | 0 | -1 |
| GatherExp/MEDIUM | 1 | 0 | -1 |
| GatherExp/SMALL | 3 | 0 | -3 |
| GatherExp/XLARGE | 1 | 0 | -1 |
| GatherScatter/LARGE | 0 | 8 | +8 |
| GatherScatter/MEDIUM | 0 | 3 | +3 |
| GatherScatter/XLARGE | 0 | 9 | +9 |
| Gelu/LARGE | 1 | 0 | -1 |
| Gelu/MEDIUM | 3 | 0 | -3 |
| Gelu/SMALL | 5 | 0 | -5 |
| Gelu/XLARGE | 1 | 0 | -1 |
| GroupedMatmul/LARGE | 1 | 0 | -1 |
| GroupedMatmul/MEDIUM | 11 | 0 | -11 |
| GroupedMatmul/SMALL | 6 | 0 | -6 |
| GroupedMatmul/XLARGE | 2 | 0 | -2 |
| GroupedMatmulSwiGlu/LARGE | 1 | 0 | -1 |
| GroupedMatmulSwiGlu/MEDIUM | 1 | 0 | -1 |
| GroupedMatmulSwiGlu/SMALL | 3 | 0 | -3 |
| GroupedMatmulSwiGlu/XLARGE | 1 | 0 | -1 |
| InsertSlice/LARGE | 0 | 5 | +5 |
| InsertSlice/MEDIUM | 0 | 13 | +13 |
| InsertSlice/SMALL | 0 | 2 | +2 |
| LinearGelu/LARGE | 1 | 0 | -1 |
| LinearGelu/MEDIUM | 1 | 0 | -1 |
| LinearGelu/SMALL | 3 | 0 | -3 |
| LinearGelu/XLARGE | 1 | 0 | -1 |
| MaskedGelu/LARGE | 1 | 0 | -1 |
| MaskedGelu/MEDIUM | 1 | 0 | -1 |
| MaskedGelu/SMALL | 5 | 0 | -5 |
| MaskedGelu/XLARGE | 1 | 0 | -1 |
| MaskedScale/LARGE | 1 | 0 | -1 |
| MaskedScale/MEDIUM | 3 | 0 | -3 |
| MaskedScale/SMALL | 15 | 0 | -15 |
| MaskedScale/XLARGE | 1 | 0 | -1 |
| MaskedSoftmax/LARGE | 1 | 0 | -1 |
| MaskedSoftmax/MEDIUM | 1 | 0 | -1 |
| MaskedSoftmax/SMALL | 5 | 0 | -5 |
| MaskedSoftmax/XLARGE | 1 | 0 | -1 |
| Maximum/LARGE | 1 | 0 | -1 |
| Maximum/MEDIUM | 3 | 0 | -3 |
| Maximum/SMALL | 15 | 0 | -15 |
| Maximum/XLARGE | 1 | 0 | -1 |
| MoeGatingTopKSoftmax/LARGE | 1 | 0 | -1 |
| MoeGatingTopKSoftmax/MEDIUM | 1 | 0 | -1 |
| MoeGatingTopKSoftmax/SMALL | 2 | 0 | -2 |
| MoeGatingTopKSoftmax/XLARGE | 1 | 0 | -1 |
| PaddedGatherScatter/LARGE | 0 | 8 | +8 |
| PaddedGatherScatter/MEDIUM | 0 | 3 | +3 |
| PaddedGatherScatter/XLARGE | 0 | 9 | +9 |
| RmsNorm/LARGE | 1 | 0 | -1 |
| RmsNorm/MEDIUM | 2 | 0 | -2 |
| RmsNorm/SMALL | 5 | 0 | -5 |
| RmsNorm/XLARGE | 1 | 0 | -1 |
| RmsNormLinear/LARGE | 1 | 0 | -1 |
| RmsNormLinear/MEDIUM | 1 | 0 | -1 |
| RmsNormLinear/SMALL | 3 | 0 | -3 |
| RmsNormLinear/XLARGE | 1 | 0 | -1 |
| Scatter/LARGE | 1 | 0 | -1 |
| Scatter/MEDIUM | 2 | 0 | -2 |
| Scatter/SMALL | 6 | 0 | -6 |
| Scatter/XLARGE | 1 | 0 | -1 |
| Softmax/LARGE | 1 | 0 | -1 |
| Softmax/MEDIUM | 3 | 0 | -3 |
| Softmax/SMALL | 6 | 0 | -6 |
| Softmax/XLARGE | 1 | 0 | -1 |
| SwiGlu/LARGE | 1 | 0 | -1 |
| SwiGlu/MEDIUM | 1 | 0 | -1 |
| SwiGlu/SMALL | 3 | 0 | -3 |
| SwiGlu/XLARGE | 1 | 0 | -1 |
| Transpose/LARGE | 1 | 0 | -1 |
| Transpose/MEDIUM | 2 | 0 | -2 |
| Transpose/SMALL | 16 | 0 | -16 |
| Transpose/XLARGE | 1 | 0 | -1 |
| TransposeLinear/LARGE | 1 | 0 | -1 |
| TransposeLinear/MEDIUM | 1 | 0 | -1 |
| TransposeLinear/SMALL | 3 | 0 | -3 |
| TransposeLinear/XLARGE | 1 | 0 | -1 |
| UnsortedSegmentSum/LARGE | 1 | 0 | -1 |
| UnsortedSegmentSum/MEDIUM | 2 | 0 | -2 |
| UnsortedSegmentSum/SMALL | 6 | 0 | -6 |
| UnsortedSegmentSum/XLARGE | 1 | 0 | -1 |
| VectorAdd/LARGE | 0 | 2 | +2 |
| VectorAdd/MEDIUM | 0 | 9 | +9 |
| VectorAdd/SMALL | 0 | 9 | +9 |
| VectorCmp/MEDIUM | 0 | 8 | +8 |
| VectorCmp/SMALL | 0 | 12 | +12 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| AddRmsNorm/bfloat16 | 1 | 0 | -1 |
| AddRmsNorm/float16 | 2 | 0 | -2 |
| AddRmsNorm/float32 | 3 | 0 | -3 |
| ApplyRotaryPosEmb/bfloat16 | 2 | 0 | -2 |
| ApplyRotaryPosEmb/float16 | 3 | 0 | -3 |
| ApplyRotaryPosEmb/float32 | 1 | 0 | -1 |
| ArgMax/bfloat16 | 4 | 0 | -4 |
| ArgMax/float16 | 5 | 0 | -5 |
| ArgMax/float32 | 6 | 0 | -6 |
| ArgMax/int32 | 3 | 0 | -3 |
| ArgMax/int64 | 2 | 0 | -2 |
| AssignReqToTokenPool/int32 | 0 | 20 | +20 |
| BinnedGatherScatter/float16 | 0 | 20 | +20 |
| CausalConv1d/bfloat16 | 0 | 20 | +20 |
| CrossEntropyLoss/bfloat16 | 4 | 0 | -4 |
| CrossEntropyLoss/float16 | 5 | 0 | -5 |
| CrossEntropyLoss/float32 | 5 | 0 | -5 |
| CrossEntropyLoss/int32 | 3 | 0 | -3 |
| CrossEntropyLoss/int64 | 3 | 0 | -3 |
| Cummin/bfloat16 | 3 | 0 | -3 |
| Cummin/float16 | 4 | 0 | -4 |
| Cummin/float32 | 8 | 0 | -8 |
| Cummin/int32 | 2 | 0 | -2 |
| Cummin/int64 | 3 | 0 | -3 |
| DecodeGroupedAttention/bfloat16 | 0 | 20 | +20 |
| DiscreteMemoryAccess/float32 | 0 | 10 | +10 |
| DiscreteMemoryAccess/int32 | 0 | 10 | +10 |
| Exp/bfloat16 | 1 | 0 | -1 |
| Exp/float16 | 4 | 0 | -4 |
| Exp/float32 | 5 | 0 | -5 |
| ExtractSlice/float32 | 0 | 20 | +20 |
| Gather/bfloat16 | 2 | 0 | -2 |
| Gather/float16 | 4 | 0 | -4 |
| Gather/float32 | 7 | 0 | -7 |
| Gather/int32 | 3 | 0 | -3 |
| Gather/int64 | 2 | 0 | -2 |
| Gather/int8 | 2 | 0 | -2 |
| GatherExp/bfloat16 | 1 | 0 | -1 |
| GatherExp/float16 | 2 | 0 | -2 |
| GatherExp/float32 | 3 | 0 | -3 |
| GatherScatter/float16 | 0 | 20 | +20 |
| Gelu/bfloat16 | 1 | 0 | -1 |
| Gelu/float16 | 3 | 0 | -3 |
| Gelu/float32 | 6 | 0 | -6 |
| GroupedMatmul/bfloat16 | 6 | 0 | -6 |
| GroupedMatmul/float16 | 5 | 0 | -5 |
| GroupedMatmul/float32 | 9 | 0 | -9 |
| GroupedMatmulSwiGlu/bfloat16 | 1 | 0 | -1 |
| GroupedMatmulSwiGlu/float16 | 2 | 0 | -2 |
| GroupedMatmulSwiGlu/float32 | 3 | 0 | -3 |
| InsertSlice/float32 | 0 | 20 | +20 |
| LinearGelu/bfloat16 | 1 | 0 | -1 |
| LinearGelu/float16 | 2 | 0 | -2 |
| LinearGelu/float32 | 3 | 0 | -3 |
| MaskedGelu/bfloat16 | 1 | 0 | -1 |
| MaskedGelu/float16 | 2 | 0 | -2 |
| MaskedGelu/float32 | 3 | 0 | -3 |
| MaskedGelu/int8 | 1 | 0 | -1 |
| MaskedGelu/uint8 | 1 | 0 | -1 |
| MaskedScale/bfloat16 | 3 | 0 | -3 |
| MaskedScale/float16 | 5 | 0 | -5 |
| MaskedScale/float32 | 7 | 0 | -7 |
| MaskedScale/int8 | 2 | 0 | -2 |
| MaskedScale/uint8 | 3 | 0 | -3 |
| MaskedSoftmax/bfloat16 | 1 | 0 | -1 |
| MaskedSoftmax/float16 | 2 | 0 | -2 |
| MaskedSoftmax/float32 | 3 | 0 | -3 |
| MaskedSoftmax/int8 | 1 | 0 | -1 |
| MaskedSoftmax/uint8 | 1 | 0 | -1 |
| Maximum/bfloat16 | 3 | 0 | -3 |
| Maximum/float16 | 3 | 0 | -3 |
| Maximum/float32 | 5 | 0 | -5 |
| Maximum/int32 | 2 | 0 | -2 |
| Maximum/int64 | 4 | 0 | -4 |
| Maximum/int8 | 3 | 0 | -3 |
| MoeGatingTopKSoftmax/bfloat16 | 2 | 0 | -2 |
| MoeGatingTopKSoftmax/float32 | 3 | 0 | -3 |
| PaddedGatherScatter/float16 | 0 | 20 | +20 |
| RmsNorm/bfloat16 | 1 | 0 | -1 |
| RmsNorm/float16 | 3 | 0 | -3 |
| RmsNorm/float32 | 5 | 0 | -5 |
| RmsNormLinear/bfloat16 | 1 | 0 | -1 |
| RmsNormLinear/float16 | 2 | 0 | -2 |
| RmsNormLinear/float32 | 3 | 0 | -3 |
| Scatter/bfloat16 | 1 | 0 | -1 |
| Scatter/float16 | 3 | 0 | -3 |
| Scatter/float32 | 4 | 0 | -4 |
| Scatter/int32 | 1 | 0 | -1 |
| Scatter/int64 | 1 | 0 | -1 |
| Softmax/bfloat16 | 1 | 0 | -1 |
| Softmax/float16 | 3 | 0 | -3 |
| Softmax/float32 | 7 | 0 | -7 |
| SwiGlu/bfloat16 | 1 | 0 | -1 |
| SwiGlu/float16 | 3 | 0 | -3 |
| SwiGlu/float32 | 2 | 0 | -2 |
| Transpose/bfloat16 | 3 | 0 | -3 |
| Transpose/float16 | 2 | 0 | -2 |
| Transpose/float32 | 6 | 0 | -6 |
| Transpose/int16 | 2 | 0 | -2 |
| Transpose/int32 | 2 | 0 | -2 |
| Transpose/int64 | 3 | 0 | -3 |
| Transpose/int8 | 2 | 0 | -2 |
| TransposeLinear/bfloat16 | 1 | 0 | -1 |
| TransposeLinear/float16 | 2 | 0 | -2 |
| TransposeLinear/float32 | 3 | 0 | -3 |
| UnsortedSegmentSum/bfloat16 | 1 | 0 | -1 |
| UnsortedSegmentSum/float16 | 2 | 0 | -2 |
| UnsortedSegmentSum/float32 | 5 | 0 | -5 |
| UnsortedSegmentSum/int32 | 1 | 0 | -1 |
| UnsortedSegmentSum/int64 | 1 | 0 | -1 |
| VectorAdd/float32 | 0 | 7 | +7 |
| VectorAdd/int32 | 0 | 7 | +7 |
| VectorAdd/int64 | 0 | 6 | +6 |
| VectorCmp/float32 | 0 | 20 | +20 |

### 覆盖原子得失

- 原子覆盖数：71 → 23
- 新覆盖：无
- 不再覆盖：FA-ATTR-LIST、FA-AXIS-FIRST、FA-AXIS-INNER、FA-AXIS-LAST、FA-CONCUR-ACCUM、FA-CTRL-STATIC-UNROLL、FA-DT-INT16、FA-DT-INT8、FA-DT-UINT8、FA-FUSE-2STAGE、FA-FUSE-CROSS-CATEGORY、FA-FUSE-ELEM-INTRA、FA-FUSE-INTERM-ELIDE、FA-FUSE-RED-INTRA、FA-IDX-GATHER、FA-IDX-PERMUTE、FA-IDX-SCATTER、FA-IDX-WHERE、FA-LINALG-BIAS、FA-LINALG-DOT、FA-LOAD-STRIDED、FA-MATH-ADD、FA-MATH-EXP、FA-MATH-GELU、FA-MATH-MAXMIN、FA-MATH-MUL、FA-MATH-SIGMOID、FA-MATH-SINCOS、FA-MATH-SQRT、FA-MULTI-INPUT、FA-MULTI-OUTPUT、FA-NORM-RMS、FA-NORM-SOFTMAX、FA-NUM-MIXED、FA-NUM-ZEROS、FA-PAT-MASK-TAIL、FA-PAT-MULTIPASS-REDUCE、FA-PAT-SCAN-BLOCKED、FA-PAT-SPLIT-K、FA-PAT-TILE-GEMM、FA-PAT-VEC-BROADCAST、FA-PROG-GRID4D、FA-REDUCE-ARG、FA-REDUCE-MAX、FA-REDUCE-MIN、FA-REDUCE-SUM、FA-STABLE-LOGSUMEXP、FA-STABLE-MAXSUB

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.5-cov20-frozen：generation-log.yaml 共 9 条记录（日志起点 2026-10-14T07:58:58.348781+00:00，此前构建无日志）
- v0.6-ascend-ops：无 generation-log.yaml（该版构建早于日志机制上线）


## v0.6-ascend-ops → v0.7-ascend-contract

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-22T17:05:59.140708+00:00 | 2026-10-23T03:53:02.363824+00:00 |
| 清单哈希 | `e6ef84915692…` | `3b46ed7755b9…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | v1.1-fusion | 1.1-fusion |
| 算子数 | 11 | 11 |
| 用例数 | 220 | 220 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：17 个
  - binnedgatherscatter-float16-32
  - causalconv1d-bfloat16-20
  - causalconv1d-bfloat16-21
  - discretememoryaccess-float32-20
  - gatherscatter-float16-20
  - gatherscatter-float16-21
  - gatherscatter-float16-22
  - gatherscatter-float16-23
  - gatherscatter-float16-48
  - paddedgatherscatter-float16-20
  - paddedgatherscatter-float16-21
  - paddedgatherscatter-float16-22
  - paddedgatherscatter-float16-23
  - paddedgatherscatter-float16-48
  - vectoradd-float32-21
  - vectorcmp-float32-20
  - vectorcmp-float32-21
- 移除用例：17 个
  - binnedgatherscatter-float16-19
  - causalconv1d-bfloat16-18
  - causalconv1d-bfloat16-19
  - discretememoryaccess-int32-19
  - gatherscatter-float16-13
  - gatherscatter-float16-14
  - gatherscatter-float16-15
  - gatherscatter-float16-16
  - gatherscatter-float16-17
  - paddedgatherscatter-float16-13
  - paddedgatherscatter-float16-14
  - paddedgatherscatter-float16-15
  - paddedgatherscatter-float16-16
  - paddedgatherscatter-float16-17
  - vectoradd-int32-19
  - vectorcmp-float32-16
  - vectorcmp-float32-17
- **同编号但内容变更** 85 个（编号是弱键：不编码形状/数值，跨版本同编号不等于同题；字段变更频次：attrs×49、feature_atoms×63、shape×85）
  - 超过 20 个，按算子折叠：AssignReqToTokenPool×20、BinnedGatherScatter×19、CausalConv1d×8、ExtractSlice×4、GatherScatter×15、InsertSlice×4、PaddedGatherScatter×15
- 内容未变的用例：118 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| BinnedGatherScatter/LARGE | 6 | 7 | +1 |
| BinnedGatherScatter/MEDIUM | 2 | 12 | +10 |
| BinnedGatherScatter/XLARGE | 12 | 1 | -11 |
| ExtractSlice/MEDIUM | 13 | 14 | +1 |
| ExtractSlice/SMALL | 2 | 1 | -1 |
| GatherScatter/LARGE | 8 | 1 | -7 |
| GatherScatter/MEDIUM | 3 | 18 | +15 |
| GatherScatter/XLARGE | 9 | 1 | -8 |
| InsertSlice/MEDIUM | 13 | 14 | +1 |
| InsertSlice/SMALL | 2 | 1 | -1 |
| PaddedGatherScatter/LARGE | 8 | 1 | -7 |
| PaddedGatherScatter/MEDIUM | 3 | 18 | +15 |
| PaddedGatherScatter/XLARGE | 9 | 1 | -8 |
| VectorAdd/LARGE | 2 | 1 | -1 |
| VectorAdd/XLARGE | 0 | 1 | +1 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| DiscreteMemoryAccess/float32 | 10 | 11 | +1 |
| DiscreteMemoryAccess/int32 | 10 | 9 | -1 |
| VectorAdd/float32 | 7 | 8 | +1 |
| VectorAdd/int32 | 7 | 6 | -1 |

### 覆盖原子得失

- 原子覆盖数：23 → 23
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.6-ascend-ops：无 generation-log.yaml（该版构建早于日志机制上线）
- v0.7-ascend-contract：generation-log.yaml 共 11 条记录（日志起点 2026-10-23T04:05:58.765277+00:00，此前构建无日志）


## v0.7-ascend-contract → v0.8-official-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-23T03:53:02.363824+00:00 | 2026-10-23T10:13:44.636556+00:00 |
| 清单哈希 | `3b46ed7755b9…` | `249394cfdc65…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 11 | 68 |
| 用例数 | 220 | 1360 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子 68 个：Abs（Official source candidate / Elementwise）、Add（Official source candidate / Elementwise）、ArgMax（Official source candidate / SortSelect）、ArgMin（Official source candidate / SortSelect）、BitwiseAnd（Official source candidate / Elementwise）、BitwiseNot（Official source candidate / Elementwise）、BitwiseOr（Official source candidate / Elementwise）、BitwiseXor（Official source candidate / Elementwise）、Ceil（Official source candidate / Elementwise）、Clamp（Official source candidate / Elementwise）、Concat（Official source candidate / LayoutTransform）、Cos（Official source candidate / Elementwise）、Cosh（Official source candidate / Elementwise）、CumProd（Official source candidate / Reduction）、CumSum（Official source candidate / Reduction）、Div（Official source candidate / Elementwise）、Equal（Official source candidate / MaskPredicate）、Exp（Official source candidate / Elementwise）、Exp2（Official source candidate / Elementwise）、Expm1（Official source candidate / Elementwise）、Flip（Official source candidate / LayoutTransform）、Floor（Official source candidate / Elementwise）、FloorDiv（Official source candidate / Elementwise）、Gather（Official source candidate / IndexGather）、Gelu（Official source candidate / Elementwise）、Greater（Official source candidate / MaskPredicate）、GreaterEqual（Official source candidate / MaskPredicate）、LayerNorm（Official source candidate / Normalization）、Less（Official source candidate / MaskPredicate）、LessEqual（Official source candidate / MaskPredicate）、Log（Official source candidate / Elementwise）、Log10（Official source candidate / Elementwise）、Log1p（Official source candidate / Elementwise）、Log2（Official source candidate / Elementwise）、LogicalAnd（Official source candidate / MaskPredicate）、LogicalOr（Official source candidate / MaskPredicate）、Matmul（Official source candidate / Contraction）、Maximum（Official source candidate / Elementwise）、Minimum（Official source candidate / Elementwise）、Mish（Official source candidate / Elementwise）、Mod（Official source candidate / Elementwise）、Mul（Official source candidate / Elementwise）、Neg（Official source candidate / Elementwise）、NotEqual（Official source candidate / MaskPredicate）、Pow（Official source candidate / Elementwise）、ReduceMax（Official source candidate / Reduction）、ReduceMean（Official source candidate / Reduction）、ReduceMin（Official source candidate / Reduction）、ReduceSum（Official source candidate / Reduction）、Relu（Official source candidate / Elementwise）、RmsNorm（Official source candidate / Normalization）、Rsqrt（Official source candidate / Elementwise）、Sigmoid（Official source candidate / Elementwise）、Signbit（Official source candidate / MaskPredicate）、Sin（Official source candidate / Elementwise）、Sinh（Official source candidate / Elementwise）、Softmax（Official source candidate / Normalization）、SortValues（Official source candidate / SortSelect）、Sqrt（Official source candidate / Elementwise）、Sub（Official source candidate / Elementwise）、SwiGlu（Official source candidate / FusedComposite）、Tan（Official source candidate / Elementwise）、Tanh（Official source candidate / Elementwise）、TopKValues（Official source candidate / SortSelect）、Transpose（Official source candidate / LayoutTransform）、Trunc（Official source candidate / Elementwise）、Where（Official source candidate / MaskPredicate）、XorSum（Official source candidate / Reduction）
- 移除算子 11 个：AssignReqToTokenPool（ScatterUpdate / L3）、BinnedGatherScatter（IndexGather / L3）、CausalConv1d（FusedComposite / L3）、DecodeGroupedAttention（Contraction / L3）、DiscreteMemoryAccess（IndexGather / L2）、ExtractSlice（LayoutTransform / L2）、GatherScatter（IndexGather / L3）、InsertSlice（LayoutTransform / L2）、PaddedGatherScatter（IndexGather / L3）、VectorAdd（Elementwise / L1）、VectorCmp（Normalization / L2）

### 逐算子用例数变化

| 算子 | 旧 | 新 | Δ |
|---|---|---|---|
| Abs | 0 | 20 | +20 |
| Add | 0 | 20 | +20 |
| ArgMax | 0 | 20 | +20 |
| ArgMin | 0 | 20 | +20 |
| AssignReqToTokenPool | 20 | 0 | -20 |
| BinnedGatherScatter | 20 | 0 | -20 |
| BitwiseAnd | 0 | 20 | +20 |
| BitwiseNot | 0 | 20 | +20 |
| BitwiseOr | 0 | 20 | +20 |
| BitwiseXor | 0 | 20 | +20 |
| CausalConv1d | 20 | 0 | -20 |
| Ceil | 0 | 20 | +20 |
| Clamp | 0 | 20 | +20 |
| Concat | 0 | 20 | +20 |
| Cos | 0 | 20 | +20 |
| Cosh | 0 | 20 | +20 |
| CumProd | 0 | 20 | +20 |
| CumSum | 0 | 20 | +20 |
| DecodeGroupedAttention | 20 | 0 | -20 |
| DiscreteMemoryAccess | 20 | 0 | -20 |
| Div | 0 | 20 | +20 |
| Equal | 0 | 20 | +20 |
| Exp | 0 | 20 | +20 |
| Exp2 | 0 | 20 | +20 |
| Expm1 | 0 | 20 | +20 |
| ExtractSlice | 20 | 0 | -20 |
| Flip | 0 | 20 | +20 |
| Floor | 0 | 20 | +20 |
| FloorDiv | 0 | 20 | +20 |
| Gather | 0 | 20 | +20 |
| GatherScatter | 20 | 0 | -20 |
| Gelu | 0 | 20 | +20 |
| Greater | 0 | 20 | +20 |
| GreaterEqual | 0 | 20 | +20 |
| InsertSlice | 20 | 0 | -20 |
| LayerNorm | 0 | 20 | +20 |
| Less | 0 | 20 | +20 |
| LessEqual | 0 | 20 | +20 |
| Log | 0 | 20 | +20 |
| Log10 | 0 | 20 | +20 |
| Log1p | 0 | 20 | +20 |
| Log2 | 0 | 20 | +20 |
| LogicalAnd | 0 | 20 | +20 |
| LogicalOr | 0 | 20 | +20 |
| Matmul | 0 | 20 | +20 |
| Maximum | 0 | 20 | +20 |
| Minimum | 0 | 20 | +20 |
| Mish | 0 | 20 | +20 |
| Mod | 0 | 20 | +20 |
| Mul | 0 | 20 | +20 |
| Neg | 0 | 20 | +20 |
| NotEqual | 0 | 20 | +20 |
| PaddedGatherScatter | 20 | 0 | -20 |
| Pow | 0 | 20 | +20 |
| ReduceMax | 0 | 20 | +20 |
| ReduceMean | 0 | 20 | +20 |
| ReduceMin | 0 | 20 | +20 |
| ReduceSum | 0 | 20 | +20 |
| Relu | 0 | 20 | +20 |
| RmsNorm | 0 | 20 | +20 |
| Rsqrt | 0 | 20 | +20 |
| Sigmoid | 0 | 20 | +20 |
| Signbit | 0 | 20 | +20 |
| Sin | 0 | 20 | +20 |
| Sinh | 0 | 20 | +20 |
| Softmax | 0 | 20 | +20 |
| SortValues | 0 | 20 | +20 |
| Sqrt | 0 | 20 | +20 |
| Sub | 0 | 20 | +20 |
| SwiGlu | 0 | 20 | +20 |
| Tan | 0 | 20 | +20 |
| Tanh | 0 | 20 | +20 |
| TopKValues | 0 | 20 | +20 |
| Transpose | 0 | 20 | +20 |
| Trunc | 0 | 20 | +20 |
| VectorAdd | 20 | 0 | -20 |
| VectorCmp | 20 | 0 | -20 |
| Where | 0 | 20 | +20 |
| XorSum | 0 | 20 | +20 |
（其余 0 个算子用例数不变）

### 用例级变化

- 新增用例：1360 个（超过 20，按算子折叠）
  - official：1360
- 移除用例：220 个（超过 20，按算子折叠）
  - assignreqtotokenpool：20
  - binnedgatherscatter：20
  - causalconv1d：20
  - decodegroupedattention：20
  - discretememoryaccess：20
  - extractslice：20
  - gatherscatter：20
  - insertslice：20
  - paddedgatherscatter：20
  - vectoradd：20
  - vectorcmp：20
- 同编号但内容变更：无
- 内容未变的用例：0 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| Abs/MEDIUM | 0 | 2 | +2 |
| Abs/SMALL | 0 | 18 | +18 |
| Add/MEDIUM | 0 | 2 | +2 |
| Add/SMALL | 0 | 18 | +18 |
| ArgMax/MEDIUM | 0 | 2 | +2 |
| ArgMax/SMALL | 0 | 18 | +18 |
| ArgMin/MEDIUM | 0 | 2 | +2 |
| ArgMin/SMALL | 0 | 18 | +18 |
| AssignReqToTokenPool/SMALL | 20 | 0 | -20 |
| BinnedGatherScatter/LARGE | 7 | 0 | -7 |
| BinnedGatherScatter/MEDIUM | 12 | 0 | -12 |
| BinnedGatherScatter/XLARGE | 1 | 0 | -1 |
| BitwiseAnd/MEDIUM | 0 | 5 | +5 |
| BitwiseAnd/SMALL | 0 | 15 | +15 |
| BitwiseNot/MEDIUM | 0 | 5 | +5 |
| BitwiseNot/SMALL | 0 | 15 | +15 |
| BitwiseOr/MEDIUM | 0 | 5 | +5 |
| BitwiseOr/SMALL | 0 | 15 | +15 |
| BitwiseXor/MEDIUM | 0 | 5 | +5 |
| BitwiseXor/SMALL | 0 | 15 | +15 |
| CausalConv1d/MEDIUM | 10 | 0 | -10 |
| CausalConv1d/SMALL | 10 | 0 | -10 |
| Ceil/MEDIUM | 0 | 2 | +2 |
| Ceil/SMALL | 0 | 18 | +18 |
| Clamp/MEDIUM | 0 | 2 | +2 |
| Clamp/SMALL | 0 | 18 | +18 |
| Concat/MEDIUM | 0 | 2 | +2 |
| Concat/SMALL | 0 | 18 | +18 |
| Cos/MEDIUM | 0 | 2 | +2 |
| Cos/SMALL | 0 | 18 | +18 |
| Cosh/MEDIUM | 0 | 2 | +2 |
| Cosh/SMALL | 0 | 18 | +18 |
| CumProd/MEDIUM | 0 | 2 | +2 |
| CumProd/SMALL | 0 | 18 | +18 |
| CumSum/MEDIUM | 0 | 2 | +2 |
| CumSum/SMALL | 0 | 18 | +18 |
| DecodeGroupedAttention/MEDIUM | 10 | 0 | -10 |
| DecodeGroupedAttention/SMALL | 10 | 0 | -10 |
| DiscreteMemoryAccess/SMALL | 20 | 0 | -20 |
| Div/MEDIUM | 0 | 2 | +2 |
| Div/SMALL | 0 | 18 | +18 |
| Equal/MEDIUM | 0 | 2 | +2 |
| Equal/SMALL | 0 | 18 | +18 |
| Exp/MEDIUM | 0 | 2 | +2 |
| Exp/SMALL | 0 | 18 | +18 |
| Exp2/MEDIUM | 0 | 2 | +2 |
| Exp2/SMALL | 0 | 18 | +18 |
| Expm1/MEDIUM | 0 | 2 | +2 |
| Expm1/SMALL | 0 | 18 | +18 |
| ExtractSlice/LARGE | 5 | 0 | -5 |
| ExtractSlice/MEDIUM | 14 | 0 | -14 |
| ExtractSlice/SMALL | 1 | 0 | -1 |
| Flip/MEDIUM | 0 | 2 | +2 |
| Flip/SMALL | 0 | 18 | +18 |
| Floor/MEDIUM | 0 | 2 | +2 |
| Floor/SMALL | 0 | 18 | +18 |
| FloorDiv/MEDIUM | 0 | 5 | +5 |
| FloorDiv/SMALL | 0 | 15 | +15 |
| Gather/MEDIUM | 0 | 2 | +2 |
| Gather/SMALL | 0 | 18 | +18 |
| GatherScatter/LARGE | 1 | 0 | -1 |
| GatherScatter/MEDIUM | 18 | 0 | -18 |
| GatherScatter/XLARGE | 1 | 0 | -1 |
| Gelu/MEDIUM | 0 | 2 | +2 |
| Gelu/SMALL | 0 | 18 | +18 |
| Greater/MEDIUM | 0 | 2 | +2 |
| Greater/SMALL | 0 | 18 | +18 |
| GreaterEqual/MEDIUM | 0 | 2 | +2 |
| GreaterEqual/SMALL | 0 | 18 | +18 |
| InsertSlice/LARGE | 5 | 0 | -5 |
| InsertSlice/MEDIUM | 14 | 0 | -14 |
| InsertSlice/SMALL | 1 | 0 | -1 |
| LayerNorm/MEDIUM | 0 | 2 | +2 |
| LayerNorm/SMALL | 0 | 18 | +18 |
| Less/MEDIUM | 0 | 2 | +2 |
| Less/SMALL | 0 | 18 | +18 |
| LessEqual/MEDIUM | 0 | 2 | +2 |
| LessEqual/SMALL | 0 | 18 | +18 |
| Log/MEDIUM | 0 | 2 | +2 |
| Log/SMALL | 0 | 18 | +18 |
| Log10/MEDIUM | 0 | 2 | +2 |
| Log10/SMALL | 0 | 18 | +18 |
| Log1p/MEDIUM | 0 | 2 | +2 |
| Log1p/SMALL | 0 | 18 | +18 |
| Log2/MEDIUM | 0 | 2 | +2 |
| Log2/SMALL | 0 | 18 | +18 |
| LogicalAnd/MEDIUM | 0 | 2 | +2 |
| LogicalAnd/SMALL | 0 | 18 | +18 |
| LogicalOr/MEDIUM | 0 | 2 | +2 |
| LogicalOr/SMALL | 0 | 18 | +18 |
| Matmul/MEDIUM | 0 | 5 | +5 |
| Matmul/SMALL | 0 | 15 | +15 |
| Maximum/MEDIUM | 0 | 2 | +2 |
| Maximum/SMALL | 0 | 18 | +18 |
| Minimum/MEDIUM | 0 | 2 | +2 |
| Minimum/SMALL | 0 | 18 | +18 |
| Mish/MEDIUM | 0 | 2 | +2 |
| Mish/SMALL | 0 | 18 | +18 |
| Mod/MEDIUM | 0 | 2 | +2 |
| Mod/SMALL | 0 | 18 | +18 |
| Mul/MEDIUM | 0 | 2 | +2 |
| Mul/SMALL | 0 | 18 | +18 |
| Neg/MEDIUM | 0 | 2 | +2 |
| Neg/SMALL | 0 | 18 | +18 |
| NotEqual/MEDIUM | 0 | 2 | +2 |
| NotEqual/SMALL | 0 | 18 | +18 |
| PaddedGatherScatter/LARGE | 1 | 0 | -1 |
| PaddedGatherScatter/MEDIUM | 18 | 0 | -18 |
| PaddedGatherScatter/XLARGE | 1 | 0 | -1 |
| Pow/MEDIUM | 0 | 2 | +2 |
| Pow/SMALL | 0 | 18 | +18 |
| ReduceMax/MEDIUM | 0 | 2 | +2 |
| ReduceMax/SMALL | 0 | 18 | +18 |
| ReduceMean/MEDIUM | 0 | 2 | +2 |
| ReduceMean/SMALL | 0 | 18 | +18 |
| ReduceMin/MEDIUM | 0 | 2 | +2 |
| ReduceMin/SMALL | 0 | 18 | +18 |
| ReduceSum/MEDIUM | 0 | 2 | +2 |
| ReduceSum/SMALL | 0 | 18 | +18 |
| Relu/MEDIUM | 0 | 2 | +2 |
| Relu/SMALL | 0 | 18 | +18 |
| RmsNorm/MEDIUM | 0 | 2 | +2 |
| RmsNorm/SMALL | 0 | 18 | +18 |
| Rsqrt/MEDIUM | 0 | 2 | +2 |
| Rsqrt/SMALL | 0 | 18 | +18 |
| Sigmoid/MEDIUM | 0 | 2 | +2 |
| Sigmoid/SMALL | 0 | 18 | +18 |
| Signbit/MEDIUM | 0 | 2 | +2 |
| Signbit/SMALL | 0 | 18 | +18 |
| Sin/MEDIUM | 0 | 2 | +2 |
| Sin/SMALL | 0 | 18 | +18 |
| Sinh/MEDIUM | 0 | 2 | +2 |
| Sinh/SMALL | 0 | 18 | +18 |
| Softmax/MEDIUM | 0 | 2 | +2 |
| Softmax/SMALL | 0 | 18 | +18 |
| SortValues/MEDIUM | 0 | 2 | +2 |
| SortValues/SMALL | 0 | 18 | +18 |
| Sqrt/MEDIUM | 0 | 2 | +2 |
| Sqrt/SMALL | 0 | 18 | +18 |
| Sub/MEDIUM | 0 | 2 | +2 |
| Sub/SMALL | 0 | 18 | +18 |
| SwiGlu/MEDIUM | 0 | 2 | +2 |
| SwiGlu/SMALL | 0 | 18 | +18 |
| Tan/MEDIUM | 0 | 2 | +2 |
| Tan/SMALL | 0 | 18 | +18 |
| Tanh/MEDIUM | 0 | 2 | +2 |
| Tanh/SMALL | 0 | 18 | +18 |
| TopKValues/MEDIUM | 0 | 2 | +2 |
| TopKValues/SMALL | 0 | 18 | +18 |
| Transpose/MEDIUM | 0 | 2 | +2 |
| Transpose/SMALL | 0 | 18 | +18 |
| Trunc/MEDIUM | 0 | 2 | +2 |
| Trunc/SMALL | 0 | 18 | +18 |
| VectorAdd/LARGE | 1 | 0 | -1 |
| VectorAdd/MEDIUM | 9 | 0 | -9 |
| VectorAdd/SMALL | 9 | 0 | -9 |
| VectorAdd/XLARGE | 1 | 0 | -1 |
| VectorCmp/MEDIUM | 8 | 0 | -8 |
| VectorCmp/SMALL | 12 | 0 | -12 |
| Where/MEDIUM | 0 | 2 | +2 |
| Where/SMALL | 0 | 18 | +18 |
| XorSum/MEDIUM | 0 | 5 | +5 |
| XorSum/SMALL | 0 | 15 | +15 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| Abs/float16 | 0 | 8 | +8 |
| Abs/float32 | 0 | 12 | +12 |
| Add/float16 | 0 | 8 | +8 |
| Add/float32 | 0 | 12 | +12 |
| ArgMax/float16 | 0 | 8 | +8 |
| ArgMax/float32 | 0 | 12 | +12 |
| ArgMin/float16 | 0 | 8 | +8 |
| ArgMin/float32 | 0 | 12 | +12 |
| AssignReqToTokenPool/int32 | 20 | 0 | -20 |
| BinnedGatherScatter/float16 | 20 | 0 | -20 |
| BitwiseAnd/int32 | 0 | 20 | +20 |
| BitwiseNot/int32 | 0 | 20 | +20 |
| BitwiseOr/int32 | 0 | 20 | +20 |
| BitwiseXor/int32 | 0 | 20 | +20 |
| CausalConv1d/bfloat16 | 20 | 0 | -20 |
| Ceil/float16 | 0 | 8 | +8 |
| Ceil/float32 | 0 | 12 | +12 |
| Clamp/float16 | 0 | 8 | +8 |
| Clamp/float32 | 0 | 12 | +12 |
| Concat/float16 | 0 | 8 | +8 |
| Concat/float32 | 0 | 12 | +12 |
| Cos/float16 | 0 | 8 | +8 |
| Cos/float32 | 0 | 12 | +12 |
| Cosh/float16 | 0 | 8 | +8 |
| Cosh/float32 | 0 | 12 | +12 |
| CumProd/float16 | 0 | 8 | +8 |
| CumProd/float32 | 0 | 12 | +12 |
| CumSum/float16 | 0 | 8 | +8 |
| CumSum/float32 | 0 | 12 | +12 |
| DecodeGroupedAttention/bfloat16 | 20 | 0 | -20 |
| DiscreteMemoryAccess/float32 | 11 | 0 | -11 |
| DiscreteMemoryAccess/int32 | 9 | 0 | -9 |
| Div/float16 | 0 | 8 | +8 |
| Div/float32 | 0 | 12 | +12 |
| Equal/float16 | 0 | 8 | +8 |
| Equal/float32 | 0 | 12 | +12 |
| Exp/float16 | 0 | 8 | +8 |
| Exp/float32 | 0 | 12 | +12 |
| Exp2/float16 | 0 | 8 | +8 |
| Exp2/float32 | 0 | 12 | +12 |
| Expm1/float16 | 0 | 8 | +8 |
| Expm1/float32 | 0 | 12 | +12 |
| ExtractSlice/float32 | 20 | 0 | -20 |
| Flip/float16 | 0 | 8 | +8 |
| Flip/float32 | 0 | 12 | +12 |
| Floor/float16 | 0 | 8 | +8 |
| Floor/float32 | 0 | 12 | +12 |
| FloorDiv/int32 | 0 | 20 | +20 |
| Gather/float16 | 0 | 8 | +8 |
| Gather/float32 | 0 | 12 | +12 |
| GatherScatter/float16 | 20 | 0 | -20 |
| Gelu/float16 | 0 | 8 | +8 |
| Gelu/float32 | 0 | 12 | +12 |
| Greater/float16 | 0 | 8 | +8 |
| Greater/float32 | 0 | 12 | +12 |
| GreaterEqual/float16 | 0 | 8 | +8 |
| GreaterEqual/float32 | 0 | 12 | +12 |
| InsertSlice/float32 | 20 | 0 | -20 |
| LayerNorm/float16 | 0 | 8 | +8 |
| LayerNorm/float32 | 0 | 12 | +12 |
| Less/float16 | 0 | 8 | +8 |
| Less/float32 | 0 | 12 | +12 |
| LessEqual/float16 | 0 | 8 | +8 |
| LessEqual/float32 | 0 | 12 | +12 |
| Log/float16 | 0 | 8 | +8 |
| Log/float32 | 0 | 12 | +12 |
| Log10/float16 | 0 | 8 | +8 |
| Log10/float32 | 0 | 12 | +12 |
| Log1p/float16 | 0 | 8 | +8 |
| Log1p/float32 | 0 | 12 | +12 |
| Log2/float16 | 0 | 8 | +8 |
| Log2/float32 | 0 | 12 | +12 |
| LogicalAnd/float16 | 0 | 8 | +8 |
| LogicalAnd/float32 | 0 | 12 | +12 |
| LogicalOr/float16 | 0 | 8 | +8 |
| LogicalOr/float32 | 0 | 12 | +12 |
| Matmul/float16 | 0 | 20 | +20 |
| Maximum/float16 | 0 | 8 | +8 |
| Maximum/float32 | 0 | 12 | +12 |
| Minimum/float16 | 0 | 8 | +8 |
| Minimum/float32 | 0 | 12 | +12 |
| Mish/float16 | 0 | 8 | +8 |
| Mish/float32 | 0 | 12 | +12 |
| Mod/float16 | 0 | 8 | +8 |
| Mod/float32 | 0 | 12 | +12 |
| Mul/float16 | 0 | 8 | +8 |
| Mul/float32 | 0 | 12 | +12 |
| Neg/float16 | 0 | 8 | +8 |
| Neg/float32 | 0 | 12 | +12 |
| NotEqual/float16 | 0 | 8 | +8 |
| NotEqual/float32 | 0 | 12 | +12 |
| PaddedGatherScatter/float16 | 20 | 0 | -20 |
| Pow/float16 | 0 | 8 | +8 |
| Pow/float32 | 0 | 12 | +12 |
| ReduceMax/float16 | 0 | 8 | +8 |
| ReduceMax/float32 | 0 | 12 | +12 |
| ReduceMean/float16 | 0 | 8 | +8 |
| ReduceMean/float32 | 0 | 12 | +12 |
| ReduceMin/float16 | 0 | 8 | +8 |
| ReduceMin/float32 | 0 | 12 | +12 |
| ReduceSum/float16 | 0 | 8 | +8 |
| ReduceSum/float32 | 0 | 12 | +12 |
| Relu/float16 | 0 | 8 | +8 |
| Relu/float32 | 0 | 12 | +12 |
| RmsNorm/float16 | 0 | 8 | +8 |
| RmsNorm/float32 | 0 | 12 | +12 |
| Rsqrt/float16 | 0 | 8 | +8 |
| Rsqrt/float32 | 0 | 12 | +12 |
| Sigmoid/float16 | 0 | 8 | +8 |
| Sigmoid/float32 | 0 | 12 | +12 |
| Signbit/float16 | 0 | 8 | +8 |
| Signbit/float32 | 0 | 12 | +12 |
| Sin/float16 | 0 | 8 | +8 |
| Sin/float32 | 0 | 12 | +12 |
| Sinh/float16 | 0 | 8 | +8 |
| Sinh/float32 | 0 | 12 | +12 |
| Softmax/float16 | 0 | 8 | +8 |
| Softmax/float32 | 0 | 12 | +12 |
| SortValues/float16 | 0 | 8 | +8 |
| SortValues/float32 | 0 | 12 | +12 |
| Sqrt/float16 | 0 | 8 | +8 |
| Sqrt/float32 | 0 | 12 | +12 |
| Sub/float16 | 0 | 8 | +8 |
| Sub/float32 | 0 | 12 | +12 |
| SwiGlu/float16 | 0 | 8 | +8 |
| SwiGlu/float32 | 0 | 12 | +12 |
| Tan/float16 | 0 | 8 | +8 |
| Tan/float32 | 0 | 12 | +12 |
| Tanh/float16 | 0 | 8 | +8 |
| Tanh/float32 | 0 | 12 | +12 |
| TopKValues/float16 | 0 | 8 | +8 |
| TopKValues/float32 | 0 | 12 | +12 |
| Transpose/float16 | 0 | 8 | +8 |
| Transpose/float32 | 0 | 12 | +12 |
| Trunc/float16 | 0 | 8 | +8 |
| Trunc/float32 | 0 | 12 | +12 |
| VectorAdd/float32 | 8 | 0 | -8 |
| VectorAdd/int32 | 6 | 0 | -6 |
| VectorAdd/int64 | 6 | 0 | -6 |
| VectorCmp/float32 | 20 | 0 | -20 |
| Where/float16 | 0 | 8 | +8 |
| Where/float32 | 0 | 12 | +12 |
| XorSum/int32 | 0 | 20 | +20 |

### 覆盖原子得失

- 原子覆盖数：23 → 34
- 新覆盖：FA-ATTR-LIST、FA-AXIS-LAST、FA-IDX-GATHER、FA-IDX-PERMUTE、FA-IDX-WHERE、FA-LINALG-DOT、FA-MATH-ADD、FA-MATH-EXP、FA-MATH-GELU、FA-MATH-MAXMIN、FA-MATH-MUL、FA-MATH-SIGMOID、FA-MATH-SINCOS、FA-MATH-SQRT、FA-MULTI-INPUT、FA-NORM-RMS、FA-NORM-SOFTMAX、FA-REDUCE-ARG、FA-REDUCE-MAX、FA-REDUCE-MIN、FA-REDUCE-SUM、FA-STABLE-MAXSUB
- 不再覆盖：FA-DT-BF16、FA-DT-INT64、FA-LOAD-LARGE、FA-LOAD-XLARGE、FA-NUM-EXTREME、FA-PAT-COALESCE-GATHER、FA-PAT-VEC-LOAD、FA-PROG-GRID1D、FA-PROG-GRID3D、FA-SHAPE-CUBE、FA-SHAPE-FLAT

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.7-ascend-contract：generation-log.yaml 共 11 条记录（日志起点 2026-10-23T04:05:58.765277+00:00，此前构建无日志）
- v0.8-official-candidates：generation-log.yaml 共 3 条记录（日志起点 2026-10-23T10:12:35.589608+00:00，此前构建无日志）


## v0.8-official-candidates → v0.8.1-official-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-23T10:13:44.636556+00:00 | 2026-10-23T15:10:58.907323+00:00 |
| 清单哈希 | `249394cfdc65…` | `a614114dbc93…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 68 | 68 |
| 用例数 | 1360 | 1360 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1360 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：34 → 34
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.8-official-candidates：generation-log.yaml 共 3 条记录（日志起点 2026-10-23T10:12:35.589608+00:00，此前构建无日志）
- v0.8.1-official-candidates：generation-log.yaml 共 3 条记录（日志起点 2026-10-23T15:10:40.062962+00:00，此前构建无日志）


## v0.8.1-official-candidates → v0.9-official-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-23T15:10:58.907323+00:00 | 2026-10-24T02:58:55.566034+00:00 |
| 清单哈希 | `a614114dbc93…` | `885788c6a9be…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 68 | 68 |
| 用例数 | 1360 | 1360 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：1360 个（超过 20，按算子折叠）
  - official：1360
- 移除用例：1360 个（超过 20，按算子折叠）
  - official：1360
- 同编号但内容变更：无
- 内容未变的用例：0 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| Abs/LARGE | 0 | 1 | +1 |
| Abs/SMALL | 18 | 16 | -2 |
| Abs/XLARGE | 0 | 1 | +1 |
| Add/LARGE | 0 | 1 | +1 |
| Add/SMALL | 18 | 16 | -2 |
| Add/XLARGE | 0 | 1 | +1 |
| ArgMax/LARGE | 0 | 1 | +1 |
| ArgMax/MEDIUM | 2 | 1 | -1 |
| ArgMax/SMALL | 18 | 17 | -1 |
| ArgMax/XLARGE | 0 | 1 | +1 |
| ArgMin/LARGE | 0 | 1 | +1 |
| ArgMin/MEDIUM | 2 | 1 | -1 |
| ArgMin/SMALL | 18 | 17 | -1 |
| ArgMin/XLARGE | 0 | 1 | +1 |
| BitwiseAnd/LARGE | 0 | 1 | +1 |
| BitwiseAnd/MEDIUM | 5 | 2 | -3 |
| BitwiseAnd/SMALL | 15 | 16 | +1 |
| BitwiseAnd/XLARGE | 0 | 1 | +1 |
| BitwiseNot/LARGE | 0 | 1 | +1 |
| BitwiseNot/MEDIUM | 5 | 2 | -3 |
| BitwiseNot/SMALL | 15 | 16 | +1 |
| BitwiseNot/XLARGE | 0 | 1 | +1 |
| BitwiseOr/LARGE | 0 | 1 | +1 |
| BitwiseOr/MEDIUM | 5 | 2 | -3 |
| BitwiseOr/SMALL | 15 | 16 | +1 |
| BitwiseOr/XLARGE | 0 | 1 | +1 |
| BitwiseXor/LARGE | 0 | 1 | +1 |
| BitwiseXor/MEDIUM | 5 | 2 | -3 |
| BitwiseXor/SMALL | 15 | 16 | +1 |
| BitwiseXor/XLARGE | 0 | 1 | +1 |
| Ceil/LARGE | 0 | 1 | +1 |
| Ceil/SMALL | 18 | 16 | -2 |
| Ceil/XLARGE | 0 | 1 | +1 |
| Clamp/LARGE | 0 | 1 | +1 |
| Clamp/SMALL | 18 | 16 | -2 |
| Clamp/XLARGE | 0 | 1 | +1 |
| Concat/LARGE | 0 | 1 | +1 |
| Concat/MEDIUM | 2 | 1 | -1 |
| Concat/SMALL | 18 | 17 | -1 |
| Concat/XLARGE | 0 | 1 | +1 |
| Cos/LARGE | 0 | 1 | +1 |
| Cos/SMALL | 18 | 16 | -2 |
| Cos/XLARGE | 0 | 1 | +1 |
| Cosh/LARGE | 0 | 1 | +1 |
| Cosh/SMALL | 18 | 16 | -2 |
| Cosh/XLARGE | 0 | 1 | +1 |
| CumProd/LARGE | 0 | 1 | +1 |
| CumProd/MEDIUM | 2 | 1 | -1 |
| CumProd/SMALL | 18 | 17 | -1 |
| CumProd/XLARGE | 0 | 1 | +1 |
| CumSum/LARGE | 0 | 1 | +1 |
| CumSum/MEDIUM | 2 | 1 | -1 |
| CumSum/SMALL | 18 | 17 | -1 |
| CumSum/XLARGE | 0 | 1 | +1 |
| Div/LARGE | 0 | 1 | +1 |
| Div/SMALL | 18 | 16 | -2 |
| Div/XLARGE | 0 | 1 | +1 |
| Equal/LARGE | 0 | 1 | +1 |
| Equal/SMALL | 18 | 16 | -2 |
| Equal/XLARGE | 0 | 1 | +1 |
| Exp/LARGE | 0 | 1 | +1 |
| Exp/SMALL | 18 | 16 | -2 |
| Exp/XLARGE | 0 | 1 | +1 |
| Exp2/LARGE | 0 | 1 | +1 |
| Exp2/SMALL | 18 | 16 | -2 |
| Exp2/XLARGE | 0 | 1 | +1 |
| Expm1/LARGE | 0 | 1 | +1 |
| Expm1/SMALL | 18 | 16 | -2 |
| Expm1/XLARGE | 0 | 1 | +1 |
| Flip/LARGE | 0 | 1 | +1 |
| Flip/MEDIUM | 2 | 1 | -1 |
| Flip/SMALL | 18 | 17 | -1 |
| Flip/XLARGE | 0 | 1 | +1 |
| Floor/LARGE | 0 | 1 | +1 |
| Floor/SMALL | 18 | 16 | -2 |
| Floor/XLARGE | 0 | 1 | +1 |
| FloorDiv/LARGE | 0 | 1 | +1 |
| FloorDiv/MEDIUM | 5 | 2 | -3 |
| FloorDiv/SMALL | 15 | 16 | +1 |
| FloorDiv/XLARGE | 0 | 1 | +1 |
| Gather/LARGE | 0 | 1 | +1 |
| Gather/MEDIUM | 2 | 1 | -1 |
| Gather/SMALL | 18 | 17 | -1 |
| Gather/XLARGE | 0 | 1 | +1 |
| Gelu/LARGE | 0 | 1 | +1 |
| Gelu/SMALL | 18 | 16 | -2 |
| Gelu/XLARGE | 0 | 1 | +1 |
| Greater/LARGE | 0 | 1 | +1 |
| Greater/SMALL | 18 | 16 | -2 |
| Greater/XLARGE | 0 | 1 | +1 |
| GreaterEqual/LARGE | 0 | 1 | +1 |
| GreaterEqual/SMALL | 18 | 16 | -2 |
| GreaterEqual/XLARGE | 0 | 1 | +1 |
| LayerNorm/LARGE | 0 | 1 | +1 |
| LayerNorm/MEDIUM | 2 | 1 | -1 |
| LayerNorm/SMALL | 18 | 17 | -1 |
| LayerNorm/XLARGE | 0 | 1 | +1 |
| Less/LARGE | 0 | 1 | +1 |
| Less/SMALL | 18 | 16 | -2 |
| Less/XLARGE | 0 | 1 | +1 |
| LessEqual/LARGE | 0 | 1 | +1 |
| LessEqual/SMALL | 18 | 16 | -2 |
| LessEqual/XLARGE | 0 | 1 | +1 |
| Log/LARGE | 0 | 1 | +1 |
| Log/SMALL | 18 | 16 | -2 |
| Log/XLARGE | 0 | 1 | +1 |
| Log10/LARGE | 0 | 1 | +1 |
| Log10/SMALL | 18 | 16 | -2 |
| Log10/XLARGE | 0 | 1 | +1 |
| Log1p/LARGE | 0 | 1 | +1 |
| Log1p/SMALL | 18 | 16 | -2 |
| Log1p/XLARGE | 0 | 1 | +1 |
| Log2/LARGE | 0 | 1 | +1 |
| Log2/SMALL | 18 | 16 | -2 |
| Log2/XLARGE | 0 | 1 | +1 |
| LogicalAnd/LARGE | 0 | 1 | +1 |
| LogicalAnd/SMALL | 18 | 16 | -2 |
| LogicalAnd/XLARGE | 0 | 1 | +1 |
| LogicalOr/LARGE | 0 | 1 | +1 |
| LogicalOr/SMALL | 18 | 16 | -2 |
| LogicalOr/XLARGE | 0 | 1 | +1 |
| Matmul/LARGE | 0 | 1 | +1 |
| Matmul/MEDIUM | 5 | 1 | -4 |
| Matmul/SMALL | 15 | 18 | +3 |
| Maximum/LARGE | 0 | 1 | +1 |
| Maximum/SMALL | 18 | 16 | -2 |
| Maximum/XLARGE | 0 | 1 | +1 |
| Minimum/LARGE | 0 | 1 | +1 |
| Minimum/SMALL | 18 | 16 | -2 |
| Minimum/XLARGE | 0 | 1 | +1 |
| Mish/LARGE | 0 | 1 | +1 |
| Mish/SMALL | 18 | 16 | -2 |
| Mish/XLARGE | 0 | 1 | +1 |
| Mod/LARGE | 0 | 1 | +1 |
| Mod/SMALL | 18 | 16 | -2 |
| Mod/XLARGE | 0 | 1 | +1 |
| Mul/LARGE | 0 | 1 | +1 |
| Mul/SMALL | 18 | 16 | -2 |
| Mul/XLARGE | 0 | 1 | +1 |
| Neg/LARGE | 0 | 1 | +1 |
| Neg/SMALL | 18 | 16 | -2 |
| Neg/XLARGE | 0 | 1 | +1 |
| NotEqual/LARGE | 0 | 1 | +1 |
| NotEqual/SMALL | 18 | 16 | -2 |
| NotEqual/XLARGE | 0 | 1 | +1 |
| Pow/LARGE | 0 | 1 | +1 |
| Pow/SMALL | 18 | 16 | -2 |
| Pow/XLARGE | 0 | 1 | +1 |
| ReduceMax/LARGE | 0 | 1 | +1 |
| ReduceMax/MEDIUM | 2 | 1 | -1 |
| ReduceMax/SMALL | 18 | 17 | -1 |
| ReduceMax/XLARGE | 0 | 1 | +1 |
| ReduceMean/LARGE | 0 | 1 | +1 |
| ReduceMean/MEDIUM | 2 | 1 | -1 |
| ReduceMean/SMALL | 18 | 17 | -1 |
| ReduceMean/XLARGE | 0 | 1 | +1 |
| ReduceMin/LARGE | 0 | 1 | +1 |
| ReduceMin/MEDIUM | 2 | 1 | -1 |
| ReduceMin/SMALL | 18 | 17 | -1 |
| ReduceMin/XLARGE | 0 | 1 | +1 |
| ReduceSum/LARGE | 0 | 1 | +1 |
| ReduceSum/MEDIUM | 2 | 1 | -1 |
| ReduceSum/SMALL | 18 | 17 | -1 |
| ReduceSum/XLARGE | 0 | 1 | +1 |
| Relu/LARGE | 0 | 1 | +1 |
| Relu/SMALL | 18 | 16 | -2 |
| Relu/XLARGE | 0 | 1 | +1 |
| RmsNorm/LARGE | 0 | 1 | +1 |
| RmsNorm/MEDIUM | 2 | 1 | -1 |
| RmsNorm/SMALL | 18 | 17 | -1 |
| RmsNorm/XLARGE | 0 | 1 | +1 |
| Rsqrt/LARGE | 0 | 1 | +1 |
| Rsqrt/SMALL | 18 | 16 | -2 |
| Rsqrt/XLARGE | 0 | 1 | +1 |
| Sigmoid/LARGE | 0 | 1 | +1 |
| Sigmoid/SMALL | 18 | 16 | -2 |
| Sigmoid/XLARGE | 0 | 1 | +1 |
| Signbit/LARGE | 0 | 1 | +1 |
| Signbit/SMALL | 18 | 16 | -2 |
| Signbit/XLARGE | 0 | 1 | +1 |
| Sin/LARGE | 0 | 1 | +1 |
| Sin/SMALL | 18 | 16 | -2 |
| Sin/XLARGE | 0 | 1 | +1 |
| Sinh/LARGE | 0 | 1 | +1 |
| Sinh/SMALL | 18 | 16 | -2 |
| Sinh/XLARGE | 0 | 1 | +1 |
| Softmax/LARGE | 0 | 1 | +1 |
| Softmax/MEDIUM | 2 | 1 | -1 |
| Softmax/SMALL | 18 | 17 | -1 |
| Softmax/XLARGE | 0 | 1 | +1 |
| SortValues/LARGE | 0 | 1 | +1 |
| SortValues/MEDIUM | 2 | 1 | -1 |
| SortValues/SMALL | 18 | 17 | -1 |
| SortValues/XLARGE | 0 | 1 | +1 |
| Sqrt/LARGE | 0 | 1 | +1 |
| Sqrt/SMALL | 18 | 16 | -2 |
| Sqrt/XLARGE | 0 | 1 | +1 |
| Sub/LARGE | 0 | 1 | +1 |
| Sub/SMALL | 18 | 16 | -2 |
| Sub/XLARGE | 0 | 1 | +1 |
| SwiGlu/LARGE | 0 | 1 | +1 |
| SwiGlu/MEDIUM | 2 | 1 | -1 |
| SwiGlu/SMALL | 18 | 17 | -1 |
| SwiGlu/XLARGE | 0 | 1 | +1 |
| Tan/LARGE | 0 | 1 | +1 |
| Tan/SMALL | 18 | 16 | -2 |
| Tan/XLARGE | 0 | 1 | +1 |
| Tanh/LARGE | 0 | 1 | +1 |
| Tanh/SMALL | 18 | 16 | -2 |
| Tanh/XLARGE | 0 | 1 | +1 |
| TopKValues/LARGE | 0 | 1 | +1 |
| TopKValues/MEDIUM | 2 | 4 | +2 |
| TopKValues/SMALL | 18 | 14 | -4 |
| TopKValues/XLARGE | 0 | 1 | +1 |
| Transpose/LARGE | 0 | 1 | +1 |
| Transpose/MEDIUM | 2 | 1 | -1 |
| Transpose/SMALL | 18 | 17 | -1 |
| Transpose/XLARGE | 0 | 1 | +1 |
| Trunc/LARGE | 0 | 1 | +1 |
| Trunc/SMALL | 18 | 16 | -2 |
| Trunc/XLARGE | 0 | 1 | +1 |
| Where/LARGE | 0 | 1 | +1 |
| Where/SMALL | 18 | 16 | -2 |
| Where/XLARGE | 0 | 1 | +1 |
| XorSum/LARGE | 0 | 1 | +1 |
| XorSum/MEDIUM | 5 | 1 | -4 |
| XorSum/SMALL | 15 | 17 | +2 |
| XorSum/XLARGE | 0 | 1 | +1 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| Abs/float16 | 8 | 14 | +6 |
| Abs/float32 | 12 | 6 | -6 |
| Add/float16 | 8 | 14 | +6 |
| Add/float32 | 12 | 6 | -6 |
| ArgMax/float16 | 8 | 12 | +4 |
| ArgMax/float32 | 12 | 8 | -4 |
| ArgMin/float16 | 8 | 12 | +4 |
| ArgMin/float32 | 12 | 8 | -4 |
| Ceil/float16 | 8 | 15 | +7 |
| Ceil/float32 | 12 | 5 | -7 |
| Clamp/float16 | 8 | 14 | +6 |
| Clamp/float32 | 12 | 6 | -6 |
| Concat/float16 | 8 | 12 | +4 |
| Concat/float32 | 12 | 8 | -4 |
| Cos/float16 | 8 | 15 | +7 |
| Cos/float32 | 12 | 5 | -7 |
| Cosh/float16 | 8 | 15 | +7 |
| Cosh/float32 | 12 | 5 | -7 |
| CumProd/float16 | 8 | 12 | +4 |
| CumProd/float32 | 12 | 8 | -4 |
| CumSum/float16 | 8 | 12 | +4 |
| CumSum/float32 | 12 | 8 | -4 |
| Div/float16 | 8 | 14 | +6 |
| Div/float32 | 12 | 6 | -6 |
| Equal/float16 | 8 | 14 | +6 |
| Equal/float32 | 12 | 6 | -6 |
| Exp/float16 | 8 | 15 | +7 |
| Exp/float32 | 12 | 5 | -7 |
| Exp2/float16 | 8 | 15 | +7 |
| Exp2/float32 | 12 | 5 | -7 |
| Expm1/float16 | 8 | 15 | +7 |
| Expm1/float32 | 12 | 5 | -7 |
| Flip/float16 | 8 | 12 | +4 |
| Flip/float32 | 12 | 8 | -4 |
| Floor/float16 | 8 | 15 | +7 |
| Floor/float32 | 12 | 5 | -7 |
| Gather/float16 | 8 | 11 | +3 |
| Gather/float32 | 12 | 9 | -3 |
| Gelu/float16 | 8 | 15 | +7 |
| Gelu/float32 | 12 | 5 | -7 |
| Greater/float16 | 8 | 14 | +6 |
| Greater/float32 | 12 | 6 | -6 |
| GreaterEqual/float16 | 8 | 14 | +6 |
| GreaterEqual/float32 | 12 | 6 | -6 |
| LayerNorm/float16 | 8 | 11 | +3 |
| LayerNorm/float32 | 12 | 9 | -3 |
| Less/float16 | 8 | 14 | +6 |
| Less/float32 | 12 | 6 | -6 |
| LessEqual/float16 | 8 | 14 | +6 |
| LessEqual/float32 | 12 | 6 | -6 |
| Log/float16 | 8 | 13 | +5 |
| Log/float32 | 12 | 7 | -5 |
| Log10/float16 | 8 | 13 | +5 |
| Log10/float32 | 12 | 7 | -5 |
| Log1p/float16 | 8 | 15 | +7 |
| Log1p/float32 | 12 | 5 | -7 |
| Log2/float16 | 8 | 13 | +5 |
| Log2/float32 | 12 | 7 | -5 |
| LogicalAnd/float16 | 8 | 15 | +7 |
| LogicalAnd/float32 | 12 | 5 | -7 |
| LogicalOr/float16 | 8 | 15 | +7 |
| LogicalOr/float32 | 12 | 5 | -7 |
| Maximum/float16 | 8 | 14 | +6 |
| Maximum/float32 | 12 | 6 | -6 |
| Minimum/float16 | 8 | 14 | +6 |
| Minimum/float32 | 12 | 6 | -6 |
| Mish/float16 | 8 | 15 | +7 |
| Mish/float32 | 12 | 5 | -7 |
| Mod/float16 | 8 | 15 | +7 |
| Mod/float32 | 12 | 5 | -7 |
| Mul/float16 | 8 | 14 | +6 |
| Mul/float32 | 12 | 6 | -6 |
| Neg/float16 | 8 | 14 | +6 |
| Neg/float32 | 12 | 6 | -6 |
| NotEqual/float16 | 8 | 14 | +6 |
| NotEqual/float32 | 12 | 6 | -6 |
| Pow/float16 | 8 | 13 | +5 |
| Pow/float32 | 12 | 7 | -5 |
| ReduceMax/float16 | 8 | 12 | +4 |
| ReduceMax/float32 | 12 | 8 | -4 |
| ReduceMean/float16 | 8 | 12 | +4 |
| ReduceMean/float32 | 12 | 8 | -4 |
| ReduceMin/float16 | 8 | 12 | +4 |
| ReduceMin/float32 | 12 | 8 | -4 |
| ReduceSum/float16 | 8 | 12 | +4 |
| ReduceSum/float32 | 12 | 8 | -4 |
| Relu/float16 | 8 | 14 | +6 |
| Relu/float32 | 12 | 6 | -6 |
| RmsNorm/float16 | 8 | 11 | +3 |
| RmsNorm/float32 | 12 | 9 | -3 |
| Rsqrt/float16 | 8 | 13 | +5 |
| Rsqrt/float32 | 12 | 7 | -5 |
| Sigmoid/float16 | 8 | 14 | +6 |
| Sigmoid/float32 | 12 | 6 | -6 |
| Signbit/float16 | 8 | 15 | +7 |
| Signbit/float32 | 12 | 5 | -7 |
| Sin/float16 | 8 | 15 | +7 |
| Sin/float32 | 12 | 5 | -7 |
| Sinh/float16 | 8 | 15 | +7 |
| Sinh/float32 | 12 | 5 | -7 |
| Softmax/float16 | 8 | 11 | +3 |
| Softmax/float32 | 12 | 9 | -3 |
| SortValues/float16 | 8 | 12 | +4 |
| SortValues/float32 | 12 | 8 | -4 |
| Sqrt/float16 | 8 | 15 | +7 |
| Sqrt/float32 | 12 | 5 | -7 |
| Sub/float16 | 8 | 14 | +6 |
| Sub/float32 | 12 | 6 | -6 |
| SwiGlu/float16 | 8 | 11 | +3 |
| SwiGlu/float32 | 12 | 9 | -3 |
| Tan/float16 | 8 | 15 | +7 |
| Tan/float32 | 12 | 5 | -7 |
| Tanh/float16 | 8 | 14 | +6 |
| Tanh/float32 | 12 | 6 | -6 |
| TopKValues/float16 | 8 | 16 | +8 |
| TopKValues/float32 | 12 | 4 | -8 |
| Transpose/float16 | 8 | 12 | +4 |
| Transpose/float32 | 12 | 8 | -4 |
| Trunc/float16 | 8 | 15 | +7 |
| Trunc/float32 | 12 | 5 | -7 |
| Where/float16 | 8 | 14 | +6 |
| Where/float32 | 12 | 6 | -6 |

### 覆盖原子得失

- 原子覆盖数：34 → 49
- 新覆盖：FA-LOAD-LARGE、FA-LOAD-STRIDED、FA-LOAD-XLARGE、FA-NUM-EXTREME、FA-NUM-MIXED、FA-NUM-ZEROS、FA-PAT-COALESCE-GATHER、FA-PAT-MASK-TAIL、FA-PAT-TILE-GEMM、FA-PAT-VEC-LOAD、FA-PROG-GRID1D、FA-PROG-GRID3D、FA-PROG-GRID4D、FA-SHAPE-CUBE、FA-SHAPE-FLAT
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.8.1-official-candidates：generation-log.yaml 共 3 条记录（日志起点 2026-10-23T15:10:40.062962+00:00，此前构建无日志）
- v0.9-official-candidates：generation-log.yaml 共 4 条记录（日志起点 2026-10-24T02:58:14.714550+00:00，此前构建无日志）


## v0.9-official-candidates → v0.10-coverage-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-24T02:58:55.566034+00:00 | 2026-10-25T00:53:08.085296+00:00 |
| 清单哈希 | `885788c6a9be…` | `9542a8440b06…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 68 | 74 |
| 用例数 | 1360 | 1377 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子 6 个：ApplyRotaryPosEmb（Coverage extension; not an official-source admission）、CrossEntropyLoss（Coverage extension; not an official-source admission）、Cummin（Coverage extension; not an official-source admission）、GroupedMatmul（Coverage extension; not an official-source admission）、MaskedSoftmax（Coverage extension; not an official-source admission）、Scatter（Coverage extension; not an official-source admission）
- 移除算子：无

### 逐算子用例数变化

| 算子 | 旧 | 新 | Δ |
|---|---|---|---|
| ApplyRotaryPosEmb | 0 | 2 | +2 |
| CrossEntropyLoss | 0 | 2 | +2 |
| Cummin | 0 | 2 | +2 |
| GroupedMatmul | 0 | 2 | +2 |
| MaskedSoftmax | 0 | 5 | +5 |
| Scatter | 0 | 2 | +2 |
| Transpose | 20 | 22 | +2 |
（其余 67 个算子用例数不变）

### 用例级变化

- 新增用例：17 个
  - coverage-apply_rotary_pos_emb-037
  - coverage-apply_rotary_pos_emb-038
  - coverage-cross_entropy_loss-032
  - coverage-cross_entropy_loss-033
  - coverage-cummin-021
  - coverage-cummin-022
  - coverage-grouped_matmul-030
  - coverage-grouped_matmul-048
  - coverage-masked_softmax-052
  - coverage-masked_softmax-057
  - coverage-masked_softmax-058
  - coverage-masked_softmax-060
  - coverage-masked_softmax-061
  - coverage-scatter-018
  - coverage-scatter-019
  - coverage-transpose-063
  - coverage-transpose-064
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1360 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| ApplyRotaryPosEmb/SMALL | 0 | 2 | +2 |
| CrossEntropyLoss/SMALL | 0 | 2 | +2 |
| Cummin/SMALL | 0 | 2 | +2 |
| GroupedMatmul/SMALL | 0 | 2 | +2 |
| MaskedSoftmax/SMALL | 0 | 5 | +5 |
| Scatter/SMALL | 0 | 2 | +2 |
| Transpose/SMALL | 17 | 19 | +2 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| ApplyRotaryPosEmb/float32 | 0 | 2 | +2 |
| CrossEntropyLoss/float32 | 0 | 2 | +2 |
| Cummin/float32 | 0 | 2 | +2 |
| GroupedMatmul/bfloat16 | 0 | 1 | +1 |
| GroupedMatmul/float32 | 0 | 1 | +1 |
| MaskedSoftmax/bfloat16 | 0 | 1 | +1 |
| MaskedSoftmax/int8 | 0 | 2 | +2 |
| MaskedSoftmax/uint8 | 0 | 2 | +2 |
| Scatter/int64 | 0 | 2 | +2 |
| Transpose/int16 | 0 | 2 | +2 |

### 覆盖原子得失

- 原子覆盖数：49 → 67
- 新覆盖：FA-AXIS-FIRST、FA-AXIS-INNER、FA-CONCUR-ACCUM、FA-CTRL-STATIC-UNROLL、FA-DT-BF16、FA-DT-INT16、FA-DT-INT64、FA-DT-INT8、FA-DT-UINT8、FA-FUSE-2STAGE、FA-FUSE-CROSS-CATEGORY、FA-FUSE-ELEM-INTRA、FA-FUSE-INTERM-ELIDE、FA-FUSE-RED-INTRA、FA-IDX-SCATTER、FA-LINALG-BIAS、FA-MULTI-OUTPUT、FA-STABLE-LOGSUMEXP
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.9-official-candidates：generation-log.yaml 共 4 条记录（日志起点 2026-10-24T02:58:14.714550+00:00，此前构建无日志）
- v0.10-coverage-candidates：generation-log.yaml 共 4 条记录（日志起点 2026-10-25T00:52:00.945182+00:00，此前构建无日志）


## v0.10-coverage-candidates → v0.11-implementation-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-25T00:53:08.085296+00:00 | 2026-10-25T02:16:00.909942+00:00 |
| 清单哈希 | `9542a8440b06…` | `dbb84a6d35bb…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1377 | 1377 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1377 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：67 → 67
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.10-coverage-candidates：generation-log.yaml 共 4 条记录（日志起点 2026-10-25T00:52:00.945182+00:00，此前构建无日志）
- v0.11-implementation-candidates：generation-log.yaml 共 4 条记录（日志起点 2026-10-25T02:15:22.431365+00:00，此前构建无日志）


## v0.11-implementation-candidates → v0.11.1-contract-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-25T02:16:00.909942+00:00 | 2026-10-25T03:53:04.419898+00:00 |
| 清单哈希 | `dbb84a6d35bb…` | `cc44177bd737…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1377 | 1377 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1377 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：67 → 67
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.11-implementation-candidates：generation-log.yaml 共 4 条记录（日志起点 2026-10-25T02:15:22.431365+00:00，此前构建无日志）
- v0.11.1-contract-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-25T03:51:59.024339+00:00，此前构建无日志）


## v0.11.1-contract-candidates → v0.12-compat-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-25T03:53:04.419898+00:00 | 2026-10-26T04:17:53.142053+00:00 |
| 清单哈希 | `cc44177bd737…` | `20a4d6c5d76e…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1377 | 1377 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1377 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：67 → 67
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.11.1-contract-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-25T03:51:59.024339+00:00，此前构建无日志）
- v0.12-compat-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-26T04:17:18.681828+00:00，此前构建无日志）


## v0.12-compat-candidates → v0.13-evidence-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-26T04:17:53.142053+00:00 | 2026-10-26T08:12:55.492367+00:00 |
| 清单哈希 | `20a4d6c5d76e…` | `a5cab486bef0…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1377 | 1377 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1377 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：67 → 67
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.12-compat-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-26T04:17:18.681828+00:00，此前构建无日志）
- v0.13-evidence-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-26T08:12:21.229701+00:00，此前构建无日志）


## v0.13-evidence-candidates → v0.14-backend-evidence-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-26T08:12:55.492367+00:00 | 2026-10-26T09:41:07.309312+00:00 |
| 清单哈希 | `a5cab486bef0…` | `35ed2840195b…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1377 | 1377 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1377 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：67 → 67
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.13-evidence-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-26T08:12:21.229701+00:00，此前构建无日志）
- v0.14-backend-evidence-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-26T09:40:10.252240+00:00，此前构建无日志）


## v0.14-backend-evidence-candidates → v0.15-backend-capture-fix-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-26T09:41:07.309312+00:00 | 2026-10-26T11:26:33.137455+00:00 |
| 清单哈希 | `35ed2840195b…` | `ad023d111506…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1377 | 1377 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1377 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：67 → 67
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.14-backend-evidence-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-26T09:40:10.252240+00:00，此前构建无日志）
- v0.15-backend-capture-fix-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-26T11:25:35.516411+00:00，此前构建无日志）


## v0.15-backend-capture-fix-candidates → v0.16-backend-pattern-certification-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-26T11:26:33.137455+00:00 | 2026-10-27T01:06:16.235205+00:00 |
| 清单哈希 | `ad023d111506…` | `e758eaef22e0…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1377 | 1377 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1377 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：67 → 67
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.15-backend-capture-fix-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-26T11:25:35.516411+00:00，此前构建无日志）
- v0.16-backend-pattern-certification-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-27T01:05:30.846898+00:00，此前构建无日志）


## v0.16-backend-pattern-certification-candidates → v0.17-pattern-completion-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-27T01:06:16.235205+00:00 | 2026-10-27T08:40:08.357588+00:00 |
| 清单哈希 | `e758eaef22e0…` | `95ba03ce7a13…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1377 | 1385 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

| 算子 | 旧 | 新 | Δ |
|---|---|---|---|
| Cummin | 2 | 4 | +2 |
| GroupedMatmul | 2 | 4 | +2 |
| Matmul | 20 | 22 | +2 |
| ReduceMax | 20 | 22 | +2 |
（其余 70 个算子用例数不变）

### 用例级变化

- 新增用例：8 个
  - pattern-cummin-002
  - pattern-cummin-003
  - pattern-grouped_matmul-006
  - pattern-grouped_matmul-007
  - pattern-matmul-004
  - pattern-matmul-005
  - pattern-reduce_max-000
  - pattern-reduce_max-001
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1377 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| Cummin/LARGE | 0 | 2 | +2 |
| GroupedMatmul/MEDIUM | 0 | 2 | +2 |
| Matmul/LARGE | 1 | 2 | +1 |
| Matmul/XLARGE | 0 | 1 | +1 |
| ReduceMax/LARGE | 1 | 2 | +1 |
| ReduceMax/XLARGE | 1 | 2 | +1 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| Cummin/float32 | 2 | 4 | +2 |
| GroupedMatmul/float16 | 0 | 2 | +2 |
| Matmul/float16 | 20 | 22 | +2 |
| ReduceMax/float32 | 8 | 10 | +2 |

### 覆盖原子得失

- 原子覆盖数：67 → 71
- 新覆盖：FA-PAT-MULTIPASS-REDUCE、FA-PAT-SCAN-BLOCKED、FA-PAT-SPLIT-K、FA-PAT-VEC-BROADCAST
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.16-backend-pattern-certification-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-27T01:05:30.846898+00:00，此前构建无日志）
- v0.17-pattern-completion-candidates：generation-log.yaml 共 12 条记录（日志起点 2026-10-27T08:29:19.616119+00:00，此前构建无日志）


## v0.17-pattern-completion-candidates → v0.18-scan-certification-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-27T08:40:08.357588+00:00 | 2026-10-27T09:55:56.569631+00:00 |
| 清单哈希 | `95ba03ce7a13…` | `0890185080c5…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1385 | 1385 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

所有算子用例数不变。

### 用例级变化

- 新增用例：无
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1385 个

### 规模分布变化（Δ≠0 的算子×档位）

无变化。

### dtype 分布变化（Δ≠0 的算子×dtype）

无变化。

### 覆盖原子得失

- 原子覆盖数：71 → 71
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.17-pattern-completion-candidates：generation-log.yaml 共 12 条记录（日志起点 2026-10-27T08:29:19.616119+00:00，此前构建无日志）
- v0.18-scan-certification-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-27T09:55:15.486102+00:00，此前构建无日志）


## v0.18-scan-certification-candidates → v0.19-workload-completion-candidates

### 版本身份

| 项 | 旧 | 新 |
|---|---|---|
| 冻结时间 | 2026-10-27T09:55:56.569631+00:00 | 2026-11-13T14:50:46.158853+00:00 |
| 清单哈希 | `0890185080c5…` | `bdc1e68beb01…` |
| 评测规范版本 | 0.1-mvp | 0.1-mvp |
| 能力点清单版本 | 1.1-fusion | 1.1-fusion |
| 算子数 | 74 | 74 |
| 用例数 | 1385 | 1486 |
| 门禁剔除数 | 0 | 0 |

### 算子增删

- 新增算子：无
- 移除算子：无

### 逐算子用例数变化

| 算子 | 旧 | 新 | Δ |
|---|---|---|---|
| ApplyRotaryPosEmb | 2 | 20 | +18 |
| CrossEntropyLoss | 2 | 20 | +18 |
| Cummin | 4 | 20 | +16 |
| GroupedMatmul | 4 | 20 | +16 |
| MaskedSoftmax | 5 | 20 | +15 |
| Scatter | 2 | 20 | +18 |
（其余 68 个算子用例数不变）

### 用例级变化

- 新增用例：101 个（超过 20，按算子折叠）
  - workload：101
- 移除用例：无
- 同编号但内容变更：无
- 内容未变的用例：1385 个

### 规模分布变化（Δ≠0 的算子×档位）

| 算子/档位 | 旧 | 新 | Δ |
|---|---|---|---|
| ApplyRotaryPosEmb/LARGE | 0 | 5 | +5 |
| ApplyRotaryPosEmb/MEDIUM | 0 | 5 | +5 |
| ApplyRotaryPosEmb/SMALL | 2 | 5 | +3 |
| ApplyRotaryPosEmb/XLARGE | 0 | 5 | +5 |
| CrossEntropyLoss/LARGE | 0 | 5 | +5 |
| CrossEntropyLoss/MEDIUM | 0 | 5 | +5 |
| CrossEntropyLoss/SMALL | 2 | 5 | +3 |
| CrossEntropyLoss/XLARGE | 0 | 5 | +5 |
| Cummin/LARGE | 2 | 5 | +3 |
| Cummin/MEDIUM | 0 | 6 | +6 |
| Cummin/SMALL | 2 | 6 | +4 |
| Cummin/XLARGE | 0 | 3 | +3 |
| GroupedMatmul/LARGE | 0 | 5 | +5 |
| GroupedMatmul/MEDIUM | 2 | 5 | +3 |
| GroupedMatmul/SMALL | 2 | 5 | +3 |
| GroupedMatmul/XLARGE | 0 | 5 | +5 |
| MaskedSoftmax/LARGE | 0 | 5 | +5 |
| MaskedSoftmax/MEDIUM | 0 | 6 | +6 |
| MaskedSoftmax/SMALL | 5 | 6 | +1 |
| MaskedSoftmax/XLARGE | 0 | 3 | +3 |
| Scatter/LARGE | 0 | 5 | +5 |
| Scatter/MEDIUM | 0 | 5 | +5 |
| Scatter/SMALL | 2 | 5 | +3 |
| Scatter/XLARGE | 0 | 5 | +5 |

### dtype 分布变化（Δ≠0 的算子×dtype）

| 算子/dtype | 旧 | 新 | Δ |
|---|---|---|---|
| ApplyRotaryPosEmb/bfloat16 | 0 | 4 | +4 |
| ApplyRotaryPosEmb/float16 | 0 | 8 | +8 |
| ApplyRotaryPosEmb/float32 | 2 | 8 | +6 |
| CrossEntropyLoss/bfloat16 | 0 | 4 | +4 |
| CrossEntropyLoss/float16 | 0 | 8 | +8 |
| CrossEntropyLoss/float32 | 2 | 8 | +6 |
| Cummin/bfloat16 | 0 | 4 | +4 |
| Cummin/float16 | 0 | 6 | +6 |
| Cummin/float32 | 4 | 10 | +6 |
| GroupedMatmul/bfloat16 | 1 | 4 | +3 |
| GroupedMatmul/float16 | 2 | 7 | +5 |
| GroupedMatmul/float32 | 1 | 9 | +8 |
| MaskedSoftmax/bfloat16 | 1 | 4 | +3 |
| MaskedSoftmax/float16 | 0 | 4 | +4 |
| MaskedSoftmax/float32 | 0 | 8 | +8 |
| Scatter/bfloat16 | 0 | 3 | +3 |
| Scatter/float16 | 0 | 4 | +4 |
| Scatter/float32 | 0 | 4 | +4 |
| Scatter/int32 | 0 | 4 | +4 |
| Scatter/int64 | 2 | 5 | +3 |

### 覆盖原子得失

- 原子覆盖数：71 → 71
- 新覆盖：无
- 不再覆盖：无

### 门禁剔除记录（随清单冻结，永久可查）

- 旧版：零剔除
- 新版：零剔除

### 生成日志

- v0.18-scan-certification-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-10-27T09:55:15.486102+00:00，此前构建无日志）
- v0.19-workload-completion-candidates：generation-log.yaml 共 5 条记录（日志起点 2026-11-13T14:49:51.373238+00:00，此前构建无日志）
