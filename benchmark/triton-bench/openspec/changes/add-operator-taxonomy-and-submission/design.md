# Design

## Context

MVP 已归档，7 个 capability 固化为主 specs。动机见 `proposal.md`。

**既有状态（已实测）**：

- 5 个手写算子、20 个 case，全部属 Elementwise 与 Reduction
- `CaseSpec` 含 `kernel_path` / `entry`，即**评测方自带 kernel**
- 4 个昇腾环境上均 20/20 PASS，编译链已确证（产物含 `.npubin`）
- 共享环境，性能数据不可上榜

**参考资产**：`reference/cann-bench/`（53 算子、13 category、4 level）已入仓。
其 `docs/spec/submission_spec.md` 与 `docs/guide/submission_rules.md` 定义了
规则编号化的提交校验与 7 条反作弊行为规则，是本 design 的主要参照。

**一个关键约束差异**：cann-bench 需支持 AscendC / PyPTO / TileLang / Triton 多种
实现语言，因此**看不到源码**（C++ 编译产物），反作弊只能靠运行时 guard，
且布局变换类违规**只能人工审查**（怕误伤 Transpose / Gather 等以数据搬运为
核心语义的算子）。

我们只支持 triton-ascend（纯 Python），**源码可见**——这带来 cann-bench 没有的能力。

## Goals / Non-Goals

**Goals:**

- 建立 13 category × 4 level 的算子体系，覆盖面可度量
- 让外部提交者能上传算子并被鲁棒地评测、出分、报错
- 把尽可能多的合规判定自动化，减少人工审查
- 每项校验规则可测试（合法样例 + 非法样例）

**Non-Goals:**

- 不支持 triton 之外的实现语言
- 不做 Web 提交入口
- 不做跨类别 / 跨赛道的合成总分
- 不为 VVFusion 生成 case（无可行 triton 算子）

## Decisions

### D1 · 契约变更：case 不再绑定 kernel

把 `CaseSpec` 的 `kernel_path` / `entry` 移除，改为 `operator` 标识；被测实现由提交提供。

- **理由**：这是支持用户上传的**前提**。若 case 自带 kernel，评测方就在替被测方写实现。
- **代价**：破坏性变更，既有 20 个 case 需迁移。
- **备选**：保留 `kernel_path` 作为「缺省实现」，提交缺失时回退 → 否。
  那样会出现「评测方实现」与「提交实现」混跑，评分口径无法解释。

### D2 · 提交采用目录契约，不用 wheel

cann-bench 要求 `build.sh` + `dist/*.whl`；我们改为：

```
submissions/<submit-id>/
├── manifest.yaml
└── operators/<op>/kernel.py
```

- **理由**：triton kernel 是纯 Python 源文件，无编译环节。为其引入 wheel
  打包链路（建虚拟包 → 打包 → 安装 → 导入）是为零收益增加失败点。
- **代价**：无法封装复杂依赖。对 triton kernel 够用。
- **备选**：照搬 cann-bench 的 wheel+build.sh → 否，复杂度与收益不匹配。

### D3 · 校验分三阶段，规则编号化

```
静态 AST  →  加载  →  执行
```

每条规则有唯一编号（如 `SUB-STR-001`）、明确失败结果、合法/非法样例测试。

- **理由**：cann-bench 的做法，且实测证明有效——它让「为什么不合规」可被引述。
  我们此前吃过亏：异常被静默吞掉导致 `ORACLE_FAIL` 无从定位。
- **短路**：静态失败即中止，不进入加载与执行。省一次冷编译（实测 22 秒）。

### D4 · 反作弊：静态 AST 为主，运行时兜底

利用「源码可见」做静态检测：

```python
def softmax(x, dim=-1):
    return F.softmax(x, dim=dim)      # AST 一眼可见
```

- **理由**：比运行时 guard 更快（零执行）、更准（能定位到行号）、零误伤（可结合 category）。
- **备选**：仅运行时 guard（cann-bench 做法）→ 否，因为要跑一次才拦得住，且无法给行号。

### D5 · 按 category 差异化白名单（本设计的关键增项）

cann-bench 对布局变换类违规**只能人工审查**——因为它不知道算子的语义族。
我们知道 `category`，因此可以：

| category | 布局变换类操作 |
|---|---|
| `LayoutTransform` | **允许**（transpose / permute 本就是核心语义） |
| 其余 12 个 | **拦截** |

这把 cann-bench 的人工审查项**自动化**了。

- **理由**：分类分级体系带来的额外收益，cann-bench 未利用。
- **风险**：白名单需随 category 语义调整 → 每个 category 的白名单必须有测试。

### D6 · 算子选型按 triton 可行性分批

从 cann-bench 53 个算子中选，只选 triton 写得出来的。详见
`docs/operator-selection.md`。要点：

- **Contraction 只选 Matmul 类**（`grouped_matmul`），不选 Conv2D / DepthwiseConv2D
  —— 后者需显式 im2col，triton 极难
- **SortSelect 只选 ArgMax**，不选 TopK / Unique / NMS（需多 pass）
- **FusedComposite 只选 L1 简单融合**（`swi_glu`），不选 L4 的 MHA / MLA / GRU / LSTM
- **VVFusion 留占位**，不生成 case

证据：cann-bench 自己的 triton 示例只给了 L1 的 5 个算子（Exp / MaskedScale /
Mish / Sigmoid / SwiGlu），全是 Elementwise 与 MaskPredicate。

### D7 · 评分按 category / level 分组披露，仍不出总分

- **理由**：遵守我们自己的规范 §3.3。cann-bench 用 `0.2/0.3/0.5` 三元加权，
  但我们已明确禁止跨赛道合成总分，不因对齐外部而改变。
- **增项**：分组结果随 `scorer_version` 记录，支持重算留痕。

### D8 · case 由算子规格派生，五个维度

`shape` / `dtype` / `attrs` / `value_range` / 布局。
每 category 绑 5 个 case。总计 13 category × 5 = 60（VVFusion 为 0）。

## Risks / Trade-offs

| 风险 | 缓解 |
|---|---|
| 契约变更破坏既有 20 case | 迁移脚本：把既有算子映射为新 category（add/mul→Elementwise、sum→Reduction、softmax/layer_norm→Normalization），作为候补算子保留 |
| 静态 AST 误伤合法实现 | 按 category 白名单 + 每条规则配合法样例测试；误伤可申诉并调整白名单 |
| 某些 category 无人提交 kernel | case 规格先行（不依赖提交存在）；报告中把无提交的类别披露为「无数据」而非零分 |
| 冷编译 22 秒，60 case 全冷跑约 22 分钟 | 静态校验前置拦截（不合规不编译）；依赖 JIT 缓存复用 |
| 选型偏向简单算子，覆盖度不足 | 分批推进（第 1 批 7 类 / 第 2 批 5 类）；后续按可行性补充 |
| 共享环境，性能数据仍不可上榜 | 与 MVP 一致：性能数据标注不可验证，本变更不解决 |
| VVFusion 空缺导致 category 不全 | 保留占位并在报告列出为空缺，不静默省略 |

## Migration Plan

1. 建分类分级体系与算子规格 schema（不改动执行链路）
2. 迁移既有 20 case 到新契约（保留为候补算子）
3. 新增 14 个选定算子的规格与参考实现，生成 60 个 case
4. 改造执行器：从提交目录加载实现
5. 实现三阶段校验与反作弊（含规则测试）
6. 扩展评分与报告的分组披露
7. 端到端验证：用合法提交跑通，用各类非法提交验证拒绝

**回滚**：契约变更为破坏性，回滚需恢复 `CaseSpec.kernel_path`。
但第 1–3 步（分类 / 规格 / case 生成）本身不破坏执行，可独立保留；
真正的切换点是第 4 步。因此在第 4 步之前可分阶段验证。

## Open Questions

- `MaskPredicate` 只有 1 个算子（MaskedScale），5 个 case 的多样性受限 ——
  是否从其他来源补充算子？不影响本变更的任务分解。
- 静态 AST 白名单的粒度：按 category 还是按算子？当前选 category（更稳）；
  若后续出现误伤，再细化到算子级。
- 第 3 批（VVFusion）何时补：取决于 triton-ascend 对向量-向量融合的支持进展，
  不阻塞本变更。
