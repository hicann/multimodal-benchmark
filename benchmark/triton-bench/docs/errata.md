# 测试集勘误（errata）

**机制**（规则全文见 docs/versioning.md）：对**已打标签发布**的测试集版本，
发现缺陷时不允许悄悄修改——每条勘误在此登记、注明处置与受影响范围。
发布说明（docs/releases/）自动收录受影响版本的勘误条目。

**处置三分类**（沿用技术报告 §4.7 语义）：

| 分类 | 含义 | 后果 |
|---|---|---|
| AMEND | 修正容差 / 参考实现 / 判定参数 | 注明受影响用例；收录进下一个测试集版本 |
| RETRACT | 撤销用例（题目本身有误） | 触发 MAJOR+1，历史分数作废——**历史上尚未发生** |
| ANNOTATE | 只加注说明，不改判定 | 当前版本内生效 |

**草稿期修订的说明**：v0.3 / v0.4 / v0.5 均在**任何标签打点之前**冻结并
被真机轮次消费（"打标即发布"规则 2026-10-15 才建立）。下列发生在草稿期
的判定修正按勘误**补录**——它们不是"发布后勘误"，但同一机制登记后，
受影响版本与处置范围可查，不再散落在归档记录里。

---

## 条目

### ERR-001 gelu 的 f32 在昇腾 NPU 默认模式下实为 tanh 口径，默认容差冤枉正确实现

- 日期：2026-10-11
- 处置：AMEND
- 受影响版本：v0.3-multi-input-frozen、v0.4-cov20-frozen（草稿期）
- 受影响用例：gelu-f32-00 等 2 例（2026-10-11 v0.4 C1 轮实测）
- 内容：torch_npu 的 f32 gelu 默认走 tanh 近似而非 erf 精确口径，与
  参考实现（erf）在负尾区的偏差超过 f32 默认档（rel 2.2e-4 > 1e-4），
  正确实现被判 WRONG_RESULT。tolerance 1.1 为 gelu 登记 f32
  operator_overrides（abs=1e-3）。
- 证据：robustness-findings P1-6；tolerance.yaml 1.1 版本记录

### ERR-002 grouped_matmul 的 extreme 输入下参考自身 f32 累加噪声，冤枉正确实现

- 日期：2026-10-11
- 处置：AMEND
- 受影响版本：v0.3-multi-input-frozen、v0.4-cov20-frozen（草稿期）
- 受影响用例：grouped_matmul-f32-00/03（近零输出上 rel 4.3e-4 / 1.2e-4）
- 内容：±20σ 的 extreme 输入让参考实现自身的 f32 累加误差达到
  1.3e-5 量级，超过 f32 默认档 abs 1e-5——比参考更精确的实现反被判错。
  tolerance 1.1 为 grouped_matmul 登记 f32 operator_overrides（abs=1e-3）。
- 证据：robustness-findings P1-6；tolerance.yaml 1.1 版本记录

### ERR-003 bf16 档容差偏紧，1 ULP 偏差被判错

- 日期：2026-10-11
- 处置：AMEND
- 受影响版本：v0.3-multi-input-frozen、v0.4-cov20-frozen（草稿期）
- 受影响用例：grouped_matmul 的 bf16 用例 1 例
- 内容：bf16 档 rel 5e-3 装不下 1 ULP 的舍入差（bf16 的 8 位尾数下
  1 ULP ≈ 7.8e-3）。tolerance 1.1 放宽到 7.8e-3。
- 证据：tolerance.yaml 1.1 版本记录；spec-conflicts C2

### ERR-004 交叉熵容差键取错维度（case.dtype 应取输出 dtype）+ 判定合取改析取

- 日期：2026-10-09
- 处置：AMEND
- 受影响版本：v0.3-multi-input-frozen、v0.4-cov20-frozen（草稿期）
- 受影响用例：cross_entropy_loss 的整型 dtype 用例（输出实为 f32 标量）
- 内容：容差按 case.dtype 查表使整型用例要求精确相等，而该算子输出
  是 fp32——误差 7e-8 也判 WRONG_RESULT；同时 rel/abs 合取语义在
  消零输出上不可达。修正为按输出 dtype 查键、rel 或 abs 析取。
- 证据：robustness-findings P1-6；2026-10-09 真机实测修订

### ERR-005 MoE 门控参考实现的张量落在 CPU（缺 device=），评测子进程崩溃

- 日期：2026-10-13
- 处置：AMEND
- 受影响版本：v0.5-cov20-frozen（草稿期）
- 受影响用例：moe_gating_top_k_softmax 全部 15 个用例（首轮记录）
- 内容：参考实现内 torch.arange 未带 device= 生成 CPU 张量，与 NPU
  输入混合运算崩溃，15 个用例被误记。修复参考（arange 带输入设备）
  + evalchild 逐用例异常边界（单用例失败不再拖垮整轮）。语料在
  修复前未被任何对外发布消费。
- 证据：robustness-findings 第一轮附录；meta 设备纯度测试

### ERR-006 跨实现求和顺序噪声：f32 大 K 下 tl.dot 与 torch matmul 的固有分歧

- 日期：2026-10-13
- 处置：AMEND
- 受影响版本：v0.5-cov20-frozen（草稿期）
- 受影响用例：transpose_linear / rms_norm_linear / grouped_matmul_swiglu
  / unsorted_segment_sum 的 f32 大 K 用例（K≥2048）
- 内容：dot 归因探针实测 f32 分块 tl.dot 与 torch matmul 在 K≤1024
  逐位一致、K=2048 分歧 3.1e-4、K=65536 分歧 2.1e-3，且与分块大小
  无关——实现侧无抓手。tolerance 1.2 按归因证据登记四个算子的 f32
  覆盖（重放测试保证 10× 真错仍拦截）。
- 证据：spec-conflicts C2；dot-precision-report（记录 …-manual-1013-234810）

### ERR-007 交叉熵参考在 npu:0 上同进程两遍结果不一致（锚点自噪声）

- 日期：2026-10-14
- 处置：ANNOTATE
- 受影响版本：v0.5-cov20-frozen（草稿期）
- 受影响用例：cross_entropy_loss 的部分用例（逐轮随机出现；两窗口
  实测 9 例次，max_rel 全部 6~9×10⁻⁸）
- 内容：torch_npu 的 cross_entropy 归约存在执行间 ULP 级噪声。比
  判定容差（f32 档 rel 1e-4）低三个数量级以上，判定不受影响（历轮
  CE 用例全部 PASS）；5e 的 NPU 参考自检按"算子 + 量级界（1e-6）"
  登记放行，超界或换算子仍按新发现报出。
- 证据：robustness-findings 第六/七轮附录；verify-records …-1014-224625

### ERR-008 三个原语算子的 f16-XL 用例编译失败（疑统一缓冲区容量同类）

- 日期：2026-10-14
- 处置：ANNOTATE
- 受影响版本：v0.5-cov20-frozen（草稿期）
- 受影响用例：softmax-f16-X16 / rms_norm-f16-X32 / swi_glu-f16-X16
  （demo 提交的示范内核；非语料缺陷）
- 内容：示范内核在三个原语算子的 f16 超大档用例上编译失败，怀疑与
  融合轮同源的统一缓冲区预算问题。属 demo 维护项，按算子锚点机制
  只影响各自算子得分。登记为下一阶段维护项。
- 证据：robustness-findings 第五轮附录；1014-093413 测速归档
