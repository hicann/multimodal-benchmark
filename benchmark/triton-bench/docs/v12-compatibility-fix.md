# v0.12：重新设计 FloorDiv / GroupedMatmul 修复

后续说明：用户于 2026-10-26 明确允许将本轮两个算子的结果与历史全量结果进行叠加分析。历史版仍不作为新运行或冻结验收依据；可追溯的跨轮分析另见 `docs/v13-evidence-gaps.md`，不得将其冒充单版本、单锁定环境的完整运行。

## 状态与废止范围

按用户要求，2026-10-25 的 `291cc5c` 和 2026-10-26 的 `3c186c8` 修复版废止，不作为后续运行或验收依据；旧语料、样例、原始日志仅留档，不删除或重冻。默认生成、检查和覆盖运行入口改用 `v0.12-compat-candidates` / `coverage-v12-ascend-submission`，不得从旧结果继承通过状态、编译证据或执行覆盖分数。

本次只更换 FloorDiv 和 GroupedMatmul 的提交实现，不修改输入、参考实现、容差、标签或评分要求。原始 68×20 用例全部保留；整体仍为 74 算子、1,377 用例，固定 71 项静态标签分母不变。旧版与新版所有 case 编号及内容哈希相同。新版冻结独立框架资产指纹，旧版不能由当前框架重新认证。

## 基于真实日志的重新设计

### FloorDiv：不再经过无符号长除法

旧实现的无符号位运算触发后端不支持的 `uint32_t_to_uint64_t_rintmode` 转换。新实现位于独立模板 `scripts/templates/compatibility_kernels.py`，不再调用旧 elementwise 分支：

- 数据运算只使用有符号 int32，不使用 uint32/uint64、浮点近似商或设备整数除法。
- 对负被除数先处理 `abs(x)-1`，从而使二进制输入保持在 `[0, INT_MAX]`，避免直接求 `abs(INT_MIN)`。
- 以非正的余数和商进行 32 步恢复除法，再补回负被除数少算的 1。余数和商的有效值均可用 int32 表示。
- 按输入符号恢复商；异号且余数非零时再减 1，实现向下取整而非向零截断。
- 除零、`INT_MIN / -1` 仍遵守既有合法域排除，不改任务边界。末尾 mask 的无效分母为 1。
- 两个输入分别计算统一线性 stride，支持冻结用例中的非连续输入，不将输入拷贝为连续张量来逃避该测试。

### GroupedMatmul：组列表只在主机调度，不传入 JIT

旧实现将 tuple 作为 constexpr 传入，内核中 `GROUPS[g]` 被当前 Triton 前端拒绝。新实现将每组的行数、起始行、组号作为独立标量元数据传给通用内核，每组一次启动：

- 主机只做输出分配和属性/shape 元数据调度，不读取输入数值、不调用 Torch matmul。
- 每个内核以实际行/列 tile 构造二维 grid；K 轴采用 `tl.static_range` 的 16 元素分块累加，仍有实际静态展开与 `tl.dot` 运算。
- 通过组号选择 weight/bias，通过前缀行偏移选择 x/output；尾行、尾列和尾 K 均有 mask。
- 组数、分组顺序不硬编码成两个现有用例；CPU 模拟额外验证单组、更多组、跨 tile 和非整块维度。
- 多次内核启动的时间应全部计入性能测量；这是兼容性方案，不声称性能更优。

## 真机只跑两个算子

```bash
cd "$HOME/triton-eval-v07" &&
bash scripts/coverage-regression-verify.sh
```

这个入口不接受额外参数，固定选择 `FloorDiv,GroupedMatmul`：20 个 FloorDiv 和 2 个 GroupedMatmul 冻结用例。不会运行 CrossEntropyLoss 或其他算子的参考/评测分区，也不启动额外的 Gather 能力探针。完整提交的 74 入口合法性检查和基础设备健康探针仍保留；入口导入检查不是执行其他算子的测试。

两组用例分别经过 CPU/NPU 独立参考门禁、真实加载、编译、数值校验及既有 smoke 测量；仍使用全新轮次编译缓存并上传原始证据。禁止把旧轮次结果拼接进本轮。

## 定向回归与全量覆盖分开报告

- `suite-summary.json` 标记 `validation_scope=operator_regression`，预期 22 例；其中 `complete` / `all_pass` 只描述选中分区。
- `coverage-validation.json` 新增 `selected_expected_cases` 和 `selected_operator_validation_passed`。后者仍要求选中用例全 PASS、独立参考通过、编译证据与运行身份完整可核验。
- 定向审计仍以全体固定 71 项计算编译/执行覆盖；未运行的其他算子没有覆盖信用，不缩分母。未选择的分区不误报为本次回归执行失败。
- 全量 `expected_cases=1377`、`complete`、`all_pass` 和 `verified_npu_coverage_target_met` 保持全量口径。只跑 22 例不能使全量验证通过。
- 定向回归的退出码依据选中用例及证据的验证结果，不因未执行全量而必然返回失败。全量入口依旧严格执行原有覆盖目标门禁；并未降低 90% 要求。

## 当前验证边界

本机验证可以证明用例保留、静态覆盖、源码入口、算法及索引的 CPU 模拟一致性，不能证明当前 Ascend 后端一定编译成功。新版真机状态在获取本版结果前均为 NOT_RUN，性能收益未建立，任务清单仍为 AWAITING_JOINT_CONFIRMATION。
