# v0.13：跨轮结果叠加与剩余证据缺口

## 已确认的结果，不再称为“尚未验证”

用户明确允许叠加历史全量与本轮定向回归结果。历史版本仍不重新发布、不恢复为当前冻结验收版本。

- 全量记录 `20261025-042132-445431`，代码 `291cc5c`：编译与执行验证均为 60/71，1,353 个用例 PASS。
- 修复记录 `20261026-053611-263422`，代码 `16f5717`：FloorDiv 20/20、GroupedMatmul 2/2 PASS；CPU/NPU 参考门禁均通过。
- 将旧轮这两个算子的结果分区整体替换，而不是把百分比相加：1,375 个用例 PASS，剩下 CrossEntropyLoss 两个用例缺少执行结果。编译和执行标签并集均为 **62/71 = 87.32394366%**。
- GroupedMatmul 本轮新增确认 `FA-CTRL-STATIC-UNROLL`、`FA-LINALG-BIAS`。FloorDiv 全部修复成功，但未增加相对旧全量并集的新标签。

新增 `scripts/analyze_coverage_overlay.py`，校验原始证据哈希、冻结 case 哈希与标签分母、参考/容差资产、提交源码与路由、硬件/编译器版本，输出逐标签来源。旧报告不修改，叠加分析不写入冻结全量验收标志。

环境版本字段一致，但两轮 `venv_lock_hash` 分别为 `655acad1cfcd9f66`、`de4ed45453adafd8`。分析保留此差异，不把两个锁定环境冒充同一完整运行，也不据此否认本轮两个标签已经获得真机证据。

## 仍存在的五个差距与代码尝试

1. **FA-FUSE-INTERM-ELIDE**：修复通用 MLIR 写法 `"tt.reduce"(...) ({...})` 的解析，归约体内局部 SSA 不泄漏到外围依赖图，未闭合 region 拒绝。真实历史 MaskedSoftmax 的五个编译观测重新回放后均能识别融合证据。`--reanalyze` 单独列出这一新增认证：历史证据在新识别器下为 **63/71 = 88.73239437%**；它不是新的设备运行，也不静默重写旧报告。
2. **FA-PROG-GRID3D**：Neg 增加专门的三维调度内核，物理 grid 为 `(ceil(D2/256), D1, D0)`，三个 program ID 参与真实地址计算；支持冻结的统一线性 stride。认证核对真实输入尺寸、stride、grid 和活跃地址依赖。需要新版真机编译与运行。
3. **FA-PAT-VEC-LOAD**：保留原 LLVM/vector 证据路径，增加显式昇腾全局到 UB 块传输的活跃数据流识别。必须有明确 `hivm.hir.load`、UB 分配与到最终全局输出的链路；普通 `memref.copy`、未知地址空间、死代码均不得得分。该规则是待实机匹配的受限后端支持，不预先声称当前编译器一定生成此形式。
4. **FA-PAT-TILE-GEMM**：保留既有共享存储证明；增加显式昇腾 L1→L0A/L0B→矩阵计算→输出的数据流证明候选。必须将两个真实输入分别追溯到片上操作数，不把 `linalg.matmul`、泛化局部分配或无关 copy 当作共享分块证据。架构等价性和最终特性清单仍需共同确认，不宣称性能收益。
5. **FA-STABLE-LOGSUMEXP**：参考函数、输入、独立 CPU oracle、容差和两次逐位确定性门禁完全保留。仅在 CrossEntropyLoss 的 NPU 参考调用期间请求严格确定性算法，完成同步并恢复原模式；不支持或仍有差异时直接失败，不重试挑选“好结果”，不退回 CPU 代替 NPU 门禁。提交侧另以 `(selected-maximum)` 先消除公共大偏置，避免先加回 maximum 再相减的精度损失，数学语义不变。确定性请求是否被当前 NPU 后端兑现，必须实机验证。

## 更完整的编译证据采集

覆盖验证默认在独立子进程请求 `MLIR_ENABLE_DUMP=1`，仅在正确性 JIT 调用范围捕获 native stderr，结束或异常都恢复原文件描述符。保留编译异常时的原始 stderr。产物压缩后纳入已有哈希索引；性能采样不安装此捕获钩子。

仅完整的、同名内核的 After-pass module 可进入证据识别；不同 pass 不拼接为虚构 SSA 图。无 dump、缓存未产生 dump、后端不支持环境开关或输出超过 64 MiB 时均不得凭空补充证据。早期 `ttadapter` 不是最终硬件访存证明。

## 版本、输入与验证范围

- 新语料 `corpus/v0.13-evidence-candidates`、新样例 `examples/coverage-v13-ascend-submission`；不改写任何旧冻结版本。
- 74 算子、1,377 个用例、原 68×20 基线、所有用例内容哈希和 71 标签分母保持不变，静态仍 67/71 = 94.36619718%。不为提升分数新增或删除用例。
- 保持已验证的 FloorDiv/GroupedMatmul 内核逐字节不变；修改 Neg 调度、增加 CrossEntropyLoss 稳定提交路径及证据机制。
- 新定向入口 `bash scripts/coverage-gap-verify.sh`，只运行 Neg、Matmul、MaskedSoftmax、CrossEntropyLoss 四个承载算子，拒绝额外参数，不自动跑全量。
- 此入口共执行 47 个现有用例：Neg 20、Matmul 20、MaskedSoftmax 5、CrossEntropyLoss 2，不重新执行已经通过的 FloorDiv/GroupedMatmul。
- 原 `coverage-regression-verify.sh` 仍只选择 FloorDiv/GroupedMatmul；全量入口不放宽完整性和覆盖门槛。
- 目前特性清单仍为 `AWAITING_JOINT_CONFIRMATION`。达到 64/71 才超过 90%；历史叠加/离线重审、当版实机结果、性能验收必须分别报告。

## 本机检查结果

本机全套回归已通过（900 passed、187 skipped，后续补充的 NPU 项在本机同样跳过）；专项验证包括真实内核源码的 CPU 张量模拟、通用归约 IR 与作用域、后端活跃/死数据流、错误地址空间、原生 stderr 异常恢复、跨轮哈希/环境/分区校验。静态检查与 74 个入口源码契约均通过。CPU 模拟与合成 IR 测试不是新的真机覆盖证据，三维调度和参考确定性策略仍待实机确认。

当前四个仍需补证的标签为 GRID3D、VEC-LOAD、TILE-GEMM、STABLE-LOGSUMEXP。若三维调度通过，按当前历史重审的覆盖口径可达到 64/71；不能在获得真实结果前把这个预期写成已达标。

## 本地复核历史证据

```bash
/private/tmp/tv5/bin/python scripts/analyze_coverage_overlay.py \
  --baseline /private/tmp/triton-v12-npu-results-1sWwnM/records/20261025-042132-445431/evidence \
  --repair /private/tmp/triton-v12-npu-results-1sWwnM/records/20261026-053611-263422/evidence \
  --baseline-submission examples/coverage-v111-ascend-submission \
  --repair-submission examples/coverage-v12-ascend-submission \
  --reanalyze --out /private/tmp/triton-v13-overlay-reanalysis.json
```
