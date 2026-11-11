# v0.13 真机定向验证与三轮证据叠加

## 本轮原始结果

- 记录：`20261026-084513-531810`；结果分支提交：`1dd1258c`。
- 真机执行代码：`9196aa0113a77784e172adfdca6547f4520df0c2`。
- manifest：`a5cab486bef01773fddb5fee591516121746bce3af96a6d72fcaa800ba303ec3`。
- CrossEntropyLoss 2/2、MaskedSoftmax 5/5、Matmul 20/20、Neg 20/20，合计 47/47 PASS。
- 四个算子的 CPU/NPU 参考门禁均通过；CrossEntropyLoss 两例的最大绝对误差、最大相对误差均为 0。
- 本机使用当前审计器独立重算，结果与真机保存的 `coverage-validation.json` 完全一致；291 个索引证据文件通过哈希校验。
- 本轮仅运行四个算子，独立报告编译与执行标签均为 41/71。这是定向运行的全局分母统计，不是整个系统的覆盖率下降。

## 逐分区叠加结果

按用户授权，以历史全量为基线，用后续成功定向结果整体替换对应算子分区，不叠加百分比、不重复累计标签：

1. `20261025-042132-445431`：历史全量。
2. `20261026-053611-263422`：替换 FloorDiv、GroupedMatmul。
3. `20261026-084513-531810`：替换 CrossEntropyLoss、MaskedSoftmax、Matmul、Neg。

三轮的 1,377 个用例内容哈希、71 个标签分母、输入生成、参考函数和容差资产一致。第二轮到第三轮仅 CrossEntropyLoss 路由变化；原有共享内核文件的 AST 变化仅为已测试的 `neg` 和新增 `_witness_neg3`，其余算子路由与既有内核保持不变。

| 指标 | 覆盖 | 比例 |
| --- | --- | --- |
| 当前语料静态标签 | 67/71 | 94.37% |
| 三轮编译证据叠加 | 65/71 | 91.55% |
| 三轮执行认证证据叠加 | 65/71 | 91.55% |

叠加后 1,377/1,377 用例均有 PASS 结果，无缺失或失败分区。这不是声称第三轮重新执行了全量 1,377 例。

相对前两轮的 62/71，本轮新增确认：

- `FA-PROG-GRID3D`：`official-neg-float16-23-normal-s2`，真实三维调度与地址依赖，运行通过。
- `FA-FUSE-INTERM-ELIDE`：MaskedSoftmax 五例均获证；此前历史产物离线重审已识别，本轮再次获得实际运行证据。
- `FA-STABLE-LOGSUMEXP`：CrossEntropyLoss 两例参考门禁与提交验证均通过。

## 剩余缺口

静态有标签而编译、执行认证仍缺证的只有两项：

- `FA-PAT-VEC-LOAD`：Neg 正确性已通过，但未获得可认证的后端向量加载或显式 GM→UB 活跃访存链。
- `FA-PAT-TILE-GEMM`：Matmul 正确性已通过，但未获得两个矩阵输入经过明确片上存储层级进入矩阵计算并输出的活跃链。

本轮确实捕获到了编译 stderr：Neg 16 份、Matmul 14 份。两者分别包含 592、546 个 `IR Dump Before` 段，没有 `IR Dump After` 段，也没有明确的 `hivm.hir.load/mmad/store` 操作。当前审计器只接受完整、同名内核的 After-pass module；本轮因此没有获得可用的新增后端 pass 证据。普通 `memref.copy`、默认地址空间的局部申请或 `linalg.matmul` 不能替代要求的底层访存证明，不据此补分。

另有四项当前静态也未覆盖：`FA-PAT-MULTIPASS-REDUCE`、`FA-PAT-SCAN-BLOCKED`、`FA-PAT-SPLIT-K`、`FA-PAT-VEC-BROADCAST`。这些与前述两项合计为编译、执行覆盖剩余的六个标签，分母不删除任何一项。

## 结论边界

三轮叠加的两项比例均已超过 90%，应明确报告达标，不再称上述三项标签“尚未验证”。这仍是历史分区证据叠加，不是单版本、单锁定环境的全量冻结验收。

硬件、编译器、运行库版本字段一致，但三轮 `venv_lock_hash` 分别为 `655acad1cfcd9f66`、`de4ed45453adafd8`、`74db596056db7306`。报告保留该差异；`verified_npu_coverage_target_met` 与 `taskbook_accepted` 不因历史叠加而改写为真。特性清单仍为 `AWAITING_JOINT_CONFIRMATION`，smoke 正确性运行也不等同于性能验收。

本次本地复核产物：

- 独立审计：`/private/tmp/triton-v13-independent-audit.json`。
- 含逐标签用例来源的三轮叠加：`/private/tmp/triton-v13-three-round-overlay.json`。
- 原始本轮日志：`/private/tmp/triton-v13-npu-results-ZajISc/records/20261026-084513-531810`。
