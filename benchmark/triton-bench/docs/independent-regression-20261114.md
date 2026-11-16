# 20 算子独立提交与新用例回归（2026-11-14）

> 已取消：本轮误将“新的算子提交”理解成“新的评测用例”，偏离用户要求。
> 用户要求始终是用既有 v19 冻结用例评测新的算子实现 ZIP。
> 本文仅留存误操作过程，不作为需求完成、平台回归通过或测试集演进的证据。
> 没有修改 v19 冻结资产或 5177 主平台绑定。

## 范围与预先冻结的预期

本轮同时编写新的提交实现和新的服务端用例，避免把重新打包既有示例当成独立回归。
ZIP 仅包含 `manifest.yaml` 与三个 Python 文件，不携带用例、参考答案、评测器、凭据或历史结果。
声明 20 个已登记算子，使用独立入口名称与精确契约参数；核心计算由 Triton 完成，
torch 仅用于设备上的输出分配和元信息读取。源码位于
`examples/independent-20-submission/`，构造器为 `scripts/platform_examples/prepare_independent_regression.py`。

算子为 Abs、Add、ArgMax、Equal、Exp、Gather、LayerNorm、Matmul、Maximum、Mul、Neg、
ReduceMax、ReduceSum、Relu、RmsNorm、Sigmoid、Softmax、Sub、Transpose、Where。
它们覆盖逐元素、双输入、布尔输出、条件选择、归约、索引、布局、归一化和矩阵乘。
这是登记语义的新实现，不是新增未登记的算子种类。

新测试集 `holdout-20-independent-20261114` 每算子 12 例，共 240 例，四档规模各 3 例。
选择同时考虑 dtype、数值分布、布局、属性和标签增益；全部形状不出现在 v19 同算子用例中，
不是只更换 seed。与旧 1486 例的结构签名和 `(operator, shape)` 交集均为零。
输入限制继续遵循已登记且参考实现支持的候选域，不擅自添加越界轴、额外 dtype 或广播语义。

冻结前，240 例均通过 CPU 参考确定性、输入契约与独立第二参考校验；20 个源码入口通过
静态契约检查。它们分别是资产与源码门禁，不当作 NPU 编译或运行证据。
构造过程还遇到父语料没有可选 `official-catalog.json` 的情况，已改为按实际存在性复制；
失败草稿保留，未修改 v19。

执行前固定的清单和预期在 `runs/independent-regression-20261114/prepared/expectations.json`：

- manifest：`96c1a4019d790ab15293fc547f484ee48f1c5b810b1f6d1f6254bd0853c30a18`
- assets：`175a92dcdcadc0151d7d6a3e6b2a3f6f2d03423ff38f73588069d9d60eb59f98`
- 预期：标准档、固定 240 个用例编号、全部 PASS；不根据实际结果改写预期。

## 平台链路与隔离

使用相同 HTTP 服务、同源前端代理、持久 Worker、SSH 传输及 NPU 执行链路，
单独绑定本轮冻结用例和数据库。现有 5177/v19 服务及历史结果保持可用；本轮入口为
`http://127.0.0.1:5178`，API 端口 8773，工作目录为
`runs/independent-regression-20261114/service/`。不同清单不混入既有榜单。

客户端 `scripts/platform_examples/run_independent_regression.py` 通过 HTTP 上传原始 ZIP 并创建 standard 任务，
检查上传与建任务幂等、用例集合、任务和包身份、终态、三轮正有限计时样本、
中位数、CV、参考耗时/提交耗时之比及全部下载产物的大小/SHA。
资源启停继续使用本机 cann_retry HTTP 接口与 VS Code 连接准备，SSH 只传送资产和结果。
任务回收后须确认 `SHUTDOWN` 与云端 stopped；独立 watchdog 继续生效。

首轮任务：`d1a7fd401b7e406b9f17fba0801349dc`，远端 job：
`958d1c8b188e453d80437b3f777e0d5d`。
客户端开发时修正了事件页/产物列表的字段读取，并增加按已记录 task ID 恢复观察；
只重启观察客户端，未重提任务、未重复远端执行。

## 结果

已应用户纠正取消真机任务，不继续此新用例实验；部分执行结果不作为回归通过结论。

## 证据边界

这是新实现和新形状的有限回归，不能证明任意合法实现均可运行，也不是全量 74 算子认证。
若依据首轮结果修复提交，再次运行同一冻结集，后续通过属于修复回归，
不能继续称其为完全未见的盲测；原始失败、包版本与结果必须分别保留。
没有更换评分公式、放宽误差、删用例或静默覆盖冻结资产。


> 2026-11-16 迁移说明：以上真机结果属于原日期与原源码指纹。本轮改用 ba04485 已修复共享源码并调整提交实现；新生成 ZIP 不继承上述真机认证。冻结用例不变，当前迁移复核见 [质量记录](quality/codecheck-recovery-20261116.md)。
