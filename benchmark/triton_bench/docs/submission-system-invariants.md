<!-- Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. -->
<!-- Licensed under CANN Open Software License Agreement Version 2.0. See repository LICENSE. -->

# 提交评测链路：系统不变量与回归边界

## 1. 共享源码不等于独立可执行提交

维护目录中 `from scripts.templates...` 等引用属于**构建期包含**，不能出现在用户 ZIP 的运行期依赖中。
源码 AST 哈希一致只能证明物化遵循登记内容，不能证明登记内容本身依赖完整。

因此物化完整样例后、官方样例生成后以及联调基线打包前，都检查 Python 符号表中的未绑定全局读取。
检查递归覆盖普通函数、嵌套函数、类与推导式作用域，不导入模块、不求值默认参数、不运行入口。
辅助函数必须显式声明并随 ZIP 内联；本轮补齐 `validate_elementwise_layout`，而非由服务进程注入它。

此检查仅针对仓库维护的完整生成源码，不是用户提交的安全沙箱，也不替代 `validate_static`、
`validate_load`、逐用例执行和设备检查。故意缺符号的编译负例在基线检查**之后**生成，必须继续触发
真实的 COMPILE_FAIL，不能被改成提交预检失败。模板片段尚待生成器补齐的名称（例如 FAMILIES）
在完整产物上检查，不对片段误判。

符号表检查是保守的结构检查：不证明条件分支中的绑定一定发生，不解析任意包的传递依赖，
也不支持通过 `globals()` 动态注入名字的生成设计；运行期导入失败仍由加载契约处理。

## 2. 参数个数门禁不能牺牲编译期类型

内核把相关元数据封装为 `configuration: tl.constexpr`，避免超多参数，同时保留编译证据的元组绑定。
但普通 Python 元组解包赋值不能保证各成员在目标 Triton 编译器中保留 constexpr 类型。
真机验证发现 constexpr 包装对象不能直接下标访问。本轮统一使用
`member: tl.constexpr = configuration.value[index]`，覆盖 GroupedMatmul、模式分组矩阵乘、
elementwise 及其辅助分支、MaskedSoftmax 和 RoPE 的同类路径。不把元组重新展开成长参数列表。

编译证据分析同时兼容历史解包和新的显式成员读取；仍须先通过完整受审内核 AST 身份核验，
再按顺序与类型核验元组成员。不得接受未审源码、运行期元组冒充常量、重复/跳序索引或缺失成员。
原始捕获数据与历史受审源码登记均不重写。

显式 `.value` 在编译器前端读取元组载荷，再以成员注解保留 constexpr，不在运行期求值，
也不修改已捕获元组的 ABI。覆盖 13 个内核中的 52 处读取，包括内核入口的普通 Python
元组与 JIT 辅助调用的 constexpr 成员元组。本地使用 3.2.1 的实际前端方法验证提取，
并保留安装编译器的可选前端回归；两者均不等同于完整 JIT 编译或 NPU 执行。
`TRITON_INTERPRET=1` 的非包装常量解释模式不在该 Ascend JIT 契约内，不据此声称兼容。

## 3. 输出结构错误有稳定的诊断层次

执行顺序为：输入未被修改及防篡改 → 输出类型/容器 → 设备 → 输出个数 → 布局 → shape/dtype/数值。
其中 dtype、shape 和数值比较只在输出个数一致后进行；设备错误仍优先拒绝为 SUBMISSION_INVALID。

- 输出个数错误：由比较器诊断 SHAPE_MISMATCH，不因张量恰好是 strided 而变成另一个状态。
- 输出个数正确但非布局类算子返回非连续输出：仍拒绝 SUB-EXE-002。
- 伪张量、空输出、带非张量元数据或不受归一化器支持的嵌套容器：拒绝 SUB-EXE-001。
- 错误设备：拒绝 SUB-EXE-006；不能通过额外输出掩盖。
- 原独立 `check_output` 调用不提供个数时保留其递归 Tensor 检查能力；实际评测传入参考输出个数，
  与执行器的扁平输出协议保持一致。

不修改全局判定器的终态优先级，不放宽数值容差，不放宽负例要求，不把 SHAPE_MISMATCH 算为 PASS。
RoPE 启动处额外修正遗漏的 block 初始化和误接到异常实例上的表达式；这是依赖检查发现的同类问题。

编译证据捕获不得改变功能终态：无法将返回值归一化时，捕获器只记录诊断，仍交由类型契约判定。
`EXEC-NONTENSOR` 因而明确要求 SUBMISSION_INVALID 与 SUB-EXE-001，而不再接受捕获器
提前抛出 TypeError 所形成的 RUNTIME_FAIL。这是纠正阶段归因，并非允许非法输出通过。

## 4. 资产和验证

新修订：`submission-constexpr-20261110-payload-score`，前任为
`submission-system-20261110-dependencies-constexpr-output`。
14 套冻结资产保留逐字一致的前任索引和历史元数据，case manifest、算子集合、用例身份与评分权重不变。
新修订的硬件认证初始为 NOT_RUN；旧证据不能自动证明新源码的 NPU 行为。

本地回归：`tests/core/test_submission_system.py` 加相关原有测试；本地 CPU 检查不是 NPU 数值认证。
真机继续使用现有 `scripts/codecheck-regression-verify.sh`，只跑 independent/normal HTTP ZIP 全链路，
不增加全量 CPU 或额外直接算子测试阶段。70 个场景必须全部观察到、零不匹配，并核对得分及下载报告。
Code Check 与开源检测必须对新提交重新运行；本地规则自检不能替代服务端 CI。


## 5. 独立复算遵守公开分数精度

评分器的功能分与严格通过率按四位小数输出。联调校验器保持独立聚合与权重计算，
仅在最终比较前对重算值做同样的 `round(expected, 4)`；绝对容差仍是 1e-8，
不使用默认相对容差，不把非有限数值当作有效分数。三分之一等得分不再误报。

不修改评分器公式、状态计分范围、算子权重或数值判定容差。最后一位公开小数错误、
权重不符、计分项遗漏、严格通过率错误以及应满分的独立实现不满分，仍必须被校验拒绝。
无有效计分项时得分应为 None，而不是数值 0。场景预期和 20/50 的默认真机范围不变。
