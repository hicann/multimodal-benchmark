# Spec Delta

## Purpose

使用例执行面向「提交方提供的实现」而非评测方自带的 kernel，并扩展终态以覆盖提交类失败。

## MODIFIED Requirements

### Requirement: Terminal state machine
每个用例的执行结果 SHALL 归入一组互斥且完备的终态之一；这些终态 MUST 覆盖规范错误、编译失败、运行失败、通过、静默错误、形状不符、参考失效、环境中断、以及提交不合规。

#### Scenario: Compile failure
- **WHEN** 被测编译器无法编译该用例的 kernel
- **THEN** 结果终态为编译失败

#### Scenario: Silent wrong result
- **WHEN** 编译通过且运行通过，但输出与参考实现的偏差超出容差
- **THEN** 结果终态为静默错误，并被标记为最高优先级风险信号

#### Scenario: Reference implementation fails
- **WHEN** 参考实现自身执行失败或与交叉校验参考不一致
- **THEN** 结果终态为参考失效，且该用例 MUST NOT 计入评分分母

#### Scenario: Submission rejected before execution
- **WHEN** 提交的实现在静态或加载阶段被判定不合规
- **THEN** 结果终态为提交不合规，且 MUST NOT 计入评分分母

#### Scenario: Submission detected as cheating
- **WHEN** 提交的实现在运行期被检测为未真实实现目标算子
- **THEN** 结果终态为提交不合规，且 MUST 被标记为风险信号


## ADDED Requirements

### Requirement: Case execution requires a submission-supplied implementation
每个评测用例 SHALL 由一份声明式定义描述，其输入数据 MUST 由算子规格与随机种子派生；被测实现 MUST 来自本次评测所载入的提交，用例 MUST NOT 指定其来源位置。

#### Scenario: Kernel loaded from real file
- **WHEN** 加载被测实现
- **THEN** 系统 MUST 从提交目录下的真实源文件加载，不得从内联字符串或数据库字段加载

#### Scenario: Kernel not from inline code
- **WHEN** 被测实现仅以字符串或数据库字段形式存在
- **THEN** 系统 MUST 拒绝加载

#### Scenario: Case carries data not implementation
- **WHEN** 载入一个评测用例
- **THEN** 该用例 MUST 只描述输入数据、属性与判定参数，不得绑定被测实现的路径

#### Scenario: Implementation supplied by submission
- **WHEN** 执行一个用例
- **THEN** 被测实现 MUST 来自本次评测所载入的提交

#### Scenario: Case without submission
- **WHEN** 某用例所属算子在提交中不存在对应实现
- **THEN** 该用例 MUST 被判定为提交不合规，而非编译失败

### Requirement: Case binds to operator not to kernel
用例 SHALL 绑定到一个算子标识，MUST NOT 绑定到具体的 kernel 文件或入口函数。

#### Scenario: Same case runs against different submissions
- **WHEN** 同一用例在不同提交下被执行
- **THEN** 除被测实现外，用例的输入与判定参数 MUST 完全相同

### Requirement: Uniform invocation convention for implementation and reference
被测实现与参考实现 SHALL 按同一约定被调用：输入张量按其在输入列表中的顺序作为位置参数，算子属性以关键字参数传入。

参考实现 MUST NOT 复用提交中的任何代码，输入构造 MUST NOT 来自提交目录。

#### Scenario: Attributes passed as keywords
- **WHEN** 调用被测实现或参考实现
- **THEN** 输入张量 MUST 按序作为位置参数，属性 MUST 以关键字传入，使得属性在签名中的声明位置不影响调用

#### Scenario: Multi-input operator receives all inputs
- **WHEN** 执行一个多输入算子的用例
- **THEN** 实现 MUST 收到该算子构造过程产出的全部输入张量

#### Scenario: Attribute default supplied by operator spec
- **WHEN** 用例未给出某个可选属性
- **THEN** 系统 MUST 用算子规格声明的默认值补齐后传入

### Requirement: Non-compliant submission excluded from scoring
被判定为提交不合规的用例 MUST NOT 计入任何评分分母，但 MUST 被记录并披露。

#### Scenario: Excluded from coverage
- **WHEN** 计算功能主得分
- **THEN** 终态为提交不合规的用例 MUST 不参与计算

#### Scenario: Disclosed in report
- **WHEN** 生成评测报告
- **THEN** 提交不合规的算子清单 MUST 被单独披露
