# case-execution Specification

## Purpose

执行单个评测用例并把观测量判定为唯一终态，产出一条可追溯、语义无歧义的结果记录。终态集合互斥且完备，静默错误必须被单独识别为最高优先级风险信号。

## Requirements

### Requirement: Terminal state machine
每个用例的执行结果 SHALL 归入一组互斥且完备的终态之一；这些终态 MUST 覆盖规范错误、编译失败、运行失败、通过、静默错误、形状不符、参考失效与环境中断。

#### Scenario: Compile failure
- **WHEN** 被测编译器无法编译该用例的 kernel
- **THEN** 结果终态为编译失败

#### Scenario: Silent wrong result
- **WHEN** kernel 编译通过且运行通过，但输出与参考实现的偏差超出容差
- **THEN** 结果终态为静默错误，并被标记为最高优先级风险信号

#### Scenario: Reference implementation fails
- **WHEN** 参考实现自身执行失败或与交叉校验参考不一致
- **THEN** 结果终态为参考失效，且该用例 MUST NOT 计入评分分母

### Requirement: Single oracle with cross check
用例的正确性判定 SHALL 仅由主参考实现决定；系统 MUST 另外使用独立的第二参考实现做交叉校验，交叉校验不一致时判定为参考失效。

#### Scenario: Oracle agrees
- **WHEN** 主参考实现输出与被测输出偏差在容差内，且第二参考与主参考一致
- **THEN** 该用例判定为通过

#### Scenario: Cross check disagrees
- **WHEN** 第二参考实现与主参考实现输出不一致
- **THEN** 该用例判定为参考失效，不计入分母并触发告警

### Requirement: Dtype-based tolerance
系统 SHALL 依据用例的目标数据类型查表确定相对与绝对容差；整型数据类型 MUST 要求精确相等。

#### Scenario: Integer exact match
- **WHEN** 用例目标类型为整型
- **THEN** 输出 MUST 与参考实现完全一致，任何偏差均判定为错误

#### Scenario: Low precision tolerance
- **WHEN** 用例目标类型为半精度或脑浮点
- **THEN** 按对应类型的容差表判定，且参考实现 MUST 以更高精度计算后再转换回目标类型

### Requirement: No case-level tolerance override
系统 MUST NOT 允许单个用例覆盖按类型确定的容差。

#### Scenario: Attempt to override tolerance
- **WHEN** 一份用例定义中声明了自定义的容差数值
- **THEN** 系统 MUST 拒绝该覆盖并按类型容差表执行

### Requirement: Failure evidence retention
任何非通过的用例 SHALL 保留可供归因的失败线索与日志路径。

#### Scenario: Failure recorded
- **WHEN** 一个用例编译或运行失败
- **THEN** 结果记录 MUST 包含被截断的异常摘要与日志文件路径

### Requirement: Per-case timeout
每个用例 SHALL 有硬超时上限，超时 MUST 判定为失败而非悬挂。

#### Scenario: Case exceeds timeout
- **WHEN** 一个用例的执行时间超过其超时上限
- **THEN** 该用例被终止并判定为失败
