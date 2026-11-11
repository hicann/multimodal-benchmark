# Spec Delta

## Purpose

定义用户提交算子实现的目录契约与发现机制，使符合规范的目录无需构建即可被评测、出分与报错。

## ADDED Requirements

### Requirement: Submission uses directory layout without build step
提交 SHALL 采用目录式结构，系统 MUST 能在不执行任何构建步骤的前提下加载提交的实现。

#### Scenario: Load without building
- **WHEN** 载入一个符合目录契约的提交
- **THEN** 系统 MUST 无需执行打包或编译即可导入其实现

#### Scenario: No build artifact required
- **WHEN** 提交目录中没有打包配置或构建脚本
- **THEN** 该提交 MUST 仍被接受

### Requirement: Submission manifest declares operators
每个提交 SHALL 包含一份清单，声明所含算子的名称、类别、难度等级与入口定义。

#### Scenario: Manifest parsed
- **WHEN** 载入一个提交
- **THEN** 系统 MUST 能从清单中读出每个算子的名称、类别与入口

#### Scenario: Manifest missing
- **WHEN** 提交目录中缺少清单文件
- **THEN** 系统 MUST 拒绝该提交并指明缺少的文件名

#### Scenario: Manifest declares unknown operator
- **WHEN** 清单声明了一个未登记的算子名
- **THEN** 系统 MUST 拒绝该算子的条目并指明可用算子名

### Requirement: Entry function matches declared signature
提交的实现 SHALL 导出与算子规格同名且签名一致的入口函数，系统 MUST 校验参数个数与名称。

#### Scenario: Signature matches
- **WHEN** 提交的入口函数名与参数列表与算子规格一致
- **THEN** 该算子 MUST 被接受

#### Scenario: Entry function absent
- **WHEN** 提交目录中不存在清单声明的入口函数
- **THEN** 系统 MUST 拒绝该算子并指明期望的函数名

#### Scenario: Parameter count mismatch
- **WHEN** 提交的入口函数参数个数与规格不符
- **THEN** 系统 MUST 拒绝该算子并给出期望与实际的参数列表

### Requirement: Multiple operators per submission
一次提交 SHALL 允许包含多个算子，且每个算子的校验结果 MUST 相互独立。

#### Scenario: Partial failure isolates
- **WHEN** 提交中的某个算子校验失败
- **THEN** 其余算子 MUST 仍可被正常评测，失败不得影响其他算子

### Requirement: Submission is immutable during evaluation
评测期间系统 MUST NOT 修改提交目录中的任何文件。

#### Scenario: Read-only evaluation
- **WHEN** 完成一次对某提交的评测
- **THEN** 提交目录的内容 MUST 与评测开始前一致
