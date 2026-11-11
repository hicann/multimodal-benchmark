# Spec Delta

## Purpose

定义算子的机器可读规格，作为用例生成、提交校验与结果解释的共同依据，使算子语义不依赖具体实现代码即可被判定。

## ADDED Requirements

### Requirement: Operator declares supported data types
算子规格 SHALL 声明每个输入与输出支持的数据类型集合，系统 MUST 拒绝生成规格外类型的用例。

#### Scenario: Generate case within declared types
- **WHEN** 为一个算子生成用例
- **THEN** 所选数据类型 MUST 来自该算子声明的支持集合

#### Scenario: Out-of-range dtype rejected
- **WHEN** 一个用例声明了该算子不支持的数据类型
- **THEN** 系统 MUST 拒绝该用例

### Requirement: Operator declares shape constraints
算子规格 SHALL 声明输入输出的形状约束，包括支持的维度范围与维度间关系。

#### Scenario: Shape within constraints
- **WHEN** 生成用例的张量形状满足算子声明的约束
- **THEN** 该用例 MUST 被接受

#### Scenario: Shape violates constraints
- **WHEN** 用例形状超出算子声明的维度范围或违反维度关系
- **THEN** 系统 MUST 拒绝该用例

### Requirement: Operator declares attributes
算子规格 SHALL 声明其属性（名称、类型、默认值、是否必填），系统 MUST 允许用例在声明范围内指定属性值。

#### Scenario: Attribute with default
- **WHEN** 用例未指定某属性且该属性有默认值
- **THEN** 系统 MUST 使用默认值执行

#### Scenario: Required attribute missing
- **WHEN** 用例未指定某必填属性
- **THEN** 系统 MUST 拒绝该用例并指明缺失的属性名

#### Scenario: Attribute type mismatch
- **WHEN** 用例指定的属性值类型与规格声明不符
- **THEN** 系统 MUST 拒绝该用例

### Requirement: Operator has a reference implementation
每个算子 SHALL 附带一份参考实现，该实现 MUST 独立于任何被测提交，且 MUST 可在无加速器环境下执行以支撑交叉校验。

#### Scenario: Reference executes independently
- **WHEN** 在不含加速器的环境中执行参考实现
- **THEN** 它 MUST 能产出正确的输出张量

#### Scenario: Reference is not the submitted implementation
- **WHEN** 载入一个算子的参考实现
- **THEN** 它 MUST NOT 来自被测提交的目录

### Requirement: Cases derive from spec not from implementation
测试用例 SHALL 由算子规格派生，MUST NOT 依赖于任何具体实现代码。

#### Scenario: Cases exist without any submission
- **WHEN** 尚未有任何人提交某算子的实现
- **THEN** 该算子的用例 MUST 仍可被生成与校验
