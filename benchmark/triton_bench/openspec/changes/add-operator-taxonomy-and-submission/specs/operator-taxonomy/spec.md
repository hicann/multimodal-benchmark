# Spec Delta

## Purpose

建立算子的分类分级体系，使评测覆盖面可被结构化描述与度量，并为按类别的差异化校验与分组评分提供依据。

## ADDED Requirements

### Requirement: Category is the primary classification axis
系统 SHALL 以 `category`（语义族）作为算子分类的主维度，且 MUST 支持十三种类别：Elementwise、Reduction、Broadcast、MaskPredicate、Normalization、IndexGather、ScatterUpdate、Contraction、LayoutTransform、SortSelect、NumericalStable、FusedComposite、VVFusion。

#### Scenario: Every operator has exactly one category
- **WHEN** 登记一个算子
- **THEN** 该算子 MUST 被归入且仅归入上述十三种类别之一

#### Scenario: Unknown category rejected
- **WHEN** 登记一个算子且声明的类别不在上述集合内
- **THEN** 系统 MUST 拒绝登记并列出合法取值

### Requirement: Level records difficulty independently
系统 SHALL 记录每个算子的 `level`（L1 至 L4 难度等级），该字段 MUST 与 `category` 独立，二者共同构成算子的二维坐标。

#### Scenario: Same category spans multiple levels
- **WHEN** 同一类别下的算子分布在不同难度等级
- **THEN** 系统 MUST 保留各自的等级，不得因类别相同而合并等级

#### Scenario: Same level contains multiple categories
- **WHEN** 同一难度等级下存在多个类别的算子
- **THEN** 系统 MUST 保留各自的类别，不得因等级相同而合并类别

### Requirement: Operator provenance recorded
每个算子 SHALL 记录其来源，包括来源算子名、难度等级、以及在参考仓库中的目录位置。

#### Scenario: Provenance traceable
- **WHEN** 查询一个算子的定义
- **THEN** 系统 MUST 能给出其来源算子名、等级与参考目录

### Requirement: Feasibility annotated for the supported language
系统 SHALL 为每个类别标注其在受支持实现语言下的可行性，且对无可行算子的类别 MUST 保留占位而不静默丢弃。

#### Scenario: Category with no feasible operator
- **WHEN** 某类别在受支持语言下不存在可实现的算子
- **THEN** 系统 MUST 保留该类别定义并标记为暂缓，不得从分类体系中移除

#### Scenario: Operator selection respects feasibility
- **WHEN** 为某类别选择算子用于评测
- **THEN** 系统 MUST 只选择在该类别下受支持语言可实现的算子

### Requirement: Case count bound per category
系统 SHALL 支持为每个类别绑定固定数量的测试用例，且该数量 MUST 可配置。

#### Scenario: Default binding
- **WHEN** 未另行配置
- **THEN** 每个类别 MUST 绑定五个测试用例
