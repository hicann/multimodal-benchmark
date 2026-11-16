# Spec Delta

## Purpose

在保持不产生跨赛道合成总分的前提下，新增按算子类别与难度等级的分组聚合与披露。

## ADDED Requirements

### Requirement: Coverage aggregated per category
系统 SHALL 按算子类别聚合功能通过率，并按类别单独披露。

#### Scenario: Per-category pass rate
- **WHEN** 一次运行结束后计算功能得分
- **THEN** 系统 MUST 给出每个类别的加权通过率

#### Scenario: Category with no result
- **WHEN** 某类别下没有任何计入分母的用例
- **THEN** 该类别 MUST 被披露为无数据，不得计为零分

### Requirement: Coverage aggregated per difficulty level
系统 SHALL 按难度等级聚合功能通过率，并按等级单独披露。

#### Scenario: Per-level pass rate
- **WHEN** 一次运行结束后计算功能得分
- **THEN** 系统 MUST 给出每个难度等级的加权通过率

### Requirement: Group aggregation excludes non-counted cases
按类别或等级聚合时，系统 MUST 排除不计入分母的用例，包括参考失效与提交不合规。

#### Scenario: Non-compliant excluded from group
- **WHEN** 某类别下存在提交不合规的用例
- **THEN** 这些用例 MUST 不参与该类别的通过率计算

### Requirement: Cross-category comparison avoided
系统 MUST NOT 对不同类别之间的得分做加权合成或排名。

#### Scenario: No cross-category total
- **WHEN** 生成评分结果
- **THEN** 结果中 MUST NOT 出现跨类别合成的单一分数

### Requirement: Group results recorded with scorer version
按类别与等级聚合的结果 SHALL 随评分器版本一并记录，支持重算时留痕。

#### Scenario: Rescore keeps grouping
- **WHEN** 以新版本评分器重算同一次运行
- **THEN** 按类别与等级的分组结果 MUST 被一并重新计算并新增记录
