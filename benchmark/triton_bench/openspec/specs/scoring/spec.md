# scoring Specification

## Purpose

把用例级结果聚合为运行级分数，使分数可解释、可重算，且不产生误导性的单一总分。性能归一化在锚点不满足条件时必须声明不可用而非补零。

## Requirements

### Requirement: Weighted operator coverage score
系统 SHALL 以算子的加权变体通过率作为功能完备性的主得分，权重取自冻结测试集中的算子权重。

#### Scenario: Compute coverage score
- **WHEN** 一次运行的所有用例结果已确定
- **THEN** 系统按算子权重对通过率加权平均，产出百分制的功能主得分

#### Scenario: Non-counted cases excluded
- **WHEN** 某些用例被判定为不计入分母的终态
- **THEN** 这些用例 MUST 不参与功能主得分的计算

### Requirement: Strict operator pass rate disclosed
系统 SHALL 额外披露加权严格通过率，即算子的全部变体均通过的比例。

#### Scenario: Divergence flagged
- **WHEN** 严格通过率显著低于主得分
- **THEN** 报告 MUST 提示被测编译器存在边界敏感问题

### Requirement: Performance score anchored to first run
性能得分 SHALL 以首轮基线运行的几何平均加速比为锚点并定义为满分；后续运行的得分 MUST 相对该锚点计算，并 MUST 显式记录锚点来源。

#### Scenario: First run sets anchor
- **WHEN** 执行首个基线的评测运行
- **THEN** 该轮性能得分记为满分，其几何平均加速比被冻结为锚点

#### Scenario: Later run compared to anchor
- **WHEN** 执行一次后续评测运行
- **THEN** 性能得分按相对首轮锚点的对数加速比比例计算，并记录锚点来源标识

### Requirement: Fast ratio metrics
系统 SHALL 披露在多个加速比阈值下既正确又达标的用例占比。

#### Scenario: Fast ratio computed
- **WHEN** 计算性能聚合指标
- **THEN** 系统 MUST 给出在阈值一、一点二与二下的达标占比

### Requirement: Silent error count disclosed
系统 SHALL 单独统计并披露静默错误的用例数量。

#### Scenario: Silent errors counted
- **WHEN** 一次运行结束
- **THEN** 系统 MUST 统计终态为静默错误的用例数并单独披露

### Requirement: No cross-track total score
系统 MUST NOT 产出跨功能与性能赛道的加权合成总分。

#### Scenario: Report without total
- **WHEN** 生成评测报告
- **THEN** 报告中 MUST NOT 出现跨赛道合成的单一总分，两条赛道分别呈现

### Requirement: Score recomputation
评分 SHALL 支持在评分逻辑版本变化后对历史运行重算，并 MUST 保留每次重算的结果与版本标识。

#### Scenario: Rescore with new scorer version
- **WHEN** 使用新版本评分逻辑对同一次运行重新计算
- **THEN** 系统新增一条以新版本标识的结果记录，且 MUST NOT 覆盖原有记录
