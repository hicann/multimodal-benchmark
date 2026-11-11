# performance-measurement Specification

## Purpose

以统一口径测量 kernel 执行耗时并给出稳定性判定，使性能结论可被复核而非依赖单点读数。原始样本必须完整留存，不稳定测量必须被标记。

## Requirements

### Requirement: Benchmark invocation
性能测量 SHALL 使用统一的计时封装，并 MUST 显式指定返回中位数而非均值。

#### Scenario: Median is used
- **WHEN** 对一个 kernel 执行计时
- **THEN** 系统 MUST 取多次重复的中位数作为该次测量的耗时值

### Requirement: Raw sample retention
每次性能测量 SHALL 保留全部原始计时样本，而不仅是汇总值。

#### Scenario: Samples stored
- **WHEN** 一次测量完成
- **THEN** 结果记录 MUST 包含完整的原始样本数组

### Requirement: Same-session baseline interleaving
性能基线 SHALL 在同一运行会话内与被测对象交错测量，系统 MUST NOT 使用历史存档值作为基线。

#### Scenario: Baseline measured in-session
- **WHEN** 计算一个用例的加速比
- **THEN** 基线的耗时 MUST 来自同一次运行会话内的测量

### Requirement: Stability gate
系统 SHALL 依据变异系数判定测量是否稳定，超过阈值的测量 MUST 被标记为不稳定。

#### Scenario: Stable measurement
- **WHEN** 一次测量的变异系数未超过当前档位阈值
- **THEN** 该测量被接受用于聚合

#### Scenario: Unstable measurement
- **WHEN** 一次测量的变异系数超过阈值
- **THEN** 该测量被标记为不稳定，并计入不稳定比例

### Requirement: Unlocked clock disclosure
当执行环境无法锁定时钟频率时，产出的性能数据 SHALL 被标注为未验证且不得用于排名。

#### Scenario: Unlocked environment
- **WHEN** 一次运行的环境指纹记录锁频状态为未锁定
- **THEN** 该轮产出的全部性能数据 MUST 标注为未验证
