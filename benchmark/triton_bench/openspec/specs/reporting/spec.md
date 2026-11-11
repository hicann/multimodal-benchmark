# reporting Specification

## Purpose

生成标准化评测报告，披露分数、风险信号与可追溯信息，并显式声明结果的解释边界。不得出现跨赛道合成的单一总分。

## Requirements

### Requirement: Report generated from run
系统 SHALL 能依据一次运行的标识生成本地报告文件。

#### Scenario: Generate report
- **WHEN** 用户指定一次已完成的运行请求生成报告
- **THEN** 系统输出一份包含该运行全部关键结果的报告文件

#### Scenario: Invalid run reported
- **WHEN** 请求为一次被判定无效的运行生成报告
- **THEN** 报告 MUST 显著标注该运行无效并展示原因

### Requirement: Mandatory report sections
报告 SHALL 至少包含元信息、环境声明、被测对象、功能结果、性能结果、稳定性结果与可追溯附录。

#### Scenario: Sections present
- **WHEN** 生成一份报告
- **THEN** 报告 MUST 包含规范版本、测试集版本、评分器版本、环境指纹、被测编译器标识与结果索引

### Requirement: Silent errors highlighted
报告 SHALL 将静默错误单独成节并高亮，与之并列展示产生静默错误的用例清单。

#### Scenario: Silent error section
- **WHEN** 一次运行中存在静默错误用例
- **THEN** 报告 MUST 单独列出这些用例并标注为最高优先级风险

### Requirement: Unverified performance disclosure
当运行环境未锁定时钟频率时，报告 SHALL 对其中的性能数据标注未验证且声明不得用于排名。

#### Scenario: Unlocked clock disclosure
- **WHEN** 报告对应的运行记录锁频状态为未锁定
- **THEN** 报告中全部性能数据 MUST 标注未验证

### Requirement: Interpretation boundary statement
报告 SHALL 包含固定条文的解释边界声明，明确结果仅对特定软硬件、编译器版本与测试集版本成立。

#### Scenario: Boundary stated
- **WHEN** 生成任意一份报告
- **THEN** 报告 MUST 包含解释边界声明，且声明中 MUST 说明不做跨硬件厂商的单一总分排名

### Requirement: Recompute without rerun
评分与报告 SHALL 能在无需重新执行评测的前提下，依据已存的序列化结果重新计算。

#### Scenario: Rescore then re-report
- **WHEN** 评分逻辑更新后对历史运行重算
- **THEN** 系统 MUST 能基于新的评分结果重新生成报告，且不触发任何重新执行
