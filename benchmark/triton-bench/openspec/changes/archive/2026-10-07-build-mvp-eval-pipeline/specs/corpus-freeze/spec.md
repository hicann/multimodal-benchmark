# Spec Delta

## Purpose

定义、冻结与校验评测用测试集，保证评测输入可复现、来源可标记、且在冻结后不可被静默篡改。

## ADDED Requirements

### Requirement: Case definition
每个评测用例 SHALL 由一份声明式定义描述，且 MUST 指向文件系统上真实存在的 kernel 源文件与入口函数名。

#### Scenario: Kernel loaded from real file
- **WHEN** 加载一个评测用例
- **THEN** 系统从声明的路径读取真实源文件并定位声明的入口函数

#### Scenario: Kernel not from inline code
- **WHEN** 一个用例的 kernel 仅以字符串或数据库字段形式存在
- **THEN** 该用例 MUST 被判定为非法，不进入执行

### Requirement: Case reproducibility inputs
每个评测用例 SHALL 声明其输入数据的随机种子与固定的启动配置，使同一用例在任何运行中都产生相同输入与相同执行形态。

#### Scenario: Same seed yields same inputs
- **WHEN** 同一用例在不同运行中被执行
- **THEN** 两次生成的输入张量 MUST 完全一致

### Requirement: Case content integrity
系统 SHALL 为每个用例计算内容哈希，并 MUST 在每次运行前后校验该哈希一致。

#### Scenario: Content unchanged
- **WHEN** 一次运行开始与结束时分别计算用例内容哈希
- **THEN** 两次哈希 MUST 相同

#### Scenario: Content tampered
- **WHEN** 冻结后的用例内容在运行期间发生变化导致哈希不一致
- **THEN** 该轮运行 MUST 被标记为无效

### Requirement: Corpus freeze manifest
测试集在被使用前 SHALL 被冻结，并 MUST 生成一份清单，记录版本标识、清单哈希、用例数量与冻结时间。

#### Scenario: Manifest verified at run start
- **WHEN** 一次评测运行启动
- **THEN** 系统校验清单哈希与用例数量与冻结时一致，不一致则拒绝执行

### Requirement: Determinism screening before freeze
用例在被纳入冻结测试集前 SHALL 接受重复执行筛查；重复执行结果不一致的用例 MUST 被剔除并记录排除原因。

#### Scenario: Case is deterministic
- **WHEN** 一个用例连续重复执行多次且结果完全一致
- **THEN** 该用例被纳入冻结测试集

#### Scenario: Case is nondeterministic
- **WHEN** 一个用例在重复执行中出现结果不一致
- **THEN** 该用例 MUST 被剔除并记录到排除列表

### Requirement: Case provenance tagging
每个用例 SHALL 携带来源标记，用于区分其是否来自被测方自带的自测用例。

#### Scenario: Vendor self-test case tagged
- **WHEN** 一个用例来源于被测编译器自带的教程或单元测试
- **THEN** 该用例 MUST 被标记为被测方自测来源

### Requirement: Operator weight
每个算子 SHALL 携带一个权重值用于加权聚合，该权重 MUST 随测试集版本冻结且不随运行变化。

#### Scenario: Equal weight in MVP
- **WHEN** 冻结首个版本的测试集
- **THEN** 所有算子权重取相同值
