# result-storage Specification

## Purpose

持久化评测结果数据，保证真相源可迁移、数据库可重建、且每次运行的用例结果唯一。序列化结果文件是真相源，数据库是从其重建的查询层。

## Requirements

### Requirement: Result record lineage
每条用例结果 SHALL 归属于唯一的一次运行与唯一的用例，且 MUST 在该次运行中唯一。

#### Scenario: Duplicate result rejected
- **WHEN** 尝试为同一次运行中的同一个用例写入第二条结果
- **THEN** 系统 MUST 拒绝该写入

### Requirement: Serialized result as source of truth
每次运行 SHALL 产出一份序列化的结果文件作为真相源，数据库 MUST 可由该文件重建。

#### Scenario: Rebuild from serialized file
- **WHEN** 数据库丢失或损坏，且结果文件仍在
- **THEN** 系统 MUST 能从结果文件完整重建该次运行的全部记录

### Requirement: Environment fingerprint persisted
每次运行 SHALL 持久化其完整环境信息与指纹值。

#### Scenario: Persist environment
- **WHEN** 一次运行开始
- **THEN** 运行记录 MUST 包含完整环境信息与指纹值

### Requirement: Run status and invalidation
每次运行 SHALL 具有明确状态；被判定为无效时 MUST 记录原因且不产出可用分数。

#### Scenario: Invalid run recorded
- **WHEN** 一次运行因健康探针失败或内容校验失败被判定无效
- **THEN** 运行记录 MUST 包含具体原因，且不得产出分数

### Requirement: Runtime data excluded from version control
运行产生的数据目录 MUST NOT 被纳入版本控制。

#### Scenario: Runs directory ignored
- **WHEN** 检查版本控制忽略规则
- **THEN** 运行数据目录 MUST 被明确排除

### Requirement: Submission identity
被测编译器 SHALL 以名称、版本与提交号唯一标识。

#### Scenario: Duplicate submission rejected
- **WHEN** 尝试登记一个名称、版本与提交号均相同的被测编译器
- **THEN** 系统 MUST 拒绝重复登记
