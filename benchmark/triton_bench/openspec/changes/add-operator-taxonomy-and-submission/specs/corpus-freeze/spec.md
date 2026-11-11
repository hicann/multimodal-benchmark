# Spec Delta

## Purpose

使冻结的测试集按算子类别分层组织，且用例来源从「绑定实现」变为「由算子规格派生」。

## MODIFIED Requirements

### Requirement: Case definition
每个评测用例 SHALL 声明其所属算子、输入数据规格与判定参数，MUST NOT 声明被测实现的文件路径或入口函数名。

#### Scenario: Case references operator
- **WHEN** 载入一个评测用例
- **THEN** 该用例 MUST 能解析出其所属算子标识

#### Scenario: Kernel path field rejected
- **WHEN** 一个用例声明了被测实现的文件路径
- **THEN** 该用例 MUST 被拒绝

#### Scenario: Kernel loaded from real file
- **WHEN** 被测实现被加载
- **THEN** 它 MUST 来自文件系统上的真实源文件，不得来自内联字符串或数据库字段

#### Scenario: Kernel not from inline code
- **WHEN** 被测实现仅以字符串或数据库字段形式存在
- **THEN** 系统 MUST 拒绝加载

### Requirement: Case provenance tagging
每个用例 SHALL 携带来源标记，用于区分其是否来自被测方自带的自测用例，且 MUST 记录其所属算子的类别与难度等级。

#### Scenario: Vendor self-test case tagged
- **WHEN** 一个用例来源于被测编译器自带的教程或单元测试
- **THEN** 该用例 MUST 被标记为被测方自测来源

#### Scenario: Category recorded on case
- **WHEN** 载入一个评测用例
- **THEN** 该用例 MUST 能给出所属算子的类别与难度等级

## ADDED Requirements

### Requirement: Corpus organized by category
冻结的测试集 SHALL 按算子类别组织，且 MUST 能按类别独立统计用例数量与覆盖情况。

#### Scenario: Count per category
- **WHEN** 查询冻结测试集的构成
- **THEN** 系统 MUST 能给出每个类别的算子数与用例数

#### Scenario: Missing category surfaced
- **WHEN** 某类别下没有任何用例
- **THEN** 系统 MUST 将该类别列出为空缺，不得静默省略

### Requirement: Cases derived from operator spec
用例 SHALL 由算子规格派生，其数据类型、形状与属性 MUST 落在算子规格声明的允许范围内。

#### Scenario: Derived case within spec
- **WHEN** 由算子规格派生用例
- **THEN** 生成的用例 MUST 满足该算子声明的类型、形状与属性约束

#### Scenario: Out-of-spec derivation rejected
- **WHEN** 派生参数超出算子规格声明的范围
- **THEN** 系统 MUST 拒绝生成该用例并指明越界的维度

### Requirement: Case set size configurable per category
每个类别绑定的用例数量 SHALL 可配置，且未配置时 MUST 取默认值。

#### Scenario: Default binding
- **WHEN** 未配置每类用例数
- **THEN** 每个类别 MUST 绑定五个用例

### Requirement: Case freezes operator attributes
用例 SHALL 冻结其算子属性（如归约维、置换、分组表），必填属性 MUST 在用例中给出且落在算子规格声明的取值域内；未给出的可选属性由算子规格默认值补齐。

派生出的属性 MUST 随用例一起冻结，不得只存在于派生过程中。

#### Scenario: Required attribute carried into execution
- **WHEN** 执行一个带必填属性的用例
- **THEN** 该属性 MUST 被传入参考实现与被测实现

#### Scenario: Attribute outside declared domain rejected
- **WHEN** 某属性的取值不属于该属性声明的合法集合
- **THEN** 系统 MUST 拒绝该用例并指明属性名
- **AND** 字符串属性 MUST NOT 被赋予通用占位值（实测教训：曾对所有字符串属性统一填 "sum"/"mean"，gelu 收到 approximate="sum"、scatter 收到 reduce="mean"，两者都非法）

#### Scenario: Attributes survive freezing
- **WHEN** 派生用例被写入冻结清单
- **THEN** 清单中的用例 MUST 保留派生阶段生成的属性，缺失必填属性的用例 MUST 被视为不可执行

### Requirement: Multi-input operators supply their own input construction
多输入算子的输入 SHALL 由该算子的参考实现所提供的构造过程产出；单输入算子 MAY 使用默认的单张量构造。

默认构造 MUST NOT 被用于多输入算子：单个形状无法推导出第二个张量的形状与取值范围。

#### Scenario: Second input derived from operator semantics
- **WHEN** 构造一个多输入算子的用例输入
- **THEN** 每个辅助输入的取值范围 MUST 满足该算子的语义约束（如索引落在合法区间内）

#### Scenario: Input construction failure attributed to evaluator
- **WHEN** 算子提供的输入构造过程执行失败
- **THEN** 该用例 MUST 被判为参考失效，MUST NOT 归因给被测方

#### Scenario: Constructed inputs are deterministic
- **WHEN** 同一种子与属性被重复用于构造输入
- **THEN** 两次构造 MUST 得到逐位相同的输入

### Requirement: Reference self-check is a freeze gate
冻结测试集之前 SHALL 对每个用例执行一次参考实现自检；自检不通过的用例 MUST 被剔除并记录原因，MUST NOT 进入冻结集。

自检 MUST 覆盖：输入可构造、参考可执行、输出非空且有限、以及**同一输入两次执行结果一致**。

为什么是门禁而不是善后步骤：
    不可复现的参考会让「静默错误」与「测量噪声」无法区分，
    而它造成的失败记为参考失效，长着一张「被测方没跑通」的脸。
    实测教训：cross_entropy_loss 的两个整型用例在昇腾上两次执行结果不一致，
    却已经进了冻结集——因为冻结时从没跑过自检。

#### Scenario: Non-deterministic reference excluded
- **WHEN** 某用例的参考实现两次执行结果不一致
- **THEN** 该用例 MUST 被剔除，剔除原因 MUST 记入冻结清单

#### Scenario: Self-check skips when torch unavailable
- **WHEN** 冻结在没有 torch 的机器上执行
- **THEN** 系统 MUST 明确说明自检被跳过，不得假装通过

#### Scenario: Excluded cases disclosed
- **WHEN** 查询冻结清单
- **THEN** 被剔除的用例及其原因 MUST 可被列出

### Requirement: Frozen set carries a reference implementation per operator
冻结的测试集 SHALL 为每个算子提供参考实现，且该参考实现 MUST 独立于任何提交。

#### Scenario: Missing reference disclosed
- **WHEN** 冻结或加载测试集时某算子缺少参考实现
- **THEN** 系统 MUST 明确指出缺失的算子，不得静默略过

#### Scenario: Missing reference judged as evaluator gap
- **WHEN** 执行一个缺少参考实现的算子的用例
- **THEN** 结果终态 MUST 为参考失效，且 MUST NOT 计入评分分母

#### Scenario: Reference executable without accelerator
- **WHEN** 对全部冻结用例执行一次参考实现自检
- **THEN** 每个用例 MUST 能构造输入并得到非空、有限且可复现的参考输出
