# Spec Delta

## Purpose

检测并拦截「未真实实现算子」的提交，保证被测的确实是提交方自己的实现，而非对框架内置能力的转发。

## ADDED Requirements

### Requirement: Delegation to built-in compute APIs is rejected
提交的实现 MUST NOT 把目标算子的核心计算交给框架内置的计算接口完成。

#### Scenario: Direct built-in call detected
- **WHEN** 提交的实现直接调用框架内置的同类计算接口
- **THEN** 该提交 MUST 被判为无效并指明违规调用与其位置

#### Scenario: Equivalent spelling also detected
- **WHEN** 提交以等价写法调用同类内置计算（如运算符重载或算子变体）
- **THEN** 该提交 MUST 同样被判为无效

### Requirement: Device fallback is rejected
提交的实现 MUST NOT 把输入搬回主机侧计算后再搬回加速器。

#### Scenario: Host round-trip detected
- **WHEN** 提交的实现将张量移至主机侧完成计算再移回加速器
- **THEN** 该提交 MUST 被判为无效

### Requirement: Result caching or preset output is rejected
提交 MUST NOT 缓存执行结果或依据固定形状、数据类型、属性、输入地址返回预设结果。

#### Scenario: Address-keyed cache detected
- **WHEN** 提交的实现以输入张量地址作为缓存键
- **THEN** 该提交 MUST 被判为无效

#### Scenario: Input rotation exposes staleness
- **WHEN** 性能测量阶段轮换输入地址后复检精度
- **THEN** 依赖地址命中的缓存 MUST 暴露为精度失败

### Requirement: Fake or lazy tensor objects are rejected
提交 MUST 返回真实的张量对象，MUST NOT 返回包装器、懒求值对象或张量子类来延迟或伪装计算。

#### Scenario: Wrapper object detected
- **WHEN** 提交返回一个非真实张量的包装对象
- **THEN** 该提交 MUST 被判为无效

### Requirement: Profiler or timing tampering is rejected
提交 MUST NOT 替换、伪装或删除评测环境的计时、同步或性能采集接口。

#### Scenario: Synchronization neutralized
- **WHEN** 提交将同步接口替换为空操作
- **THEN** 该提交 MUST 被判为无效

### Requirement: Data movement allowance is category-aware
系统 SHALL 根据算子类别决定是否允许布局变换类操作；以其为核心语义的类别 MUST 被允许，其余类别 MUST 被拦截。

#### Scenario: Layout category permits movement
- **WHEN** 算子属于以布局变换为核心语义的类别
- **THEN** 布局变换类操作 MUST 被允许

#### Scenario: Non-layout category blocks movement
- **WHEN** 算子不属于布局变换类且实现中出现实质性布局变换
- **THEN** 该提交 MUST 被判为无效

#### Scenario: Contiguity verified structurally
- **WHEN** 提交的输出张量非连续且该算子类别不属布局变换类
- **THEN** 系统 MUST 判定为结构性失败而非精度不达标

### Requirement: Static detection precedes execution
系统 SHALL 在执行前通过静态分析检测违规，且 MUST 优先于运行期检测上报。

#### Scenario: Rejected without execution
- **WHEN** 静态分析发现违规
- **THEN** 该提交 MUST 在执行任何计算之前被拒绝

#### Scenario: Runtime detection as backstop
- **WHEN** 静态分析未发现但运行期检测到违规
- **THEN** 系统 MUST 在运行期拦截并判定无效
