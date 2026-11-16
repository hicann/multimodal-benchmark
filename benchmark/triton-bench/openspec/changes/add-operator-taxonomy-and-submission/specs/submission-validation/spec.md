# Spec Delta

## Purpose

对用户提交进行分阶段校验，使任何不合规的提交在执行前或执行中被明确拒绝，并给出可定位的失败原因，而非笼统报错或静默失败。

## ADDED Requirements

### Requirement: Runtime output guard applies per case
运行期输出校验 SHALL 对每个用例的执行结果生效，MUST NOT 只在离线校验中定义。

伪张量（包装器、懒求值对象、张量子类）与非连续输出（布局变换类除外）MUST 在精度比对之前被拦截，所属用例 MUST 判为提交不合规。

#### Scenario: Fake tensor rejected during evaluation
- **WHEN** 某用例的被测实现返回包装器、懒求值对象或张量子类
- **THEN** 该用例 MUST 被判为提交不合规，不得进入误差比对

#### Scenario: Non-contiguous output rejected outside layout category
- **WHEN** 非布局变换类算子的实现返回非连续输出
- **THEN** 该用例 MUST 被判为提交不合规，且 MUST NOT 计入评分分母

#### Scenario: Layout transform exempted
- **WHEN** 布局变换类算子的实现返回非连续输出
- **THEN** 该输出 MUST 被放行——搬运本身就是这类算子的语义

#### Scenario: Guard precedes accuracy judgement
- **WHEN** 输出不合规但其数值恰好落在容差内
- **THEN** 结果 MUST 仍为提交不合规：拿非真实张量比对得出的「通过」不成立

### Requirement: Submission identity recorded per run
每一轮评测 SHALL 记录被测提交的标识、来源目录、内容指纹与合规校验结论，且 MUST 在报告中披露。

#### Scenario: Identity persisted even when run aborted
- **WHEN** 环境探针失败导致整轮作废
- **THEN** 提交身份与校验结论 MUST 仍然落库，不得因为本轮无结果而丢失

#### Scenario: Snapshot fingerprint recorded
- **WHEN** 记录一份提交
- **THEN** 系统 MUST 记录其内容指纹，使得目录名相同而内容不同可被区分

### Requirement: Validation runs in ordered stages
系统 SHALL 按「静态检查 → 加载检查 → 执行检查」的顺序校验提交，且前一阶段失败 MUST 中止后续阶段。

#### Scenario: Static failure short-circuits
- **WHEN** 提交在静态检查阶段失败
- **THEN** 系统 MUST 不进入加载与执行阶段，并直接返回静态阶段的失败原因

#### Scenario: Load failure short-circuits
- **WHEN** 提交在加载检查阶段失败
- **THEN** 系统 MUST 不进入执行阶段

### Requirement: Every rule has an identifier and a stated failure result
每条校验规则 SHALL 拥有唯一编号，且 MUST 声明其失败时对应的判定结果。

#### Scenario: Failure cites rule id
- **WHEN** 一条规则判定失败
- **THEN** 报错信息 MUST 包含该规则的编号

#### Scenario: Rules are enumerated
- **WHEN** 查询校验规则清单
- **THEN** 系统 MUST 能列出全部规则编号及其失败结果

### Requirement: Validation rules have both valid and invalid samples
每条校验规则 SHALL 至少配备一个合法样例与一个非法样例，且两者 MUST 都有自动化测试覆盖。

#### Scenario: Valid sample passes
- **WHEN** 对某规则的合法样例执行校验
- **THEN** 该规则 MUST 判定通过

#### Scenario: Invalid sample fails
- **WHEN** 对某规则的非法样例执行校验
- **THEN** 该规则 MUST 判定失败并给出编号

### Requirement: Failure location is reported
校验失败时系统 SHALL 给出可定位的信息，包括算子名、阶段、规则编号与具体位置或值。

#### Scenario: Report points to source location
- **WHEN** 静态检查发现违规调用
- **THEN** 报错 MUST 指明所在文件与行号

#### Scenario: Report states expected versus actual
- **WHEN** 因签名不符而失败
- **THEN** 报错 MUST 同时给出期望与实际的参数信息

### Requirement: Validation never crashes the run
校验过程中出现任何未预期异常时，系统 SHALL 将其归入对应阶段的失败，MUST NOT 导致整个评测进程终止。

#### Scenario: Unexpected exception contained
- **WHEN** 校验某算子时抛出未预期异常
- **THEN** 该算子 MUST 被判为校验失败，其余算子 MUST 继续评测

### Requirement: Validation is deterministic
对同一提交的同一算子重复校验 MUST 得到相同结果。

#### Scenario: Repeat validation
- **WHEN** 对同一提交连续执行两次校验
- **THEN** 两次的规则编号与失败结果 MUST 完全一致
