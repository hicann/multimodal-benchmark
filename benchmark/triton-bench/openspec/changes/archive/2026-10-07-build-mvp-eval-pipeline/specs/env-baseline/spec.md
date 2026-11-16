# Spec Delta

## Purpose

固化评测执行环境，使其可被精确重建、跨运行比对与执行前验证，为评测结果的可复现性提供地基。

## ADDED Requirements

### Requirement: Environment bootstrap
系统 SHALL 提供单一的环境入口脚本，该脚本 MUST 完成 CANN 环境变量注入，并 MUST 将执行锁定到单张 NPU 卡。未提供该脚本的 shell 不得视为可用执行环境。

#### Scenario: Source environment script
- **WHEN** 用户在昇腾执行机上执行环境入口脚本
- **THEN** CANN 的环境变量被注入当前 shell，且可见 NPU 设备被固定为一台

#### Scenario: Lock to single NPU
- **WHEN** 执行机上存在多张可见 NPU 卡
- **THEN** 任意一次评测运行 MUST 只使用其中一张，禁止跨卡混跑

### Requirement: Compiler under test isolation
被测编译器 MUST 安装在独立于评测工具链的虚拟环境中；评测工具链的依赖清单 MUST NOT 包含与被测编译器同名的社区包。

#### Scenario: Toolchain dependency guard
- **WHEN** 解析评测工具链的依赖清单
- **THEN** 该清单中不得出现与被测编译器安装目录同名的包，否则校验失败

#### Scenario: Snapshot compiler environment
- **WHEN** 采集环境指纹
- **THEN** 记录被测编译器虚拟环境的路径与其依赖锁定的哈希

### Requirement: Environment fingerprint capture
每次评测运行 SHALL 采集完整的环境信息，并 MUST 计算一个确定性的环境指纹值存入该次运行记录。

#### Scenario: Fingerprint is computed
- **WHEN** 一次评测运行开始
- **THEN** 系统采集架构、操作系统、内核、NPU 型号与数量、驱动、CANN 版本与路径、Python、框架版本、被测编译器名称版本与来源、实测导入版本、虚拟环境路径、锁频状态，并计算指纹写入运行记录

#### Scenario: Import version is measured not declared
- **WHEN** 记录被测编译器版本
- **THEN** MUST 使用运行时实际导入得到的版本值，而非包管理器的元数据声明值

### Requirement: Pre-run health probe
每次评测运行开始前 SHALL 执行一个最小可执行探针；探针失败时整轮运行 MUST 被判定为无效且不得产出分数。

#### Scenario: Probe passes
- **WHEN** 运行开始前的最小探针执行成功且结果与参考一致
- **THEN** 该轮运行继续正常执行

#### Scenario: Probe fails
- **WHEN** 最小探针编译失败、执行失败或结果与参考不一致
- **THEN** 该轮运行被标记为无效，并记录失败原因，不产出任何分数

### Requirement: Cross-run comparability by fingerprint
两次评测运行的环境指纹不一致时，系统 SHALL 明确标识二者不可直接比较。

#### Scenario: Different fingerprints
- **WHEN** 对比两次运行的结果且两者环境指纹不同
- **THEN** 系统 MUST 标注「环境不一致，结果不可直接比较」

#### Scenario: Missing lock capability
- **WHEN** 执行环境无法锁定时钟频率
- **THEN** 环境指纹 MUST 记录锁频状态为未锁定，且该环境产出的性能数据 MUST 标注为未验证
