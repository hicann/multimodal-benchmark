# v0.19：新增算子负载与 20 例补齐

2026-11-13。父版本为 v0.18-scan-certification-candidates。父版本 1385 例、参考文件与已有提交 Python 源码均保留，新建 v0.19-workload-completion-candidates，不刷新旧 manifest 或旧资产索引。

| 新增算子 | 原例数 | 补充数 | 新例数 | 小/中/大/超大档 |
|---|---:|---:|---:|---|
| Scatter | 2 | 18 | 20 | 5/5/5/5 |
| Cummin | 4 | 16 | 20 | 6/6/5/3 |
| GroupedMatmul | 4 | 16 | 20 | 5/5/5/5 |
| CrossEntropyLoss | 2 | 18 | 20 | 5/5/5/5 |
| MaskedSoftmax | 5 | 15 | 20 | 6/6/5/3 |
| ApplyRotaryPosEmb | 2 | 18 | 20 | 5/5/5/5 |

合计新增 101 例，74 算子、1486 例。其他算子的原有用例不删减；少数算子此前已有额外模式 witness，所以全集不是简单的 74×20。

选择沿用现有覆盖引导思路：结构签名去重（种子差异不计新增结构）、规模与 dtype 配额、特性×dtype×规模边际覆盖、属性组合、成本与确定性排序；保留全部父例，补到每目标算子恰好 20 例。候选池只生成符合现有规格的形状/类型/轴/属性组合，不能用改种子凑配额。selection_trace 持久记录每次选取的规模、既有代表数和新增组合收益。小档≤4096、中档≤262144、大档≤2097152、超大档>2097152，按主输入元素数，非总内存或总算量。

Cummin 的新增负载沿短扫描轴扩展独立行数，不把每次扫描轴无限放大；原有两条百万级长轴分块扫描 witness 保留。GroupedMatmul 保持 K=32、输出宽度16，扩展行数、分组和 dtype；原模式 witness 原样保留。新输入域只在 v19 且属于生成候选域时启用，不放宽旧版本或任意未知形状的校验。

构建入口：`.venv/pattern-build/bin/python scripts/build_workload_corpus.py`。不提供跳过参考门禁参数。冻结前全部1486例独立CPU参考门禁、74个源码入口预检通过；重复构建只核验身份，不改冻结文件。静态标签仍71/71，编译/执行认证与双方确认不由CPU门禁代替。

- manifest：`bdc1e68beb0158b000fc4a607b81fbf9831d895fbaf84c11a7ff91ae7d8b5cb3`
- assets：`e10dcbcc52bdfb0c5ca1632e16c1fe1449f564aeffb664414882b4c297f25e11`
- 提交：`examples/coverage-v19-ascend-submission`
- 生成/门禁证据：新 corpus 中 coverage-plan.json、generation-log.jsonl、coverage.json。

平台默认 corpus/source 更新为 v19，通用 CLI 的历史默认未改。框架新增契约后旧版本完整资产校验会按设计拒绝直接用新框架执行；旧冻结文件和历史结论保持不变。历史测试仅在临时CPU夹具中重建当前框架快照，不能改写生产旧资产以通过测试。远端 bundle 增加对冻结索引中额外框架依赖的打包，避免只在本机有构建器、远端缺文件。

已完成半量标准实验：37/74算子、746例，含全部6个新增算子和12类负载，每个选中算子的全部冻结用例均运行；最终单任务746PASS，功能100，实测0.823829×，25例CV超阈值。随后实现同条件试运行榜单，详见[完整实验及榜单记录](half-standard-experiment-20261114.md)。

## 扩容后真机发现与提交实现修复（2026-11-14完成复验）

新冻结用例与参考、容差未因真机失败而更改；旧 v18 文件也未改写。v19 示例提交在初始复制后进行了下列修复，不能再将当前 v19 示例的全部 Python 文件称为与 v18 逐字相同：

- int64 Scatter 原子的跨 program 累加路径在 `workload-scatter-0105` 停滞。改为每个目标一个写入者、逐更新索引匹配并累加 int64，避免跨 program int64 原子竞争；六算子首轮定向复验中 Scatter20/20PASS，原卡住用例也通过。这是路径替换后的设备验证，不据此宣称已经定位编译器内部根因。
- Cummin 原始每输出一个 program 在新 XL 例产生2113568个物理program，设备报上限65535；首次修改为128输出局部块又在10例触发UB容量不足。当前提交进一步改为16输出局部块、最多4096物理program并循环步进，保留原长轴三阶段扫描路径。最终修复已于2026-11-14通过20例真机确认。

| 实验 | 实际结论 | 解释 |
|---|---|---|
| 半量首轮，37算子746例 | 648PASS，98RUNTIME_FAIL，任务FAILED、score=null；654文件哈希一致，最终stopped | 首个int64Scatter卡住后监督终止；98含后续未执行的补记，不是98个独立实现缺陷 |
| 六算子定向复验1，120例 | 37PASS、1RUNTIME_FAIL、82ORACLE_FAIL，任务FAILED、最终stopped | Scatter全部通过；Cummin物理网格超限后出现参考异常，不据此判断82个独立参考缺陷 |
| 六算子定向复验2，120例 | 110PASS、10COMPILE_FAIL，任务COMPLETED，功能91.6667，最终stopped | 编译失败均为Cummin；其余5个算子各20/20通过；COMPLETED不等于所有用例通过 |
| Cummin专项，20例 | 20PASS，功能100、任务内性能100，29文件哈希一致，最终stopped | 小局部块、受限物理网格及原长轴路径通过；任务f980f253e31b47d784b434c101c66674 |

完整原始证据分别位于 `runs/half-standard-20261113/`、`runs/v19-extension-repair-20261113/`、`runs/v19-extension-repair2-20261113/`、`runs/v19-cummin-repair-20261113/`。每轮由独立guard脚本管理总时限、回收和finally关停，不重复同一远端job的launch。

Cummin专项通过后，以修复后的相同Python源码完成最终37算子746例单任务标准复跑：
任务`3e05b944428640b4a9412363fa6fd745`，746/746PASS、功能100，实测几何平均加速比
0.823828597×、任务自身性能分100，2238轮中位数样本完整，755文件大小/哈希一致。
任务COMPLETED、launch=1、SHUTDOWN，guard最终确认云状态stopped。证据位于
`runs/half-standard-final-20261113/`，独立审计复算样本中位数、CV及速度比全部通过。
此结果不是拼接分区；未执行全量74算子标准实验。之后才开展榜单实现和联调。

## 重现实验准备

使用 `.venv/pattern-build/bin/python scripts/platform_examples/prepare_half_experiment.py --out runs/half-replay` 生成确定性 submission.zip 和 plan.json。此命令不启动 NPU。计划包含37个算子、746例、12类负载，保留每个选中算子的全部冻结用例；同时记录源码文件哈希与ZIP哈希。已核对其算子选择和Python源码与最终真机复跑一致（ZIP元数据采用固定时间，因此不要求与此前手工包的字节哈希相同）。8项准备/生成回归通过。


> 2026-11-16 迁移说明：以上真机结果属于原日期与原源码指纹。本轮改用 ba04485 已修复共享源码并调整提交实现；新生成 ZIP 不继承上述真机认证。冻结用例不变，当前迁移复核见 [质量记录](quality/codecheck-recovery-20261116.md)。
