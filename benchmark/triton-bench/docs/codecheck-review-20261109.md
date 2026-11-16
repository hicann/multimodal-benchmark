> 历史工作记录。旧本地修复工具与输出保留在迁移前备份中，不作为当前平台发布工具；当前防回归入口见 [质量记录](quality/codecheck-recovery-20261116.md)。

> 后续实施说明：本文保留初次评估的历史论证；其中保留公开参数名、仅说明退出规则例外的建议已被用户否决并由实际联动修改取代。最终方案与验证入口见目标仓库 `benchmark/triton_bench/docs/codecheck-20261109-regression.md`，完整操作命令见 `.local-only/codecheck-20261109-workflow.md`。

# 2026-11-09 Code Check 剩余 28 条告警修改方案评估

评估对象：`CANN_multimodal-benchmark_全量导出（未解决）_20261109124716.xlsx`。本轮仅评估，没有修改业务源码、扫描配置、测试或资产。

结论：建议优先对 1 条字典访问做等价、局部修改；6 条命名问题涉及已发布签名，需要按契约例外或正式接口迁移处理；11 条 SystemExit 问题必须区分安全边界、故障注入和仅引用异常类型的代码；10 条 binary 分类问题建议在根目录 OAT.xml 按精确路径固化核验结果，并复扫确认。不能把上述条目机械地统一改名、改异常或改后缀。

## 1. 核验范围与证据

导出共有 28 条，10 条严重、18 条一般，全部状态为未解决。原表行号以下均指工作表 `代码问题详情(1)` 的物理行号，表头为第 1 行。

| 规则 | 数量 | 级别 | 处理方向 |
| --- | ---: | --- | --- |
| OAT.1 | 10 | 严重 | 文件类型分类核查 |
| G.NAM.04 | 6 | 一般 | 固定接口参数冲突，保留契约或正式迁移 |
| G.ERR.11 | 11 | 一般 | 逐项适用性复核，保留退出防护和负向测试 |
| G.TYP.07 | 1 | 一般 | 可以局部改写，消除条件安全访问告警 |

当前工作区是旧布局，报告对应源码位于：

`/Users/mac/CodeBuddy/triton-bench-gate-quality-modify/multimodal-benchmark`

该仓库当前 HEAD 为 `336fcb3ec2cf2879dae5fd37156c61c11530c486`，评估时工作区干净。28 个报告路径均存在，18 条 Python 告警的全部导出上下文与本地源码逐行一致。因此本报告依据的是匹配导出的目标实现，而非旧工作区的近似代码。未取得服务端扫描提交、范围和有效配置的完整元数据，不能仅据此断言当前 HEAD 就是服务端扫描提交。

逐条文件 SHA-256、工作簿 SHA-256、解析结果及参数改名实验结果保存在当前工作区的 `runs/codecheck-review-20261109-124716/assessment.json`。这些记录是本地评估证据，所有 28 条远程关闭状态均未确认。

## 2. G.ERR.11：11 条应按行为分类

用户给出的建议“在 __main__ 或入口直接调用的函数使用退出”适用于主动结束可信 CLI 进程。此次报告主要落在评测系统处理不可信代码的边界与测试中，不能套用为“把所有 SystemExit 挪到 main”。

以下路径均省略公共前缀 `benchmark/triton_bench/`。

| 表行 | 文件与代码行 | 实际行为 | 评估与建议 |
| ---: | --- | --- | --- |
| 18 | `examples/adversarial-submissions/ADV-E3-systemexit/operators/exp/kernel.py:28` | 模块末尾 `raise SystemExit(0)` | 故意构造导入期退出攻击。移入 `if __name__ == '__main__'` 后，正常模块导入不再触发攻击，原样例失效。建议保留并说明负向样例用途。 |
| 19 | `src/triton_eval/safeexec.py:63` | 捕获 `(SystemExit, GeneratorExit)` 并转换为 `SubmissionEscape` | 不调用退出，而是阻止提交代码终止评测器。按普通应用逻辑“必须一直向上抛”改写会破坏安全边界。建议保留并记录边界设计理由。 |
| 20 | `src/triton_eval/safeexec.py:69` | `isinstance` 识别逃逸异常 | 没有退出调用或抛出，建议作为规则适用性疑点复核，保留代码。 |
| 21 | `tests/core/test_codecheck_compat.py:1486` | `pytest.raises(SystemExit)` 检验恶意样例 | 是异常断言，建议复核告警适用性；保留退出码 0、属于 BaseException 而非 Exception、退出后无语句的检查。 |
| 22 | `tests/coverage/test_backend_diagnostics.py:274` | 断言 CLI 拒绝把输出写入证据目录时退出 | 被调用的 `DIAG.main()` 使用 `argparse.parser.error()`；测试本身没有退出调用。CLI 入口允许退出，建议复核告警适用性。 |
| 23 | `tests/runtime/test_safeexec.py:37` | `guard_call(sys.exit, 0)` | 故意向边界传入真实退出函数，检验转换为 `SubmissionEscape`。改为普通异常无法证明退出防护有效，建议保留为安全测试例外。 |
| 24 | `tests/runtime/test_safeexec.py:56` | 构造 `SystemExit()` 供类型识别 | 实例化不会触发进程退出，建议复核告警适用性。 |
| 25 | `tests/runtime/test_safeexec.py:58` | 构造嵌套的 `SubmissionEscape(SystemExit())` | 同上，验证包装后的异常仍可识别，不应替换成无关异常。 |
| 26 | `tests/submissions/test_submission_simulation.py:162` | 故障注入函数主动抛出 SystemExit | 模拟 worker 中断，让任务保留 RUNNING 状态，供重启恢复测试。不能直接改为 RuntimeError。建议保留故障注入例外；若必须重构，改为真实独立 worker 进程的终止与恢复测试。 |
| 27 | `tests/submissions/test_submission_simulation.py:168` | 捕获上项故障注入 | 是测试断言。与表行 26 是同一测试行为的两条告警，必须逐条保留映射。 |
| 28 | `tests/submissions/test_submission_verify.py:206` | 断言未确认隔离条件的 CLI 以退出码 2 拒绝运行 | `ENTRY.main()` 经 `parser.error()` 退出；应保留隔离门禁、退出码断言和 CLI 行为，复核告警适用性。 |

其中 7 条仅涉及类型识别、构造或测试断言（表行 20、21、22、24、25、27、28），属于优先复核适用性的候选；另外 4 条涉及主动攻击构造、真实退出测试、故障注入或边界捕获（表行 18、19、23、26），需要明确安全设计及测试例外依据，不能简单都称为误报。

`EvaluationService.work_once()` 会捕获普通 `Exception`，写 worker 日志、暂停服务并将任务置为 FAILED。现有 SystemExit 不属于 Exception，故障注入才能留下 RUNNING 供新实例恢复为 INTERRUPTED。用 RuntimeError 替换会走另一条路径，虽然消除了退出符号，却改变了测试内容。

也不建议自定义直接继承 BaseException 的异常来替代：仓库提供的规则集 `mr_ruleset_20261106134530.xlsx` 还列有 Python `G.ERR.03 异常的类型定义必须继承自Exception`，级别为严重。当前扫描是否启用该项尚缺配置证明，但这种方案存在引入新规则问题的明确风险。把 SystemExit 隐藏在别名、动态属性或字符串执行后面也没有改善设计，不推荐。

如果要求所有代码告警必须通过源码重构消除，工作进程恢复测试可以评估独立子进程方案：在子进程入口结束进程，父进程核验持久化任务、退出状态、队列保留、服务锁释放和重启后 INTERRUPTED 状态。这是一次测试结构调整，需覆盖原恢复语义，并不能解决 `safeexec` 必须识别和拦截退出异常的根本需求。

## 3. G.NAM.04：6 条均是公开参数 `input`

| 表行 | 文件与代码行 | 入口 |
| ---: | --- | --- |
| 12 | `examples/demo-submission/operators/swi_glu/kernel.py:55` | `swi_glu(input, dim=-1)` |
| 13 | `scripts/example_sources/adversarial_submissions_scalar_kernels_340c775b28.py:99` | `arg_max(input, dim=None, keepdim=False)` |
| 14 | 同上 `:314` | `swi_glu(input, dim=-1)` |
| 15 | `scripts/templates/coverage_kernels.py:75` | `cummin(input, dim)` |
| 16 | 同上 `:146` | `cross_entropy_loss(input, target, reduction='mean', ignore_index=-100)` |
| 17 | `scripts/templates/independent_submission_kernels.py:222` | `cummin(input, dim)` |

这 6 处都是保留接口签名的薄包装，内部实现已使用 `input_value` 等名称，没有证据表明这里意外调用或赋值覆盖了内置 input。规则命名冲突确实存在，但与无约束的局部变量重命名不同。

实际契约在 `src/triton_eval/operator_registry.py` 明确要求：Cummin、ArgMax、CrossEntropyLoss、SwiGlu 的首参数名为 `input`。`src/triton_eval/validator.py:255` 起不仅比较参数个数，还精确比较完整参数名列表。

本地用真实 `source_callable()` 和 `signature_errors()` 对 6 个原始函数逐一核验：原签名全部通过；仅把参数名改为 `input_value`，6 个全部返回 `SUB-LOD-007`。实验仅操作内存 AST，没有修改源码。这也说明“张量运行时按位置传参”不足以证明改名兼容，加载与静态预检还会先检查名字；公开的 `input=` 关键字调用也会受影响。

方案比较：

| 方案 | 行为与成本 | 建议 |
| --- | --- | --- |
| 只把 6 处参数改名 | 直接破坏现行签名校验；模板生成后会扩散问题 | 不采用 |
| 外层继续保留 `input`，内部使用 `input_value` | 目标源码已经如此；保留兼容，但告警仍存在 | 可作为契约边界设计，配合正式例外处理 |
| 用装饰器、`__signature__` 或 `**kwargs` 伪装兼容 | 现有预检要求一个直接、无装饰器的同步入口，并检查源码参数；还可能改变参数错误语义 | 不作为低成本修复方案 |
| 正式引入参数别名或迁移公开接口 | 需更新注册表、静态与动态校验、调用兼容、模板、样例、文档与测试 | 仅在决定修改接口契约后实施；严格限制到明确算子和参数，不放宽全局精确签名约束 |
| 保留固定公开参数，记录规则例外 | 保持现有提交与调用兼容；内部继续避免使用内置同名变量 | 当前优先建议；未正式接受前仍保持未解决 |

这次是方案评估，不据此修改扫描抑制配置或标记服务端已忽略。若后续做接口迁移，还应检查 `scripts/example_sources/layout.json` 的共享源认证和 `fixture_sources` 物化路径；修改源文件后须同步合法更新身份及构建验证，不能只改扫描所见文件或简单重新计算哈希就宣称兼容。

## 4. G.TYP.07：1 条可局部等价改写

表行 29：`tests/runtime/test_exec_runtime_guard.py:128`。

原实现已经先检查 `kind in ('child', 'grandchild')`，字典也恰好包含两个同名键，因此当前路径没有可见 KeyError 风险。为减少静态分析歧义，可以直接表达两类选择：

```python
if kind == 'child':
    output = original.as_subclass(TensorChild)
elif kind == 'grandchild':
    output = original.as_subclass(TensorGrandchild)
elif kind == 'parameter':
    output = torch.nn.Parameter(original)
else:
    output = original
```

此方案保持 Tensor、子类、孙类、Parameter 及嵌套输出的测试范围，删除了不必要的字典索引。建议优先采用，成本和行为风险较低。单纯改成 `classes.get(kind)` 会引入返回 None 的可能，表达反而不如显式分支清楚。

本轮未应用该修改；现有测试的 4 种 kind × 2 种 nested 共 8 种组合已通过，可作为后续修改前后的针对性回归。

## 5. OAT.1：10 条文件类型问题

表行 2—11，全部标记为严重、行号为 0，描述一致为 `Invalid File Type: binary`：

| 表行 | 文件 |
| ---: | --- |
| 2 | `spec/feature_atoms.yaml` |
| 3 | `spec/spec.yaml` |
| 4 | `spec/tolerance.yaml` |
| 5 | `examples/adversarial-submissions/ADV-E1-path-escape/manifest.yaml` |
| 6 | `examples/adversarial-submissions/ADV-E2-empty-operators/manifest.yaml` |
| 7 | `examples/ascend-ops-submission/manifest.yaml` |
| 8 | `examples/invalid-submissions/SUB-LOD-002/manifest.yaml` |
| 9 | `examples/adversarial-submissions/ADV-C1-fullhouse/manifest.yaml` |
| 10 | `src/triton_eval/templates/report.html.j2` |
| 11 | `src/triton_eval/templates/report.md.j2` |

逐文件核验结果：10 个均可严格解码为 UTF-8，NUL 字节均为 0；8 个 YAML 可由 `yaml.safe_load` 解析，2 个 Jinja 模板可由 `jinja2.Environment().parse` 解析。这些事实支持“本地文件是可解析文本”，无法证明服务端为何判为 binary，也不代表故意无效的提交 manifest 应通过业务校验。

进一步核查发现，目标仓库根目录已有 `OAT.xml`，其中 `binaryFileTypePolicyFilter` 为空。现有 `defaultPolicyFilter` 虽有 `*.yaml`、`*.yml`，但它不是二进制类型策略的过滤器，不能据此认为已经处理了 OAT.1。

具体建议：在根目录 `OAT.xml` 的 `binaryFileTypePolicyFilter` 下，为本表 10 个文件分别增加一条 `type="filepath"` 的精确路径记录，路径包含 `benchmark/triton_bench/` 前缀；每条 `desc` 说明已核验 UTF-8、配置或模板用途。不要使用全局 `*.yaml`、`*.j2` 或全目录通配符。

```xml
<filefilter name="binaryFileTypePolicyFilter" desc="Filters for binary file policies">
    <filteritem type="filepath"
                name="benchmark/triton_bench/spec/spec.yaml"
                desc="Verified UTF-8 YAML configuration; required by specification loading; not an executable binary"/>
    <!-- 其余 9 个路径见补丁，每个文件单独列出并说明用途。 -->
</filefilter>
```

OAT 官方说明将 `Invalid File Type` 的核实结果固化在该过滤器中，仓级路径相对仓库根目录配置。此做法是针对已核实文件的告警确认/过滤，不是让检测器重新识别文件类型，也不是修改文件内容。实际 Code Check 服务是否加载并沿用该机制仍需复扫确认；若流程禁止此类确认过滤，应改由扫描侧修正识别配置或工具缺陷，不能把过滤称为源码修复。

已生成可审阅补丁 `runs/codecheck-review-20261109-124716/oat1-exact-paths.patch` 和完整候选文件 `OAT.xml.proposed`，均位于当前可写工作区，尚未应用到目标仓库。候选 XML 可解析，10 个路径与原表逐条一致，除此之外 XML 配置不变。应用后应核对扫描日志确实加载根目录 `OAT.xml`，再按原 10 条路径复扫对账；若告警仍在，先排查配置加载和路径基准，不扩大过滤范围。

依据：OpenHarmony-SIG/tools_oat 官方 `README_zh.md` 中“开源仓规则配置”和“OAT问题确认 / 二进制文件（Invalid File Type）”。地址：`https://gitee.com/openharmony-sig/tools_oat/blob/master/README_zh.md`。

同时保留服务端扫描输入、类型识别依据的核查项。不要改后缀、删除模板/manifest 或改写内容来规避分类；这些文件被加载、模板渲染、提交校验和负向测试实际使用。

## 6. 验证与后续顺序

在匹配报告的目标仓库运行现有针对性测试，结果为 **21 passed in 10.81s**：

- `tests/runtime/test_safeexec.py`：9 项，含退出转换、GeneratorExit、KeyboardInterrupt 原样传播、验证与 CLI 退出行为。
- `tests/runtime/test_exec_runtime_guard.py::test_tensor_instance_policy_accepts_real_tensor_variants`：8 项。
- 导入期退出样例 AST 与退出性质、诊断 CLI 证据目录保护、worker 中断后重启恢复、隔离确认门禁：各 1 项，共 4 项。

使用当前工作区 `.venv/pattern-build/bin/python`，设置 `PYTHONDONTWRITEBYTECODE=1`、目标仓库的 `PYTHONPATH`，关闭 pytest 缓存，临时目录位于 `/private/tmp/codecheck-20261109-assessment`。未安装依赖、未改写目标仓库源码。测试证明当前相关行为成立，不是尚未实施方案的通过证据，也不等同于远程 Code Check 关闭。

建议后续顺序：

1. 将表行 29 做上述显式分支改写，运行 8 种张量组合回归，再复扫确认规则结果。
2. 对 G.ERR.11 的 7 条仅引用异常的告警提交适用性依据；对另外 4 条记录具体安全与负向测试设计，保留所有 11 条身份。
3. 对 G.NAM.04 的 6 条按固定接口例外处理；如要求源码消除，单独实施受控接口迁移，先评审兼容边界。
4. 按候选补丁在 OAT.xml 的 binaryFileTypePolicyFilter 固化 OAT.1 的 10 条逐文件核验，复扫确认配置加载和处理结果。
5. 最终按同一源码、配置及范围复扫逐项对账；条目没有出现在下一份导出，不足以单独证明已经修复。

本轮状态：28 条评估完成，业务修复实施 0 条，远程确认关闭 0 条。报告中的例外和适用性建议均为评估意见，尚未替代正式处理结果。
