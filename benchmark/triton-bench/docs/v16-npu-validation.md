# v0.16 真机复核：两个后端模式的编译、执行认证通过

## 本轮结果

结果记录为 `20261027-043324-279333`，实际执行代码为
`0546fb488c4a12ac2304fc203bba95ba00a765c0`。入口正常结束，返回码为 0。
只运行 Neg、Matmul 各 20 例，共 40/40 PASS，两者 CPU、NPU 参考门禁均通过。
载入预检检查 74 个声明入口并通过，不代表执行了全部 74 个算子。

冻结 manifest 为
`e758eaef22e06c06a0d533bc5dabb1ee2bd35b74ff2af490dd2d718c57eff546`，
评测资产指纹为
`4f1688c8cbed58ddfe8e02453734bb3f25d73febbfe8ba70ad2eebe5d52d43ab`。
与发布前冻结版本一致，没有修改用例、输入、参考、容差或提交内核来获得认证。

| 标签 | 编译认证 | 执行认证 | 本轮有效 witness 数 |
| --- | --- | --- | --- |
| `FA-PAT-VEC-LOAD` | True | True | 2 |
| `FA-PAT-TILE-GEMM` | True | True | 1 |

独立 `backend-pattern-validation.json` 为 `passed=true`，绑定的覆盖报告
SHA256 为 `38d546c832f0fb4fca3cd07f9142c1785ac1a14487550d5b865a6dffad310c14`。
这次不只是算子功能 PASS，而是两个目标模式的正式认证门禁也已通过。

## 编译与执行证据

| 观测 | Neg | Matmul |
| --- | --- | --- |
| 正确性观测中的 JIT launch | 20 | 20 |
| 本轮实际编译 | 16 | 14 |
| 本轮同特化证据复用 | 4 | 6 |
| 输入/返回二进制绑定通过，含复用 | 20 | 20 |
| 每次绑定取得的目标函数快照 | 2 | 2 |

30 次真实编译与 10 次同运行复用不能写成 40 次真实编译。
两种模式均来自已绑定后端编译的 `hivm-graph-sync-solver` 函数级快照，
不是诊断重放文件，也不是把普通 TTIR load/dot 改名当作后端认证。

- Neg 的有效用例为 `official-neg-float16-27-normal-c`、
  `official-neg-float16-28-normal-c`。其证据为真实输入经
  `hivm.hir.load` 进入 UB，经运算后写回真实 Out 的存活数据流。
  识别器报告的 256 是块传输宽度，不宣称是硬件 SIMD lane 数。
- Matmul 的有效用例为 `official-matmul-float16-24-normal-c`。
  两路输入分别进入 CBUF，参与 `hivm.hir.mmadL1`，生成 CC 结果，
  并经 fixpipe 写回真实 Out。仅报告实际观察到的 CBUF/CC，不虚构 L0A/L0B。
- 执行认证还要求对应 case 身份、内核源、真实张量绑定、同步记录、功能 PASS
  与 CPU/NPU 参考门禁成立；只有后端文本出现操作名不能通过。

## 本机独立复核的范围

校验本轮 318 个报告索引文件的 SHA256，以及 manifest、1377 个 case 内容
哈希和全部冻结资产文件哈希。重新读取全部 40 个真实 observation，调用
现有 `observation`、`feature_witness` 和后端绑定识别器进行离线重放，
得到的编译 witness、执行 witness 和详细 implementation evidence 与归档报告一致。
模式门禁绑定的报告哈希也一致。

本机旧 Python 环境缺少可用的 pydantic/PyYAML，未成功重新调用完整
`audit_suite`。因此本次独立复核不表述为“完整审计入口重跑成功”，而是
标准库身份/哈希检查加全部观测和特征证据重放成功。本机没有执行 NPU 内核，
执行认证来源仍是本轮归档的真机记录。

## 六轮历史分区叠加

重新核验全量基线、v12、v13、v14、v15 和 v16 六份源报告及其全部索引文件
哈希，并按时间选择每个算子的最新分区。Neg、Matmul 由 v16 替换旧分区，
不累加相同 case 或相同标签，也不把百分比相加。

v16 与 v15 的 1377 个 case 内容哈希、71 标签分母、语义/参考/容差资产一致，
全部提交 Python 内核相同；提交 manifest 仅 submit_id 不同，算子路由不变。
更早轮次的兼容性沿用并绑定此前独立审计链，保留所有源报告哈希和逐标签来源。

| 口径 | 覆盖标签 | 比例 |
| --- | --- | --- |
| 全语料静态覆盖 | 67/71 | 94.37% |
| 历史分区叠加编译覆盖 | 67/71 | 94.37% |
| 历史分区叠加执行认证覆盖 | 67/71 | 94.37% |

编译与执行覆盖均由此前 65/71（91.55%）增加到 67/71（94.37%），
增加约 2.82 个百分点。全部 67 个静态已覆盖标签现均有历史适用分区的编译、
执行认证证据，`static_without_compilation=[]`。
1377/1377 个 case 有最新适用分区的 PASS，无缺失或失败。

本轮只跑两个算子的独立报告为 26/71（36.62%），此前对应定向报告为
24/71（33.80%）。这不是全系统回退，也不能把全语料静态 67/71 当成本轮
40 例单独完成的编译、执行覆盖。

仍未覆盖的四项同时缺少静态覆盖和动态认证，不再是“静态已有但动态缺证”：

- `FA-PAT-MULTIPASS-REDUCE`
- `FA-PAT-SCAN-BLOCKED`
- `FA-PAT-SPLIT-K`
- `FA-PAT-VEC-BROADCAST`

## 验收边界与下一步

历史分区叠加的标签覆盖已超过当前 71 项清单的 90% 数值门槛，
但不等于同版本全量验收，更不是编译器全部 API、IR 或代码分支覆盖率。
四项剩余模式不是达到 90% 的必要条件；如继续追求更高覆盖，应增加真实实现、
可审计识别器和新冻结版本，而不是只补标签或放宽规则。

本轮 `venv_lock_hash=ff2a31f2bac4ced4`，与历史各轮不同，其余已核对的
硬件/编译器/运行时标识一致。因此 `single_locked_environment=false`，
`verified_npu_coverage_target_met=false`、`taskbook_accepted=false` 仍保留，
共同确认状态仍为 `AWAITING_JOINT_CONFIRMATION`。
原报告中的 `complete=false`、`all_pass=false` 是以 1377 例全语料为范围，
不能误解为本轮 40 例失败；本轮 `selected_operator_validation_passed=true`。
静态子报告中的 `NOT_RUN` 是静态预检状态，不覆盖本轮独立的动态认证结果。

若要形成当前冻结版本的统一验收报告，下一步应在同一锁定环境运行全量
1377 例并重新审计三层覆盖；这是后续验收建议，不是本轮额外启动的真机任务。
性能测量、收益与任务书最终验收也不能由本轮 smoke 正确性测试代替。

## 复核产物

- 原始结果：`/private/tmp/triton-v16-npu-results-9dksku/records/20261027-043324-279333/evidence`。
- 独立复核：`/private/tmp/triton-v16-independent-audit.json`。
- 六轮叠加及逐标签来源：`/private/tmp/triton-v16-six-round-overlay.json`。
- 本次离线复核脚本：`/private/tmp/audit-triton-v16-results.py`。

本次仅新增结果分析及状态说明，不修改冻结资产、认证规则或历史真机报告。
