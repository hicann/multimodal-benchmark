# 对抗提交包（用户视角的鲁棒性攻击样例）

`ADV-C1-fullhouse` 的 14 个入口共用包内 `operators/scalar_kernels.py`，
由 `manifest.yaml` 分别声明入口函数。张量遍历、嵌套和构造只维护一份，
ArgMax 的标量输出构造单独保留；其他提交包的错误行为不受影响。
该目录仍可直接校验、运行或压为自包含 ZIP，不依赖其他提交包或评测器内部模块。
本次源码整理不改变已有真机记录，也不代表对新布局作出了真机认证。

**2026-11-08 策略调整**：输出类型改用 `isinstance`，Tensor 子类不再仅因类型身份
触发 `SUB-EXE-001`。下文 `ADV-D9` 的旧拦截预期及真机记录是历史策略下的结果，
不代表当前策略；现行边界和验证要求见 `docs/tensor-output-policy.md`。

定位：**我是提交方**，这 23 份目录是「用户上传的算子」；闭环由评测方在昇腾机上跑。
与 `examples/invalid-submissions/` 的关系：那套是**规则的正向说明**（每条规则一个反例，
都应被拒）；这套是**对抗攻击**——专门验证规则在刁钻写法、复杂结构、运行期行为下
是否真的拦得住、判得对、崩不崩。算子全部归属注册表的 14 个算子（cate 之内），
测试用例全部来自冻结 corpus（`corpus/v0.3-multi-input-frozen`，60 个用例）。

分组命名：

| 前缀 | 含义 |
|---|---|
| `ADV-C*` | 正控制：刁钻但合法（多输入 / 多输出 / 全属性变体 / 零框架依赖手写实现） |
| `ADV-D*` | 越权攻击：违反规则意图，静态阶段**应当拒** |
| `ADV-E*` | 结构 / 参数 / 提交契约攻击 |
| `ADV-F*` | 执行期判定攻击（只有真跑才暴露） |

`_shared/escapee.py` 不是提交，是 `ADV-E1` 引用的**提交目录外**文件。

---

## 一、本机预检结果（真实 CLI，静态 + 加载阶段）

本机（macOS，有 torch 无 triton/torch_npu）只能跑到 validate；执行闭环留给昇腾机。
预检命令：`python -m triton_eval validate --submission examples/adversarial-submissions/<目录>`

**2026-10-09 复检**：加固提交落地后全部重跑过。静态层现状：D 组换写法绕过
（方法形态 / 链式属性 / 别名 / setattr / id()）与 E 组契约攻击（路径逃逸 /
SystemExit / posonly / autotune）**全部被拒**。仍故意放行的两条：`ADV-D7`
（import 期投毒，静态不可见——防线在运行期篡改校验 SUB-EXE-005）、`ADV-E2`
（空清单是合法 YAML，只告警不拒）。

| 提交 | 攻击点 | 应然 | 本机预检实测（2026-10-09） | 判定 |
|---|---|---|---|---|
| `ADV-C1-fullhouse` | 14 算子全手写（正控制） | 通过 | 通过（14 算子载入） | ✓ |
| `ADV-D1-method-form` | `x.exp()` 方法形态代算 | 拒（SUB-STR-001） | **拒**（SUB-STR-001） | ✓ 已修 |
| `ADV-D2-chained-call` | `torch.nn.functional.softmax` 链式属性 | 拒（SUB-STR-001） | **拒**（SUB-STR-001） | ✓ 已修 |
| `ADV-D3-alias-import` | `import torch as _t` 别名 | 拒（SUB-STR-001） | **拒**（SUB-STR-001） | ✓ 已修 |
| `ADV-D4-host-roundtrip` | `x.to("cpu")` 主机往返 | 拒（SUB-STR-002） | **拒**（001+002） | ✓ 已修 |
| `ADV-D5-id-cache` | `id(x)` 键缓存 | 拒（SUB-STR-003） | **拒**（SUB-STR-003） | ✓ 已修 |
| `ADV-D6-import-tamper` | `setattr` 篡改 synchronize | 拒（SUB-STR-005） | **拒**（SUB-STR-005） | ✓ 已修 |
| `ADV-D7-oracle-poison` | import 时投毒 `torch.softmax` | 不可想象 | **通过**（静态不可见） | 留给 SUB-EXE-005 |
| `ADV-D9-tensor-subclass` | 张量子类（正控制） | 执行期拦（SUB-EXE-001） | 通过 | 待执行期验证 |
| `ADV-D10-noncontig-out` | 非连续输出（正控制） | 执行期拦（SUB-EXE-002） | 通过 | 待执行期验证 |
| `ADV-E1-path-escape` | 清单路径 `../` 逃逸 | 拒（根目录约束） | **拒**（SUB-LOD-008） | ✓ 已修 |
| `ADV-E2-empty-operators` | 空算子清单 | 至少告警 | 通过（0 错误 + 警告） | 静默放行→有告警 |
| `ADV-E3-systemexit` | import 时 `raise SystemExit(0)` | 结构化报错 | **拒**（SUB-LOD-005，rc=1） | ✓ 已修 |
| `ADV-E4-posonly-signature` | positional-only 形参 | 拒（绑定语义不符） | **拒**（SUB-LOD-009） | ✓ 已修 |
| `ADV-E5-comment-falsepositive` | 合法实现 + 注释提禁词 | 通过 | **通过** | ✓ 已修（误拒消除） |
| `ADV-F1-garbage-indices` | cummin 第二输出全错 | WRONG_RESULT | 通过 | 待执行期验证 |
| `ADV-F2-single-output` | cummin 只交一个输出 | 拒 | 通过 | 待执行期验证 |
| `ADV-F3-wrong-dtype` | softmax 返回 float64 | 拒/WrongResult | 通过 | 待执行期验证 |
| `ADV-F4-timeout-swallow` | 吞掉 SIGALRM 超时 | 60s 截断 | 通过 | 待执行期验证 |
| `ADV-F5-compile-mimic` | 运行期错误伪装编译失败 | RUNTIME_FAIL | 通过 | 待执行期验证 |
| `ADV-F6-mutate-inputs` | 毁掉评测方输入张量 | 至少告警 | 通过 | 待执行期验证 |
| `ADV-F7-dim-blind` | 无视 dim 属性 | 部分 WRONG_RESULT | 通过 | 待执行期验证 |
| `ADV-F8-autotune` | `@triton.autotune` | 拒（冻结启动配置） | **拒**（SUB-STR-007） | ✓ 已修 |

注：早期版本的预检里 D1–D7/E1/E4/F8 全部漏放、E3 静默崩溃、E5 误拒——
上表是修复后的现状。历史证据见 `docs/robustness-findings.md`。

## 二、纯代码审查即成立的缺陷（无需跑，供执行时对照）

1. **Scatter 调用约定自相矛盾（结构性）**：规格签名
   `scatter(data, dim, indices, updates, reduce)`，而执行器按 make_inputs
   顺序位置传参 `(data, index, updates)` + 关键字 `dim=…`——第二个位置实参
   （index 张量）绑定到形参 `dim`，再遇关键字 `dim` 必然
   `TypeError: got multiple values for argument 'dim'`。
   任何**与规格签名一致**的 scatter 提交都无法被调用；改成 make_inputs 顺序
   又过不了 SUB-LOD-007 参数名校验。**该算子在冻结集里没有任何可行签名。**
   （本机已按执行器真实调用方式复现 TypeError，见 `ADV-C1` 的 scatter。）
2. **交叉校验是空转的**：`oracle.py` 的第二参考指向 `ref_cpu.py`，
   而冻结 corpus 里 **一个都没有**——`cross_reference_for` 恒为 None，
   交叉校验从未执行，ORACLE_FAIL 的制衡不存在（这直接放大 D7 的破坏力）。
3. **提交不可变约束未执行**：`verify_unchanged` 定义了但 `cmd_run` 从不调用；
   快照只用于跑前指纹，跑后不复查（E1 的逃逸文件更是压根不在指纹里）。
4. **日志从未落盘**：`log_path` 建目录、进记录、进库，但没有任何代码写它——
   「保存原始日志」的规范要求是空转的。
5. **case.launch 是死数据**：性能口径「禁 autotune、配置写死」，但执行器
   从不把 launch 配置传给实现，autotune 也不在静态名单里（ADV-F8）。
6. **容差标定惩罚更精确的实现**：float32 档 `rel≤1e-4 且 abs≤1e-5`（合取）。
   本机对照冻结用例实测，一份比 oracle 更准的实现会错 5 个用例：
   - `gelu-f32-00`：负尾消零区（x=-3.73 → y≈-2.4e-4），oracle 自身 fp32
     舍入 5.3e-8 → rel 2.2e-4 超限（abs 2.4e-7 明明达标）；
   - `grouped_matmul-f32-00/03`：近零输出上 oracle 的 fp32 累加误差
     1.3e-5 → rel 4.3e-4 / 1.2e-4 超限；
   - `cross_entropy_loss-int32-03/int64-04`：**容差按 case.dtype 查表**，
   int 用例取「精确相等」档，但该算子输出是 fp32 标量——误差 7e-8
   也会判 WRONG_RESULT，任何与 torch 非逐位一致的实现都不可能通过。
7. **validate 的文档用法与 CLI 不符**：`examples/invalid-submissions/README.md`
   写的是位置参数用法，实际 CLI 要求 `--submission`。
8. **缺 torch/triton 环境的报错误导**：`ModuleNotFoundError` 被归为
   SUB-LOD-005「入口函数不可调用」（ADV-F8 本机预检可见）。

## 三、执行阶段的预期终态（跑完对照）

执行架构（2026-10 更新）：`run` 现在是**监督者**——提交代码的全部加载与
执行发生在独立子进程 `evalchild` 里，事件经 `events.jsonl` 流回主进程落库；
主进程负责硬超时（SIGTERM→宽限→SIGKILL）、崩溃归因与缺失用例补记。
加载后与每用例后有关键 API 篡改校验（SUB-EXE-005）。

**昇腾机实测（2026-10-09，triton-ascend 3.2.0 / CANN 8.5.0）**：
pytest 428 通过 3 跳过；zip→run→score→report 全链 ✓；11 个敌意包全部按
预期判定；D7/E3 拒收且 rc=1。下表预期已按实测修订。

耗时与环境预警：`ADV-F4` 每用例约 65–75s（超时信号被吞后由**事后耗时核对**
判 RUNTIME_FAIL）；`ADV-C1` 全量 smoke 约 4 分钟；`ADV-D5` 的计时虚高
要在 standard 档才能看到（smoke 不测性能）。

| 提交 | 预期 run 终态（含理由） |
|---|---|
| `ADV-C1` | v0.4 全量 240 题基线（2026-10-11 真机实测）：≈214 PASS；12×COMPILE_FAIL 全在 XL 档（设计行为）；10×RUNTIME_FAIL 全是 0 维张量用例（C1 的 `shape[d]` 越界缺陷被新考法抓住——战果）；4×WRONG_RESULT 已全部归因为**评测方冤案**并于当日修复（tolerance.yaml 1.1：gelu/grouped_matmul 的 f32 覆盖档 + bf16 rel 7.8e-3）——**容差修正后的下一轮预期：4 个静默错误清零（≈218 PASS / 12 CF / 10 RF），仍出现即为回归**。grouped_matmul-f32-S00 PASS——上一轮容差修复经真机复核确认 |
| `ADV-D1` | Exp 3×PASS——作弊实现拿满分 |
| `ADV-D2` | Softmax 3×PASS |
| `ADV-D3` | Maximum 5×PASS（int8 精确匹配也过） |
| `ADV-D4` | Exp 3×PASS——主机计算 + CPU 张量输出都无人拦 |
| `ADV-D5` | Softmax 3×PASS；standard 档 speedup 异常大（计时测的是 dict 查找） |
| `ADV-D6` | Exp 3×PASS——计时被篡改，无任何告警 |
| `ADV-D7` | **run 直接 INVALID（rc=1）**：import 期投毒 torch.softmax 被篡改校验检出（SUB-EXE-005），不再有「返回全零也 PASS」 |
| `ADV-D9` | SUBMISSION_INVALID ✓（SUB-EXE-001 生效的正控制） |
| `ADV-D10` | SUBMISSION_INVALID ✓（SUB-EXE-002 生效的正控制） |
| `ADV-E1` | 静态即拒（SUB-LOD-008 路径逃逸，已修复） |
| `ADV-E2` | 60×SUBMISSION_INVALID；score=0，无除零（已本机验证评分算术） |
| `ADV-E3` | **结构化报错（rc=1）**：SystemExit 在加载边界被转成 SUBMISSION_INVALID，不再静默消失 |
| `ADV-E4` | 加载期即拒（SUB-LOD-009 仅限位置参数，已修复）→ run INVALID |
| `ADV-F1` | Cummin 5×PASS——indices 全错也过 |
| `ADV-F2` | Cummin 5×WRONG/SHAPE——输出元数校验已生效（少交输出不再 PASS） |
| `ADV-F3` | Softmax 3×SUBMISSION_INVALID——输出 dtype 契约（SUB-EXE-003）已生效 |
| `ADV-F4` | Exp 3×RUNTIME_FAIL——事后耗时核对（超时信号被吞也判超时）；更极端的赖死由监督者 SIGKILL 兜底 |
| `ADV-F5` | Exp 3×RUNTIME_FAIL——归因按异常类型不按文本（F5 伪造的编译失败不成立） |
| `ADV-F6` | MaskedScale 5×SUBMISSION_INVALID——输入指纹校验（SUB-EXE-004）已生效 |
| `ADV-F7` | Softmax 3×PASS——3 个用例的 dim 恰好都是最后一维（语料盲区） |
| `ADV-F8` | 静态即拒（SUB-STR-007 autotune，已修复） |
| `ADV-E5` | validate 即拒（SUB-STR-003 命中注释），不会进执行 |

以上「预期」列是基于本机预检 + 代码审查的推断，**请以昇腾机实测为准**；
实测与预期的偏差本身就是要找的缺陷。
