# 评测闭环鲁棒性缺陷报告（对抗提交样例集）

- 样例集：`examples/adversarial-submissions/`（23 份提交，算子全部归属注册表 14 算子）
- 用例来源：冻结 corpus `v0.3-multi-input-frozen`（60 个用例，未新写任何 case）
- 本机（macOS / torch CPU / 无 triton、torch_npu）：已跑真实 `validate`（静态+加载）
  与 CPU 数值自检；执行期结论按代码审查给出预测，**待昇腾机实测确认**
- 预期终态对照表：见 `examples/adversarial-submissions/README.md`

验证层级标记：
【本机已证实】真实 CLI 或与执行器同构的调用方式复现；
【代码证实】源码级论证成立、无需运行；
【待执行验证】需昇腾机跑 run 才能落锤（README 已给出预期）。

---

## P0（判定可信度被直接击穿）

### P0-1 SystemExit 静默击穿校验器，退出码还是 0
- 样例：`ADV-E3-systemexit`
- 验证：【本机已证实】`validate` 无任何输出、退出码 0。
- 机理：`submission.load_all` / `validator.validate_load` / `cli._load_and_validate`
  的兜底全部是 `except Exception`；`SystemExit`/`KeyboardInterrupt` 是
  `BaseException` 的直接子类，import 期抛出即穿透全部防线。
- 影响：`validate` 看起来「通过」；`run` 进程直接消失且退出码 0——
  CI / 包装脚本会把一次被攻击的评测当作成功收尾，连 INVALID 记录都没有。
- 修复：加载边界捕获 `BaseException`（转 SUB-LOD-005 结构化错误），
  CLI 顶层加 `sys.excepthook` 兜底并返回非 0。

### P0-2 Scatter 的调用约定自相矛盾：该算子不存在可行签名
- 样例：`ADV-C1-fullhouse/operators/scatter`（完全合规、语义正确的实现）
- 验证：【本机已证实】按执行器真实方式调用
  `impl(data, index, updates, dim=…)` 必然
  `TypeError: scatter() got multiple values for argument 'dim'`。
- 机理：规格参数序 `scatter(data, dim, indices, updates, reduce)`，
  但 `execute_case` 按各算子 ref.py 的 `make_inputs` 顺序**位置**传张量、
  属性走关键字。Scatter 的 make_inputs 顺序是 `(data, index, updates)`，
  第 2 个位置实参绑到形参 `dim`，再遇关键字 `dim` 冲突。把签名改成
  make_inputs 顺序又过不了 SUB-LOD-007（参数名不符）。
- 影响：任何合规提交的 scatter 用例全部 RUNTIME_FAIL，责任记在被测方头上；
  该算子 5 个用例从未有任何提交通过过（demo 提交也未包含它）。
- 修复：规格 `params` 改为「张量在前、标量属性在后」的
  `(data, indices, updates, dim, reduce)`；执行器统一按
  「spec 中 is-tensor 参数位置传、其余关键字传」组装调用（`runner.execute_kwargs`）。

### P0-3 提交代码可劫持 oracle（同进程零隔离）
- 样例：`ADV-D7-oracle-poison`
- 验证：【本机已证实】validate 全通过（无任何规则能看见 setattr 投毒）；
  执行期效果【待执行验证】：softmax 三个用例返回全零仍应判 PASS。
- 机理：加载即 import 用户代码，与参考实现共享同一进程、同一 `torch` 模块。
  `setattr(torch, "softmax", 假实现)` 后，`corpus/.../softmax/ref.py`
  调用的就是假实现——「期望值」由提交方定义。
- 放大器：【代码证实】冻结 corpus 没有任何 `ref_cpu.py`，
  `cross_reference_for` 恒为 None，交叉校验从未执行——第二道防线不存在。
- 影响：判定权的根基（oracle 独立性）可被一行 setattr 摧毁，
  且所有静态规则对 import 期副作用完全失明。
- 修复：执行期把 impl 放子进程（管道回传张量字节），oracle 留主进程；
  至少做到：加载后校验 `torch` 关键函数指纹（`id()`/code 对象比对），
  ref 一律在 import 期绑定私有引用而非运行期动态取模块属性；
  补齐 `ref_cpu.py` 并把「第二参考缺失」从静默跳过改为告警。

## P1（反作弊规则可被成套绕过 / 契约不设防）

### P1-1 SUB-STR-001（框架代算）三种零成本绕过
- 样例：`ADV-D1`（`x.exp()` 方法形态）、`ADV-D2`
  （`torch.nn.functional.softmax` 链式属性）、`ADV-D3`（`import torch as _t`）
- 验证：【本机已证实】三份全部 `validate` 通过。
- 机理：`anticheat._call_name` 只认「`前缀.方法()` 且前缀是 Name 节点且
  前缀 ∈ 硬编码白名单」；方法形态的接收者是变量、链式属性前缀取不到、
  别名导入不被解析（扫描器不读 import 语句）。
  讽刺点：D2 的代码与评测方自己的 `softmax/ref.py` 逐字等价。
- 影响：所有「不得代算」规则对愿意换写法的提交方形同虚设；
  demo 提交测试只覆盖了 `torch.softmax(x)` 直调形态。
- 修复：扫描前先收集 import 绑定（含 `as` 别名与 `from … import`），
  属性链回溯到 import 根后再比对；张量**方法形态**命中计算名单同样报警
  （`tl.` 与 `@triton.jit` kernel 体内部照旧豁免）。

### P1-2 SUB-STR-002（主机往返）：`x.to("cpu")` 不在名单
- 样例：`ADV-D4-host-roundtrip`
- 验证：【本机已证实】validate 通过；执行期【待执行验证】应 PASS
  （含输出为 CPU 张量——输出设备从不校验，比对前统一 `.to("cpu")`）。
- 修复：设备回退名单补 `to(...)` 携带 "cpu"/host 目标的形态
  （AST 取第一个参数字符串）；执行期校验 `out.device == 输入 device`。

### P1-3 SUB-STR-003（地址键缓存）：id() 换名即过 + 计时虚高
- 样例：`ADV-D5-id-cache`
- 验证：【本机已证实】validate 通过；standard 档计时测 dict 查找
  【待执行验证】。
- 机理：检测是 `source` 全文搜 `data_ptr` 字样；`id(x)` 语义等价、
  文本不可见。正确性阶段每用例新造输入所以真算；计时阶段同一批输入
  反复调用全部命中缓存。
- 修复：AST 化（`Call(attr='data_ptr')`），并覆盖 `id()`；
  计时前 clone 输入 / 每轮重造输入，让缓存键失效。

### P1-4 SUB-STR-005（计时篡改）：setattr 形态不可见
- 样例：`ADV-D6-import-tamper`
- 验证：【本机已证实】validate 通过（赋值形态检查只看 `Assign` 节点）。
- 修复：静态补 `setattr` 目标命中计时名单的形态；更根本的是运行期
  校验关键函数身份（见 P0-3 的进程隔离）。

### P1-5 提交根目录无约束：路径逃逸 + 指纹盲区 + 不可变未执行
- 样例：`ADV-E1-path-escape`（`path: ../_shared/escapee.py`）
- 验证：【本机已证实】validate 通过、模块正常加载；快照只含
  `manifest.yaml`，逃逸文件不在指纹里（已打印证明）。
- 机理：`load_operator` 直接 `root / entry.path`，无 resolve/contains 检查；
  `Submission.snapshot` 只 rglob 提交根；`verify_unchanged` 定义了但
  `cmd_run` 从不调用——「评测期间不可变」规范是空转的。
- 影响：「评测的就是当时那一份」对逃逸路径不成立；
  评测后（甚至评测中）改掉外部文件，事后无法追责。
- 修复：manifest path 必须 `resolve()` 后 `is_relative_to(root)`；
  快照覆盖全部实际 import 的文件；run 收尾调用 `verify_unchanged`。

### P1-6 容差标定惩罚「更精确」的实现（4 个用例判错）
- 样例：`ADV-C1-fullhouse`（比 oracle 更准的实现）
- 验证：【本机已证实】按冻结用例 CPU 对照：
  - `gelu-f32-00`：负尾消零（x=-3.73 → y≈-2.4e-4），oracle 自身 fp32
    舍入 5.3e-8 → rel 2.2e-4 > 1e-4（abs 2.4e-7 达标，但合取判失败）；
  - `grouped_matmul-f32-00/03`：近零输出上 oracle fp32 累加误差 1.3e-5
    → rel 4.3e-4 / 1.2e-4 超限；
  - `cross_entropy_loss-int32-03/int64-04`：**容差按 case.dtype 查表**，
    int 档要求精确相等，但该算子输出是 fp32 标量——误差 7e-8 也判
    WRONG_RESULT，只有与 torch 逐位一致才可能通过。
- 影响：float32 档 `rel≤1e-4 AND abs≤1e-5` 在消零/近零输出上不可达；
  CEL 的容差键取错维度。真 triton 实现与 torch 不同的累加顺序同样暴露。
- 修复：容差判定改「rel 或 abs」（对齐 allclose 语义，或近零区 abs-only）；
  容差键取**输出 dtype**（CEL 应查 fp32 档）；交叉校验资产补齐后可用
  双参考一致性给容差定标。

## P2（执行期判定与归因的盲区）

| 编号 | 样例 | 缺陷 | 修复方向 |
|---|---|---|---|
| P2-1 | `ADV-E4-posonly-signature` | 签名校验只比参数名列表，positional-only 形参通过校验、调用必炸，错误记提交方 | 校验形参 kind（拒 positional-only / VAR_*），或做一次真实绑定 dry-run |
| P2-2 | `ADV-F1-garbage-indices` | 多输出只比第一个（cummin 的 indices 全错也 PASS；ref.py docstring 自认缺口） | 元组输出逐元素比对 |
| P2-3 | `ADV-F2-single-output` | 输出元数不校验，少交输出也 PASS | 按 spec.outputs 校验元数 |
| P2-4 | `ADV-F3-wrong-dtype` | 输出 dtype 不校验（float64 偷渡也 PASS；设备同理见 P1-2） | 比对前校验 dtype/device |
| P2-5 | `ADV-F4-timeout-swallow` | SIGALRM 一次性 + 可被 `except BaseException` 吞掉，硬超时失效，用例 75s 照常 PASS | 进程级超时（子进程 + kill），或周期性重设 itimer |
| P2-6 | `ADV-F5-compile-mimic` | 终态归因靠异常文本含 "compil"——提交方可把运行期错误伪装成编译失败，把锅甩给被测编译器 | 归因按异常来源（impl 调用帧内抛的一律 RUNTIME_FAIL），不按文本 |
| P2-7 | `ADV-F6-mutate-inputs` | 输入张量被就地销毁，无告警（zero_/fill_ 也不在静态名单） | 用例前后校验输入指纹；就地方法入扫描名单 |
| P2-8 | `ADV-F7-dim-blind` | 语料盲区：softmax 3 个用例 dim 恰好都是最后一维，无视 dim 的实现全 PASS | casegen 对枚举型属性强制「非退化取值」；补 1 个非末维用例 |
| P2-9 | `ADV-F8-autotune` | 「禁 autotune / launch 写死」不设防：case.launch 从不传给实现，autotune 装饰器不在任何名单 | 把 case.launch 传入调用约定；或静态禁 autotune |
| P2-10 | `ADV-E2-empty-operators` | 空算子清单 0 错误静默通过（run 全 SUBMISSION_INVALID、score=0 无崩溃，已本机验证算术） | validate/run 对 0 算子提交给出显式告警 |

## P3（体验与文档）

| 编号 | 缺陷 | 证据 |
|---|---|---|
| P3-1 | SUB-STR-003 文本检测命中**注释/docstring**：合法实现因注释里提到禁词被拒，报错行指向注释 | `ADV-E5`：实现与 C1 相同，仅注释含禁词即被拒（第 18 行） |
| P3-2 | `log_path` 建目录、进记录、进库，但无任何代码写日志——「保存原始日志」空转 | 全仓 grep：仅 `cli.py:162-163` mkdir/赋值 |
| P3-3 | 交叉校验资产全缺失且静默跳过（同时是 P0-3 的放大器） | `find corpus -name ref_cpu.py` 为 0；`cross_reference_for` 恒 None |
| P3-4 | `examples/invalid-submissions/README.md` 的 validate 用法（位置参数）与 CLI（要求 `--submission`）不符 | 本机实测位置参数直接 argparse 报错 |
| P3-5 | 缺 torch/triton 时错误归为「SUB-LOD-005 入口函数不可调用」，误导定位（本为环境缺依赖） | `ADV-F8` 本机预检输出 |
| P3-6 | `run` 在无 NPU 机器上裸 traceback 崩溃（虽有「需要昇腾主机」说明，失败形态仍是栈） | 本机实测 ModuleNotFoundError 栈 |

## 统计与建议摘要

- 23 份对抗提交中，**20 份通过了静态+加载校验**：8 份按规则意图本应被拒
  （D1–D7 与 E1），1 份空清单静默放行（E2），9 份属执行期才可判定的攻击
  （D9/D10/F1–F7），1 份正控制（C1），1 份签名盲区（E4）；
  另有 1 份造成静默崩溃且退出码 0（E3），1 份被误拒（E5），1 份需真 triton 环境（F8）。
- 静态防线对「换写法」零抵抗（方法形态 / 链式属性 / 别名 / setattr / id()），
  根因是**文本与浅层 AST 匹配**，建议 import 绑定解析 + 属性链回溯 + 形参 kind 校验。
- 执行防线缺**结构化输出契约**（元数 / 逐输出 dtype / device）与**运行环境完整性**
  （函数指纹 / 输入指纹 / 超时可靠性），单进程架构是 P0-3 与 P2 系列的共同根因，
  子进程隔离一次性解决大半。
- 容差体系有两处标定错误（合取语义、case.dtype 作 CEL 容差键），一份更精确的
  实现错 5 个用例；Scatter 的调用约定让该算子整体不可通过——这两个是
  corpus/规格层的缺陷，与提交方无关。

---

## 附：2026-10-09 第二轮加固（本报告之后落地的结构性修复）

上述发现驱动了三轮修复（git 9bf9996 / 9553dbb / 2a6a9b4 + 本轮工作区改动）。
本节记录**结构性变化**——不是逐条打补丁，而是换掉了出问题的那层架构：

1. **执行改子进程隔离**（`evalchild` + 监督者）：提交代码的加载与执行
   不再与评测主进程同室。主进程绝不 import 提交代码，只做静态校验、
   落库、硬超时（SIGTERM→宽限→SIGKILL）与崩溃归因。这使 P0-3 类
   （同进程劫持 oracle）的爆炸半径从「整个评测」缩到「一次运行」，
   也是 P2-5（超时被吞）的可靠兜底。
2. **运行时篡改校验 SUB-EXE-005**（`integrity.ApiGuard`）：对计时/同步/
   oracle 数学接口/比对链路张量方法做身份快照→校验→恢复。快照点在
   任何提交代码 import 之前；import 后与每用例后各查一次，检出即恢复
   并作废。静态扫描对「换写法」失明的根因被这层兜住（D7 的 import 期
   投毒现在整份拒收）。
3. **环境一致性校验**：子进程实际加载的 torch/torch_npu/triton 版本
   与父进程环境指纹核对（cann-bench 的「构建期与运行期错配」事故的教训）。
4. **zip 摄取层 SUB-PKG-001~009**（`package.py`）：路径逃逸/符号链接/
   炸弹/扩展名白名单在写盘前逐条把关；内容寻址目录幂等解包；
   `ingest` 子命令 + `run --submission pkg.zip` 直进。提交身份链补齐为
   「包 sha256 → 解包目录 → 快照指纹」三层。
5. **产物契约**：每用例日志真实落盘（fd 级重定向，P3-2 关闭）、
   events.jsonl 协议留档、integrity.json（篡改校验/环境一致性/
   不可变核查）、artifacts.json（产物 sha256 清单）；
   `verify_unchanged` 接进 run 收尾（P1-5 后半）。
6. 静态层修复（前两轮已落，本节汇总）：import 绑定解析与属性链回溯
   （D1–D3）、主机往返名单补 `to("cpu")`（D4）、`id()` 入名单（D5）、
   setattr 形态（D6）、路径逃逸 SUB-LOD-008（E1）、posonly SUB-LOD-009
   （E4）、autotune SUB-STR-007（F8）、注释误报消除（E5）、SystemExit
   边界 guard（E3）、多输出逐元比对（F1/F2）、输出 dtype 契约（F3）、
   超时事后核对（F4）、归因按异常类型（F5）、输入指纹（F6）、
   容差按输出 dtype 查表（P1-6 的 CEL 部分）。

**仍开放**（按优先级）：交叉校验资产缺失（P3-3，ref_cpu.py 一个没有）、
F7 类语料盲区（属性非退化取值）、`log_path` 之外的 case.launch 死数据
（P2-9）。

**昇腾机实测修订（2026-10-09）**：scatter 调用约定（P0-2）与 CEL 容差键
（P1-6 后半）确认已修复——C1 全手写提交 60 用例 58 PASS，scatter 5×PASS、
CEL int32/int64 PASS，仅容差合取仍错杀 grouped_matmul 2 例。

**后续关闭（2026-10-11）**：两个「仍开放」项已落地——

1. **容差合取语义（P1-6 前半）**：`within_tolerance` 改「rel 或 abs」析取，
   并新增逐元素 allclose 通道（|impl−ref| ≤ atol + rtol·|ref|，对近零参考
   有正确的渐近行为）。C1 的 grouped_matmul f32-00/f16-01 两例在代码层面
   不再错杀；回归钉见 `tests/test_judge.py` 的
   `test_allclose_rescues_near_zero_grouped_matmul`。
   **2026-10-11 真机复核确认**：全新机器自举（CANN 8.5.0）+ v0.4 全量
   240 题跑 C1，grouped_matmul-f32-S00 判 PASS，17 项检查全过、
   pytest 461 通过/3 跳过。C1 在 v0.4 的其余失败属预期战果：
   12 题 XL 档纯 Python 超时（COMPILE_FAIL，负载分层的设计行为）、
   4 题 WRONG_RESULT 在 tanh 近似/大幅值新维度（纯 Python 实现的
   精度边界，待 diagnose 分型）、11 题 ArgMax RUNTIME_FAIL（未归因，
   留观）。
2. **跨轮回归对比命令**：CLI 新增 `compare`（两轮对比：功能回归/修复、
   加速比显著变化的清单与报告落盘，见 cli.py 的 cmd_compare）。
   另注：性能锚点同期从「轮级首轮基线」改为按算子锚点
   （anchors.py，死锁论证见其模块注释）。

**真机遗留观察项归因（2026-10-11 第二轮验证；从归档 results.jsonl 本地复核）**：

C1 在 v0.4 全量 240 题的 27 个非 PASS 全部归因完毕，判罚核对结论——
**26 个判罚正确，4 个静默错误全部是评测方的问题（冤案），已于当日修复**：

- **12×COMPILE_FAIL（XL 档）**：设计内（负载分层淘汰纯 Python）。
- **10×RUNTIME_FAIL（ArgMax）**：C1 实现的真实缺陷——0 维张量输入在
  `shape[d]` 直接 IndexError 崩溃（duration=1ms，全部 10 个 0 维用例）。
  v0.4 新增的 0 维形态考法抓到旧实现边界。**语料的战果，判罚正确。**
- **4×WRONG_RESULT（静默错误）——全部冤案，判卷口径已于 2026-10-11
  修正（tolerance.yaml 1.1 + scorer 0.2.1），待下轮真机复核**：
  1. `gelu-f32-S120 / M48`：**参考侧精度冤案**。昇腾 NPU 上 torch 的
     f32 gelu 以**默认模式**（approximate='none'，语义要求 erf 精确式）
     调用时实际返回 **tanh 近似精度**的结果——真机观测的 impl−ref 差
     与 |tanh_gelu − erf_gelu| 在各用例输入上的最大差逐位吻合
     （4.589e-4 / 4.735e-4）。方向排除了「C1 走 tanh 分支」的反向解释：
     C1 代码在 attrs={} 时固定走 erf，且反向解释预测的相对误差
     （3.492e-2，按 erf 分母）与实测（3.619e-2）不符，差值恰是 tanh
     与 erf 分母之比。全 9 个 gelu 用例的实际结果与该机制 100% 一致
     （approximate='tanh' 的用例双方同为 tanh，全部 PASS）。C1 的
     f64 erf 精确式（与真值差为 0）被判错。
     **修复：gelu.float32 覆盖档 abs=1e-3（精确式与近似式都过）。**
  2. `grouped_matmul-f32-M18`：**参考侧精度冤案**。extreme 输入
     （|x| 至 ~84）× K=128 的 f32 累加，参考自身与 f64 真值就有
     max_abs 6.0e-4 / max_rel 1.6e-3 的偏差（本地复现实测）——默认档
     abs=1e-5 比参考自身的累加噪声还紧两个数量级。C1 以 f64 累加更
     精确，被判错。**修复：grouped_matmul.float32 覆盖档 abs=1e-3。**
  3. `swi_glu-bf16-M40`：**bf16 容差标定问题**。参考量级 ~336 处
     1 ULP = 2.0（rel 5.95e-3），刚好超过旧 bf16 rel 容差 5e-3。bf16
     的 1-ULP 相对误差理论区间是 (2⁻⁸, 2⁻⁷] = (3.9e-3, 7.8e-3]，旧容差
     落在区间内会冤枉 binade 下半段的合法舍入（fp16 的 1e-3 已容纳
     1 ULP）。**修复：bf16 rel 5e-3 → 7.8e-3（= 2⁻⁷）。**

**判卷口径修正（2026-10-11，tolerance.yaml 1.1，scorer 0.2.1）**：
bf16 rel 5e-3 → 7.8e-3；新增算子级覆盖机制（operator_overrides：
仅当参考自身偏差实测超过默认档时登记，须附归因证据；case 级覆盖
仍然禁止）。回归钉见 tests/test_judge.py（bf16 容纳 1 ULP、覆盖档
优先级、真错仍拦）。真错实现的误差在 1e-1 以上，不会被 1e-3 档放过。
下轮真机复核预期：C1 的 4 个静默错误清零——仍出现即为回归。

**首次真机测速的两个发现（2026-10-12，standard-window，scorer 0.2.2）**：

demo 提交此前在真机上只跑过 `--limit 5`，从未见过全量题库；首个测速
窗口（standard 档全量 240 题）暴露两件事——都被评测体系按设计抓住：

1. **demo 的 7 个静默错误全部根因于非连续输入**（v0.4 内存布局维度，
   `contiguous=false` 的用例）。demo 覆盖的 8 个算子里共 12 个非连续
   用例：11 个失败（7×WRONG_RESULT + 4×RUNTIME_FAIL），唯一例外
   masked_scale-f16-S203 与 stride-aware 的 transpose（4/4 过）。
   归因：exp/gelu/maximum/masked_scale/rms_norm/swi_glu 的内核按
   连续内存线性寻址（`ptr + offs`），非连续输入直接读错内存——
   误差量级 rel 1e+15~1e+32（不是容差问题，是真读错）。C1（纯
   Python，经 torch 索引天然按步长）在同批用例全过，排除参考侧。
   **修复**：六个内核的包装函数加非连续守卫，明确报
   NotImplementedError（与 demo 对 gelu erf 模式/softmax 维度的既有
   声明风格一致）。不做 `.contiguous()` 绕道——布局搬运调用仅布局类
   算子可用（反作弊规则，demo 必须身体力行它教的规则）。完整的
   stride 寻址内核留作后续（需真机迭代验证，盲写风险大）。
   **修复后真机复验（同日 18:08，run-2）**：demo 静默错误 7 → **0**，
   被守卫的用例全部转 RUNTIME_FAIL（声明式未实现）；masked_scale-
   f16-S203 由此前"碰巧对"转诚实报未实现（守卫比旧实现更严，正确）。
   语料的内存布局维度首战告捷——它抓住了评测方自己样例实现的
   真实能力缺口。
2. **亚线性锚点的方向反转**（scorer 0.2.2 修复）：transpose 全对
   → 首个真实锚点冻结为 0.14x（demo 内核比官方框架慢约 7 倍，几何
   平均 20 用例实测）。旧公式 `100·ln(P)/ln(锚点)` 在锚点<1 时两
   个对数均为负——快 2 倍算 64 分、慢 2 倍算 135 分。改为统一公式
   `100 + 100·ln(P/锚点)/|ln(锚点)|`（锚点>1 代数恒等，锚点<1 方向
**compare 命令的真机验证与随之抓到的第三个缺陷（同日 18:23–18:29）**：

同窗口两轮 standard（run-1 vs run-2）首次真机跑 `compare`，五个部分
全部正常：身份对照、分数对照带 Δ（静默错误 7→0 一眼可见）、功能回归
清单（masked_scale-f16-S203：上轮"碰巧对"→本轮守卫报未实现，线索带
着报错原文）、性能变化清单（±10% 门槛）、缺失用例核对（0）。随之抓到：

1. **环境指纹误报"跨环境"**：同机同卡相隔 47 分钟的两轮被判指纹不同。
   根因：指纹哈希把整个 env 字典算进去，含 probed_at（探测时间戳，
   每轮必不同）与 venv_lock_hash（pip freeze 哈希，每次 pip install
   抖动）。与 C12 的 frozen_at 同源缺陷——指纹必须只覆盖身份不覆盖
   时间。修复：FP_VOLATILE_KEYS 排除清单（probed_at / venv_lock_hash /
   npu_smi_info_text 原始读数），回归钉
   test_fingerprint_ignores_volatile_fields。
2. **51 个用例加速比"显著回归"中的真信号**：分解显示全部是**被测实现
   侧**变慢（exp 1µs→16µs、masked_scale 1µs→17µs；参考侧稳定、未加
   守卫的 transpose 仅慢 1.3 倍）——即 demo 非连续守卫的 Python 开销
   在微秒级小用例的计时里被放大。诚实记录：修静默错误的代价是这些
   微秒级用例的计时变慢（包装层开销本就是被测计时的一部分）；demo
   是教学样例，正确性优先。µs 级的具体机制（为何一个 is_contiguous
   检查在 torch_npu 上值 ~15µs）待下个窗口微基准钉死。compare 把
   "哪一侧动了"直接摆上台面——这正是它的用途。

   另：两轮 P_geo 0.93→0.37、不稳定比例 0.13→0.32——同一份未守卫
   算子（transpose）仅漂移 5.7%，说明大头是守卫叠加测量噪声，不是
   环境突变（时钟未锁定，波动区间如实披露）。

   正确；追平框架 = 200 分）。回归钉见 tests/test_scorer.py 的
   `test_sublinear_anchor_direction_not_inverted`。真机实测的裸值：
   transpose-X49 impl 13.9ms vs ref 0.09ms、L07 3.15ms vs 0.04ms——
   按算子性能分机制端到端走通（自锚 100、按算子出分、无分原因披露）。
   **修复后真机复验（同日 18:08，同窗口第二轮）**：同一 demo 复跑，
   transpose 几何平均 0.1419x → 0.1338x（漂移 −5.7%，测量噪声），
   机器按新公式对锚点评出 **97.02**——与本地用归档裸值独立复算
   完全一致；旧公式会得 **102.98**（跑慢了反而更高）。方向修正
   经真实数据三重验证：单元钉 + 真机 score + 归档裸值复算。

**覆盖档查找的名字归一化缺陷（2026-10-12 真机复核抓到并修复）**：

2026-10-12 真机复核（新机器自举 + 容差 1.1 代码）结果：C1 静默错误
**4 → 1**，gelu 两例与 swi_glu bf16 例全部转 PASS（修复生效），但
`grouped_matmul-f32-M18` 冤案**复发**（abs 2.44e-4，明明在覆盖档
abs=1e-3 之内）。归因：覆盖档查找只做 `operator.lower()`，而评测记录
里的算子名是注册表**类名**——`"Gelu".lower()` 恰好等于 yaml 键
`gelu`（碰巧命中），`"GroupedMatmul".lower()` 得 `groupedmatmul`
≠ yaml 键 `grouped_matmul`（错失覆盖档，回落默认档被错杀）。
单测当时用算子 ID（slug）喂入，两种写法在 gelu 上无法区分，所以
全绿但真机穿帮。

- **修复**：`ToleranceTable._op_keys` 生成候选键（全小写 / camel→snake /
  去下划线），注册与查询两侧都按候选展开，类名与算子 ID 任意组合
  都能命中；未登记覆盖的算子仍走默认档（不因归一化而误放宽）。
- **回归钉**：tests/test_judge.py 的
  `test_operator_override_matches_class_name`——用真机观测的 M18
  原始数值（rel 5.3718e-3 / abs 2.4414e-4）+ 类名 `GroupedMatmul`
  重放，修复前必挂；另钉住 gelu 类名形式与未覆盖算子的默认档。
- **下次真机复核预期**：C1 静默错误清零（217 PASS / 12 COMPILE_FAIL
  / 10 RUNTIME_FAIL / 0 WRONG_RESULT）。仍非 0 即为回归。

---

## 附：2026-10-13 v0.5 首轮真机流水暴露的缺陷与平台发现

验证记录 `6844fd88a482075088cbf377-1013-192217`（910B4 / CANN 8.5.0 /
torch_npu 2.9.0 / triton-ascend 3.2.0）：21 项判定 **18 ✓ / 3 ✗**，真机
全量测试 502 过，**C1 静默错误 0**（核心回归断言在 v0.5 上成立）。三个
✗ 全在本轮新增步骤（5c 融合冒烟、5d 探针实证），逐条归因如下。顺带
结论：RoPE 的 tl.sin / tl.cos 在该平台**编译可用**——原待办④有答案，
其用例的 WRONG 全部归因于样例内核寻址缺陷，不是内建缺失。

### 平台发现三条

1. **1 维 grid 上限 < 65536**。朴素 GEMM 探针 (1024,1024)×N=64 恰好
   65536 个 program，压线即炸
   `RuntimeError: grid should be less than 65536!`（运行时自提示
   `TRITON_ALL_BLOCKS_PARALLEL=1` 可绕开）。处置：probe_validation.py
   自动设该变量 + 逐形状隔离（一个形状炸只废该形状的实证）；「每输出
   元素一个 program」的大 grid 内核形态在该平台不可用，评测资产与
   提交方都需知晓。
2. **统一缓冲区（UB）瓦片预算 1572864 位（192 KiB）**。masked_softmax
   样例内核 BLOCK=1024 的 f32/i8 瓦片需 2097152 位，BiShengHIR 编译期
   报 `ub overflow`——f16/bf16 塞得下，所以同一内核的半精度用例侥幸
   通过、f32 全炸。瓦片预算与 dtype、multi-buffer 流水深度相关。处置：
   BLOCK=256（3× 余量）；段和样例内核 2D 路径 [1024,1024] 瓦片
   （4 MiB）同病，重构为行块 256 × 列块 64（未上真机即提前对正）。
3. **f32 tl.dot 与 torch-npu matmul 的数值分歧随 K 增长**。三个融合
   GEMM 算子的大 K f32 用例 WRONG（grouped_matmul_swiglu [2048,2048]
   rel 2.0% / transpose_linear [2048,2048] 近零输出 rel 21% /
   rms_norm_linear [64,65536] rel 0.20%），小 K 同族用例全过；本地
   纯 f32 复刻内核数学与参考差异为 0（逐位核对）——分歧在编译器/
   框架的数值路径，非内核逻辑。归因探针 `scripts/dot_precision_probe.py`
   （K × 分块扫描）待真机执行，据其结果按既有机制登记算子级容差覆盖
   （附归因证据）或调整内核写法；**登记前这三个用例预期仍判 WRONG**
   （如实暴露，不预防性放宽）。

### 评测方资产缺陷一条（真机炸出，已修 + 回归钉）

4. **MoE 参考的 row_idx 建在 CPU**。`torch.arange` 不带 `device=`，
   与 NPU 实现输出比对时抛设备不匹配；该异常从 `execute_case` 的
   比对段（唯一没被 try 包住的段）逃出，**杀死评测子进程**，后续
   15 个用例全部陪葬补记（监督者补记防线本身工作正常）。三处修复：
   参考显式 `device=`（版本内变更：锚点未冻结、清单哈希不含 ref 文件）；
   evalchild 逐用例异常兜底（评测侧异常只废本用例、判 ORACLE_FAIL，
   不再整轮陪葬）；meta 设备纯度测试钉住「参考输出必须与输入同设备」
   这一缺陷类（CPU 本地测试暴露不了，meta 保留设备语义可精确复现）。

### 样例提交缺陷三类（波及 7 个内核；本地模拟与真机误差逐位吻合后修复）

5. **行/线分解惯用法错误**（apply_rotary_pos_emb / moe_gating /
   add_rms_norm / masked_softmax 共患）：program id 被按**含最内维**的
   3 维形状分解、最内步长取 `a[3-rank]`——rank-1 取到填充 0（1000 个
   车道读同一元素，AddRmsNorm 6/6 WRONG）、rank-2 行基址算成平面元素
   号 p 而非 p·stride0（RoPE 5/6 WRONG；唯一过的用例是 zeros 值域，
   读错行也全零，纯属侥幸）。修复为「外维/非归约维分解 + stride(-1)」
   惯用法；AST 从提交内核提取真实助函数的回归测试钉住不重不漏覆盖
   （tests/test_demo_kernel_addressing.py，不经 import 杜绝测试漂移）。
6. **融合段语义未舍回输入精度**（rms_norm_linear /
   grouped_matmul_swiglu）：组合参考 = 原语参考顺序调用，原语出口是
   `.to(x.dtype)`——中间张量本来就是输入精度的；样例内核全程 fp32
   不舍回，半精度用例全错（bf16 rel 3.9 / f16 abs 0.25，模拟逐位
   复现，舍回后归零）。内核注释里"参考是 fp32 全程"的旧判断已更正。
7. **段和样例内核 rank-1 输出形状错**：返回 (num_segments, 1) 而参考
   是 (num_segments,)——冻结用例 9/10 是 rank-1，本会全判错；真机窗口
   前自查发现并修复（连同第 2 条的 2D 瓦片超限）。

**方法论沉淀**：开发机无 NPU，但「在 CPU 上复刻内核数学（含量化点与
寻址）与参考对拍」被证明可以**逐位**归因真机误差（模拟值与真机实测
完全一致到 1e-4 位）；修复前先模拟复现、修复后模拟归零、再上真机
复核——不再拿真机窗口试错。

---

## 附：第二轮真机定向复验（记录 6844…-manual-1013-225653）——修复验证与新缺陷

**首轮修复在真机上的验证结果**：AddRmsNorm 6/6、ApplyRotaryPosEmb 6/6、
MaskedSoftmax 8/8、MoeGating 5/5 全 PASS（行/线分解与 UB 修复成立）；
TransposeLinear 只剩已知 f32-X48。同时暴露五个新缺陷，全部归因并修复：

1. **triton 循环作用域**（GroupedMatmulSwiGlu 6/6 COMPILE_FAIL，
   `NameError('a is not defined')`）：循环体内首次赋值的变量是循环
   作用域，循环外引用直接 NameError——上轮修复在循环外引用了循环内
   加载瓦片 a 的 .dtype。修复：舍回目标一律取**指针元素类型**
   （`x_ptr.dtype.element_ty`）。
2. **RmsNormLinear 舍回未生效**：半精度四用例误差与未修轮**逐位相同**
   ——`.to(a.dtype).to(tl.float32)` 的舍回在该后端疑似无效（两个候选
   机制：掩码加载的返回 dtype 不可信；f32→输入精度→f32 的对被折叠）。
   修复为不可绕过的形式：舍回后**以输入 dtype 直接参与 tl.dot**（舍入
   成为 dot 操作数类型的编译期硬性要求）。哪 个候选机制为真由下轮
   真机裁决（RML 半精度用例转 PASS 即指针 dtype 机制）。
3. **段和内核两处**：整型标量累加用 `tl.zeros((1,))` 是 1 元素块，存
   进标量指针报 "Value argument cannot be block type if pointer
   argument is not a block"（i32/i64 用例 COMPILE_FAIL）→ 改 Python
   标量累加；2D 瓦片 256×64 经 multi-buffer 需 2731264 位、仍超 UB
   192 KiB（X92/M63）→ 减半为 128×32。
4. **probe_validation 计时缺陷**：内核启动异步而计时未做设备同步——
   第二轮全部形状测得 ~0.11ms 的提交开销常数（与负载无关），**该轮
   探针计时数据全部作废**、9 条原子的披露口径维持「实证待执行」。
   修复：_time 每次调用后显式 synchronize。
5. **三个探针资产缺陷**：static_unroll 把 tl.static_range 用在**运行期
   K** 上（前端要求编译期边界，CompilationError）——该原子考察的本
   就是组循环（group_list ≤ 8 的编译期常量），重设计为组循环展开；
   scan_blocked 的 torch_ref 用了单参 torch.cummin（该 torch 版要求
   显式 dim）→ cummin(x, 0)；split_k 对拍在长 K（65536）求和顺序噪声
   下对 1e-4 必然失败 → GEMM 族探针 CROSS_TOL 文档化放宽到 5e-3
   （附实测依据）。另：dot 归因探针崩溃根因是内核定义在 main() 内、
   拿不到模块级 tl——已模块化到 probes/dot_precision.py。

**Class B（f32 大 K 分歧）理论修正**：tile_gemm 探针 (2048,2048) 的
同款分块 dot 与 torch matmul 在 1e-4 内一致——**排除「tl.dot 精度
损坏」的强假设**；三个融合用例的分歧定位为**求和顺序噪声随 K 增长、
近零输出被抵消放大**（两段 GEMM 经 silu·b 级联的 GMS 放大最甚）。
dot 探针（K × BK 扫描）下轮出数后，按既有机制登记算子级容差覆盖
（附归因证据）或调整 demo 内核分块；登记前三个用例如实判错。

---

## 附：第三轮真机定向复验（记录 6844…-manual-1013-234810）——归因定案与覆盖登记

第二轮修复验证：GMS 编译修复生效（x_ptr 指针元素类型）、**RML 半精度
三例全部转 PASS**（「以输入 dtype 参与 tl.dot」的舍回方案在真机证实
有效）、段和 2D 瓦片减半后编译通过。同时三轮数据把两个悬案定案：

1. **舍回对消除坐实（平台缺陷）**：GMS 以指针元素类型为目标仍逐位复现
   未舍回的误差（f16-M09 abs 0.25）——该后端静默消除
   f32→输入精度→f32 的舍回链。RML 的 dot-operand 方案有效但 silu 段
   是逐元素、没有 dot 可挂。处置：GMS 改**两段式**（段 1 矩阵乘把
   中间结果以输入精度落 scratch、段 2 读回做 silu·b）——显存往返不被
   优化器消除，是唯一保证段语义的形式；C13 已定案多内核合法，中间
   落盘的性能代价由评分测量，正是该维度要考察的差距。
2. **f32 大 K 分歧定案为求和顺序噪声，登记 B 类容差覆盖**：dot 归因
   探针（本轮产出）实测 K≤1024 与 torch **逐位一致**（dot 是 f32 真精
   度）、K=2048 分歧 max_abs 1.8e-4、K=65536 分歧 4.7e-3，且 BK=32/128
   **逐位相同**——与分块无关，实现侧无抓手。按 C2 机制登记四条覆盖
   （tolerance 1.2 / scorer 0.2.3）：transpose_linear 1e-3、
   rms_norm_linear 1e-2、grouped_matmul_swiglu 5e-2、
   unsorted_segment_sum 1e-3；重放钉钉住真机实测值转 PASS、10× 真错
   仍拦、半精度不受影响。
3. **UB 多缓冲规律（比第二轮更完整的图景）**：迭代数低于阈值的内层
   循环被流水线全量多缓冲，预算 ≈3×迭代数×瓦片位数（段和 [4096,64]
   32 次迭代需 6361344 位）；迭代数高时（[262144,16] 2048 次）流水线
   放弃多缓冲反而通过。同轮 tile_gemm 探针 [32,32] 瓦片 64 次迭代
   正常——触发条件与瓦片大小/迭代数之比有关，非单纯迭代数。处置：
   段和 2D 路径改**树归约**（段 1 行维上 grid 无内层循环 + 段 2 小
   瓦片合并），并把 3D 网格总量压到 32768（远离 1 维 65536 上限——
   该上限按 naive 探针实测恰好 65536 即拒）。
4. **段和整型 rank-1**：Python 标量 0 是 int32、循环内 int64 累加报
   loop-carried 类型不一致（第三轮实测）→ 1 元素块累加器 + 块指针
   存回（块值配块指针，两轮报错形式各对一头）。
5. **探针首份真实数据**（计时同步修复后）：6 条性能负载原子**实证
   达标**（tile-gemm / vec-load / vec-broadcast / multipass-reduce /
   scan-blocked / coalesce-gather，1.5×~3.3×）；STATIC-UNROLL 1.02×、
   MASK-TAIL 达标占比 50% → 如实「声明性考察」；split_k 对拍仍失败
   （K=65536 顺序噪声超 5e-3）→ 改**确定性两段式**（弃原子合并）+
   CROSS_TOL 按实测依据放宽 1e-2，待下轮重验。
6. **流程教训**：本轮命令的 git pull 与修复推送赛跑（相差 <1 分钟）
   拉到旧代码、白跑一轮的前半——同环境连续复验时**开跑前先
   `git log --oneline -1` 核对**；归档会把 runs/run-* 全部打包，
   分析前先核对 run 编号（run-1 可能是上一轮的残留目录）。

---

## 附：第四轮真机定向复验（记录 6844…-manual-1014-093413）——全绿在望

- **两段式与覆盖登记全部生效**：GroupedMatmulSwiGlu 6/6、RmsNormLinear
  6/6、TransposeLinear 6/6 全 PASS。GMS 的 f16-M09 **不带覆盖在默认档
  内通过**——显存往返的段语义舍回是精确的；三个 f32 大 K 用例由
  B 类覆盖档吸收（重放钉保证真错仍拦）。
- **段和 9/10**：唯 [4096,64] 的 2D 路径第三次撞 UB——行循环瓦片
  （256×64、128×32）与无循环树归约（[512,16]）三种设计分别需
  6.4M / 6.4M / 8.4M 位，BiShengHIR 对 2D 瓦片的多缓冲预算不可控；
  而一维 [1024] 车道循环同负载始终在预算内（rank-1 全部用例两轮
  全过）。定案：2D 路径改用 rank-1 同构的逐（段，列）一维循环——
  用被验证的结构，不与预算黑箱搏斗。
- **探针收益实证收官**：split_k 确定性两段式重验 **3.5× 达标**；
  最终 9 条性能负载原子 **7 条实证达标**（1.5×~3.5×）、2 条如实
  声明性（STATIC-UNROLL 1.02×、MASK-TAIL 50%）。dot 归因数据与
  第三轮逐位复现（确定性确认）。
- **流程**：本轮命令的 `git log --oneline -1` 核对生效，未再发生
  拉取赛跑。

---

## 附：第五轮真机（记录 6844…-manual-1014-093413）——段和收官与 v0.5 锚点冻结

- **段和 10/10**：rank-1 同构的逐（段，列）一维 [1024] 车道循环一次
  通过——[4096,64] 这个三连炸的形状落在已验证结构上直接成立。该形态
  的最终结论：**该平台的 2D 瓦片循环 UB 预算不可控（三种设计 6.4M~
  8.4M 位全爆），一维车道循环同负载始终安全**——已沉淀为平台内核
  工程规律。
- **v0.5 首个全量测速轮（289 用例，静默错误 0）冻结 10 个算子锚点**：
  9 个可判融合算子全体 + transpose 原语，当轮全对自锚 100。锚点值
  （对框架加速比 geo）：transpose 0.1352x（与 10-12 窗口 0.1419x 交叉
  复现，漂移 ~5%）、TransposeLinear 0.1829x、GatherExp 0.2442x、
  MoEGating 0.2521x、MaskedSoftmax 0.3524x、RmsNormLinear 0.4772x、
  GMS 0.5778x、RoPE 0.6134x、AddRmsNorm 1.2502x、段和 **11.1529x**
  （超线性——框架侧逐段掩码循环参考被并行内核反超；评分器 0.2.2 的
  统一公式对超线性锚点方向正确）。其余 15 算子无分并披露原因
  （6 未提交：ArgMax/CrossEntropyLoss/Cummin/Gather/GroupedMatmul/
  Scatter；9 已提交未全对：4 类声明式守卫生效的原语 + LinearGelu
  6 个 erf 用例全不支持 + MaskedGelu 3 个 erf 用例未全对——逐算子
  明细见第六轮附录的按证据清点，此前本报告把 9 个笼统写成"带守卫的
  原子"、把 2 个 erf 融合误计入"未提交"，均已更正）。**性能赛道在
  v0.5 上端到端成立：全对→自锚→出分→无分有因。**
- **原语 demo 内核的既有边界（非本轮工作，下阶段维护项）**：3 个
  原语 f16-XL 用例编译失败（softmax-X16 / rms_norm-X32 / swi_glu-X16，
  疑与融合轮同源的 UB 预算问题）；声明式守卫明细（第五轮测速归档
  逐条可查）：erf 模式 gelu/linear_gelu/masked_gelu、非末维
  softmax/swi_glu、非连续输入的 maximum/masked_scale/exp/rms_norm
  ——按算子锚点机制只影响各自算子，已在 score 表披露。
- **锚点连续性**：锚点由数据库历史轮推导，环境清空即失——同语料+
  同提交的下个全对轮重新冻结同量级锚点（transpose 两窗口实证漂移
  ~5%）；精确复用历史锚点可把归档 results.jsonl 导回数据库（import
  子命令）。手册 6e 已记录。

## 附：第六轮真机全量（记录 6844…-1014-170458，首个含 5e 的窗口）——审计项真机证据与两项归因

判定 21 ✓ / 2 ✗；真机测试 525 通过 / 3 跳过（本机基线 495/33）；探针
结论 7 实证 / 2 声明性与前两轮逐条一致；**清单跨机可复现 ✓**（aarch64
+ torch 2.9.0a 环境重跑生成脚本，输出「内容未变，未重写」，清单哈希
2771f8b88970… 与冻结版逐字一致；本机构建事件行随记录归档于
step-5e-npubuild-events.jsonl）。两个 ✗ 的归因：

**✗ 之一：5c 预期口径缺陷（评测方资产缺陷，非 demo 内核缺陷）**

- **现象**：融合冒烟分析判 MaskedGelu 的 5 个 PASS 用例为「显性不支持
  算子的非 RUNTIME_FAIL」。
- **归因**：5c 分析脚本把 `unsupported = {LinearGelu, MaskedGelu}` 整
  算子当「全部用例须 RUNTIME_FAIL」。实际 demo 的 MaskedGelu 内核
  **实现了 tanh 近似路径**（tl.exp 表达，内核 docstring 明示），语料中
  5 个 `approximate="tanh"` 用例自第一轮起就 PASS；3 个无 approximate
  属性的 erf 精确用例才显性 NotImplementedError。LinearGelu 的 6 个
  用例全部 erf 模式、全 RUNTIME_FAIL（预期正确部分）。
- **为何四轮未暴露**：第一轮全量窗口同样 5 PASS / 3 RUNTIME_FAIL
  （归档 run 数据可查），但当时 ✗ 主因是一批真缺陷，此项被淹没；
  第二至五轮走 6d-3 定向命令，不经过 5c 的判定。代码与语料自第一轮
  起未变——**缺陷在预期口径，不在被测对象**。
- **修复**：5c 分析改为按冻结清单逐用例判模式（results 记录不带
  attrs，模式从 manifest 取）：erf 精确用例 → RUNTIME_FAIL，其余融合
  用例（含 MaskedGelu tanh 5 例）→ PASS。修正后逻辑对第六轮真实
  results.jsonl 复算 rc=0。受此口径错误影响而过期的文档表述（playbook
  已知边界、方案 §6 三处、覆盖度报告的 15 算子披露段）已同步更正——
  覆盖度报告原把 erf 两个融合算子误计入「未提交」（实测未提交的是
  ArgMax/CrossEntropyLoss/Cummin/Gather/GroupedMatmul/Scatter 六个），
  并把 9 个「已提交未全对」笼统写成原语守卫。

**✗ 之二：cross_entropy_loss 在 npu:0 同进程两遍不一致（真平台发现）**

- **现象**：5e 首跑的 NPU 参考自检 286/289 通过，3 个用例
  `cross_entropy_loss-f32-L31 / f32-L40 / i64-M19` 报「参考实现不确定：
  两次执行结果不一致」（同一输入张量原地复算，torch.equal 严格相等
  判据）。整型用例的实际计算仍在 fp32 域（该参考固定升 f32 计算，
  dtype 归属约定见 ref.py 头注），三个用例的共性是 mean 归约。
- **定性**：torch_npu 的 `F.cross_entropy` 归约路径存在执行间不确定
  性——与本模块 docstring 记录的 v0.3 时代教训同类（当时两个整型
  用例在昇腾上两遍不一致且已入冻结集，冻结机无 NPU、门禁拦不住；
  5e 的设计目的正是把这类发现从冻结后抽查提前到每个验证窗口）。
- **对判定的影响**：无实证影响——历轮（包括本轮 C1 正控制）CE 用例
  全部 PASS，单次执行比较在默认容差内稳定落点；两遍一致是比判定更
  严的锚点质量口径。
- **处置路径**：refcheck 诊断已升级为不一致时附带 max_abs/max_rel
  实测量级（回归测试钉住）；**下个窗口 5e 输出即带量级数据**，据其
  按既有覆盖机制登记（量级在容差内 → 登记为已知的锚点自噪声平台
  属性，5e 预期改为「除已登记例外零失败」；量级超容差 → 参考侧
  归因重写）。登记前 5e 在该 3 例上如实保持 ✗。

## 附：第七轮真机全量（记录 6844…-1014-224625）——CE 锚点自噪声定案登记

判定 22 ✓ / 1 ✗（唯一 ✗ 即本条目的量级测量本身）；5c 预期口径修正
验证通过（erf 精确用例 RUNTIME_FAIL、tanh 用例 PASS 的分判生效）；
真机测试 526 通过 / 3 跳过；跨机可复现与探针结论复现。

**CE 锚点自噪声的量级定案（第六轮发现的处置完成）**：

- **量级数据**（refcheck 诊断附带的实测量级，两遍不一致判据仍为
  torch.equal 严格相等）：第七轮 6 例全部为 cross_entropy_loss——
  f32-L31（max_rel 6.4e-8）、f32-L40（7.0e-8）、f32-M21（6.9e-8）、
  i32-M27（9.1e-8）、i32-S06（9.2e-8）、i64-M19（6.4e-8）。
  max_rel 全部在 6~9×10⁻⁸（f32 的 ULP 量级）；max_abs 随损失幅值
  变化（4.8e-7 ~ 2.0e-3）。
- **随机性**：两窗口失败集合不同（第六轮 3 例、第七轮 6 例，交集
  3 例）——归约内核的执行间噪声是概率性的，不固定于特定用例；
  登记口径因此按算子 + 量级界，不按用例清单。
- **对判定的余量**：判定容差为 rel/abs 析取（容差 1.1 起），f32 档
  rel 1e-4；实测锚点噪声 rel ≤ 9.3e-8，低三个数量级以上。历轮
  （含 C1 正控制与 demo 提交）CE 用例判定全部 PASS，与此吻合。
- **登记**：5e 的 NPU 参考自检改为「除已登记锚点自噪声外零失败」——
  已登记类 = cross_entropy_loss 的两遍不一致且 max_rel ≤ 1e-6（界取
  实测最大值的 10 倍、判定容差的 1%）。**超界、或出现在其他算子上，
  仍按新发现报出**（分类逻辑已用第七轮真实数据与两个反例本地验证：
  超界 rel、非 CE 算子均正确进未知类）。
- **不动的部分**：冻结门禁（CPU）的严格相等判据不变——NPU 窗口
  噪声类发现本就只能由 5e 暴露；spec/tolerance.yaml 不加档——噪声
  在默认容差内三个数量级，无需覆盖。

## 附：第八轮真机全量（记录 6844…-1015-105622）——首个全绿，登记放行逻辑上机验证

23 ✓ / 0 ✗；真机测试 539 通过 / 3 跳过；探针 7/2 与跨机可复现结论复现。
**5e 的"算子 + 量级界"登记放行逻辑首次在真机执行**：本轮抽到 2 例 CE
两遍不一致（f32-L31 max_rel 1.282e-7、i32-M27 9.113e-8）——与前两轮
均不同的第三个随机集合，进一步佐证概率性噪声定性；两例量级均在登记界
1e-6 的 10 倍余量内，全部正确进已登记类，未知类零失败，判定 ✓。
三窗口量级观察：6.4e-8 ~ 1.28e-7，稳定低于登记界一个数量级——界取
实测最大值 10 倍的余量设计经受住第三窗口检验。
