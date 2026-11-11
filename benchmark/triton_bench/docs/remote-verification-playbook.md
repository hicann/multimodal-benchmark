# 昇腾机闭环验证手册（zip → 结果 全链路）

适用对象：已在昇腾机上拉好 triton-eval 仓库、且评测环境（CANN + venv +
triton-ascend）完整可用的机器。按顺序逐条执行，每步都有「预期」——
**实际输出与预期不符的地方就是缺陷，记下来**。

**硬约束（2026-10-12 实测）：真机环境每两小时自动清空一次目录。**
一切流程都按"窗口"组织——默认验证流程约 1 小时，一个窗口内跑完并自动
归档；standard 档全量测速（25-45+ 分钟起）放不进验证窗口，**单独开一个
窗口、跑专用测速脚本**（见 §6e）。锚点按算子持有，两个窗口测过同一
提交即可用 compare 对比。

本轮要验证的闭环（2026-10 新架构）：

```
zip 包 ──ingest──> 内容寻址目录 ──run──> 监督者(父进程)
                                        ├─ 静态校验（不 import 提交代码）
                                        ├─ 子进程 evalchild：加载→篡改校验→
                                        │   环境一致性→探针→逐用例执行→判定
                                        │   （事件流 events.jsonl 回传）
                                        └─ 硬超时/崩溃归因/缺失补记
                     → results.jsonl + logs/ + integrity.json + artifacts.json
                     → score → report
```

约定：以下命令都在**仓库根目录**执行；`<VENV>` 换成你的评测 venv
（sync-and-run.sh 里默认是 `~/venv/te-compiler`）。

```bash
source env/env.sh
source <VENV>/bin/activate
export PYTHONNOUSERSITE=1
```

---

## 1. 拉取最新代码（每轮验证前都做）

**空机一次性自举**（环境被清空后，克隆 + 拉最新 + 全链验证 + 自动归档
上传，粘一次跑完；推送凭证由脚本自动配置，无需手工 set-url。
**总时长约 1-1.5 小时（v0.5 起含融合冒烟 5c、探针实证 5d、审计真机复核
5e 三步）**，一个
两小时窗口装得下**——不含 standard 测速，那要
单独开窗口跑 scripts/standard-window.sh，见 §6e。无论脚本怎么退出——跑完、Ctrl-C、
断连——收尾都会自动把已有证据归档上传，中断也不丢）：

```bash
cd ~ && { [ -d triton-eval ] || git clone https://gitcode.com/hkhkhkzxk/triton-eval.git; } \
  && cd triton-eval \
  && git pull --rebase --autostash origin main \
  && bash scripts/remote-verify.sh
```

分步说明：

```bash
# 远程机上还没有仓库：先克隆
git clone https://gitcode.com/hkhkhkzxk/triton-eval.git && cd triton-eval

# 已有仓库：拉最新 + 重装本体（本轮改动：v0.5 语料 25 算子/289 用例 +
# 融合冒烟与探针实证两步接入验证脚本）
git pull origin main
pip install -e .

# 确认拿到的是新代码（按算子锚点模块应存在）
ls src/triton_eval/anchors.py && git log --oneline -1
```

## 2. 全量测试（含 NPU 用例——本机跑不了的那批现在应该全过）

```bash
pytest -q
```

预期：全部通过，跳过数远小于本地的 33（本机跳过的是 npu/triton 标记）。
已本机验证的基线：509 passed / 33 skipped（无 NPU 环境，含按算子锚点、
compare、近零容差回归、bf16 1-ULP 与算子级覆盖回归、融合参考组合等价、
跨维组合合法性、剔除记录防篡改、演化对照与对账、生成日志、参考不确定
性量级诊断、发布机制（标签命名/五轴快照/勘误解析/发布说明/回溯定位对
真实 git 历史）等用例）。
真机比本机多跑 npu/triton 标记用例，通过数应更高（2026-10-15 实测
539 对应本机 509）。

## 3. 包级防线（zip 摄取规则）

```bash
python scripts/adversarial_packages.py
```

预期判定（逐行对照 `rule_id`）：

| 包 | 预期 |
|---|---|
| good.zip / nested.zip | 通过（nested 剥离外层目录） |
| bomb.zip | 拒 SUB-PKG-005（压缩比 1025 倍） |
| escape.zip / absolute.zip | 拒 SUB-PKG-006（路径逃逸/绝对路径） |
| symlink.zip | 拒 SUB-PKG-007（符号链接条目） |
| payload.zip / pycache.zip | 拒 SUB-PKG-008（.so / .pyc 不是纯源码） |
| fat.zip | 拒 SUB-PKG-003（72MB > 64MB 总量） |
| nomani.zip | 拒 SUB-PKG-009（缺 manifest） |
| d7-oracle-poison.zip | **包级放行**（内容防线交给第 5 步验证） |

## 4. ingest 子命令（解包 + 完整校验，含加载）

```bash
python -m triton_eval ingest --package /tmp/triton-eval-pkgs/good.zip
```

预期：解包到 `runs/ingest/<sha12>/`，打印 sha256、文件数、声明算子，
校验通过（demo 提交在完整环境下应过静态+加载）。
`--db runs/triton-eval.db` 里多一条 submission_record，
`package_name=good.zip`、`package_sha256` 有值。

## 5. 核心验证：zip 直进 run → 出结果 → 打分 → 报告

```bash
# 5a. zip 直进（自动摄取，无需先手动 ingest）
python -m triton_eval run --profile smoke \
    --submission /tmp/triton-eval-pkgs/good.zip --limit 5

# 5b. 打分 + 报告
python -m triton_eval score --run latest
python -m triton_eval report --run latest
```

预期：5a 退出码 0，开头打印「已解包 good.zip → …」，逐用例打印
`exp-f32-00: PASS` 一类；5b 给出 OpScore（demo 提交 8/14 算子，
Exp/Softmax 等 PASS；报告里「提交覆盖 8/14」）。
冒烟档不测性能：5b 的「性能得分」一行应显示**不可用**，随后
「按算子性能分」表逐算子给出「—」与原因（无性能数据）——这是
按算子锚点的新口径，「为什么没分」必须可见。要出性能分需跑
standard/official 档。

```bash
# 5c.（本轮新增命令的实测）两轮对比：拿 5a 与第 6 节任意一轮
python -m triton_eval compare --run latest --against latest
```
预期：**拒绝**（「--run 与 --against 不能是同一轮」）。改用两个真实
轮次编号（`ls -td runs/run-* | head -2` 取最新的两个）再跑一次，
预期退出码 0、打印功能回归/修复与性能变化计数，报告落在
`runs/reports/compare-run<N>-vs-run<M>.md`。注意两轮须同一版题库。

产物检查（每项都必须存在且非空——`log_path` 曾经是空转的，P3-2）：

```bash
RUN=$(ls -td runs/run-* | head -1)
ls $RUN                      # results.jsonl events.jsonl env.json
                              # integrity.json artifacts.json logs/
head -3 $RUN/events.jsonl    # 事件协议：meta / env_check / probe / case / done
cat $RUN/integrity.json      # 环境一致性 + 子进程画像（应为 ok、无篡改）
python - <<'EOF'
import json, pathlib
arts = json.loads(next(pathlib.Path("runs").glob("run-*/artifacts.json")).read_text())
print(len(arts["files"]), "个产物，全部有 sha256:", all(f["sha256"] for f in arts["files"]))
EOF
grep -l "\[eval\]" $RUN/logs/*.log | head   # 用例日志有结构化尾注
```

## 6. 对抗提交的执行期防线

```bash
# 6a. D7：import 期投毒 torch.softmax（此前「全零也 PASS」）
python -m triton_eval run --profile smoke --limit 3 \
    --submission examples/adversarial-submissions/ADV-D7-oracle-poison
```
预期：**rc=1，run 判 INVALID**，校验错误为
`SUB-EXE-005 提交在 import 期替换了关键接口：torch.softmax`。
这是本轮最重要的断言——判定权被劫持的路径已关死。

```bash
# 6b. E3：import 期 raise SystemExit(0)（此前「静默消失、退出码 0」）
python -m triton_eval run --profile smoke --limit 3 \
    --submission examples/adversarial-submissions/ADV-E3-systemexit
```
预期：rc=1，结构化报错（SUB-LOD-005 提交试图终止进程），**不**是无输出消失。

```bash
# 6c. F4：吞掉超时信号（此前「超时照常 PASS」）+ 监督者硬超时
TRITON_EVAL_CASE_TIMEOUT_S=5 TRITON_EVAL_IDLE_MARGIN_S=5 TRITON_EVAL_GRACE_S=3 \
python -m triton_eval run --profile smoke --limit 3 \
    --submission examples/adversarial-submissions/ADV-F4-timeout-swallow
```
预期：rc=1，run 判 PARTIAL，3 个用例全部 RUNTIME_FAIL
（compile_err 含「硬超时」）；`integrity.json` 里 `child.signal` 为
SIGTERM 或 SIGKILL、`killed_by_us: true`。
（不带环境变量跑的话，走的是子进程内 SIGALRM + 事后耗时核对，
预期同样是 RUNTIME_FAIL，只是慢：每用例约 65–75s。）

```bash
# 6d. C1 正控制：14 算子全手写的「刁钻但合法」提交（全量，约 3–8 分钟）
python -m triton_eval run --profile smoke \
    --submission examples/adversarial-submissions/ADV-C1-fullhouse
python -m triton_eval score --run latest
```
预期（C1 提交 14 个原语算子，跑 v0.5 全量 289 题；2026-10-11 真机基线 +
容差 1.1 修正后的预期；v0.5 起用例按算子再分配——SwiGlu 20→6、Scatter
20→10 等，XL 超时题数会随组合漂移、量级不变）：
- **11 个融合算子的 73 个用例判 SUBMISSION_INVALID 属预期**——C1 未提交
  这些算子；融合内核的正控制在 6d-3/5c 步。
- **XL 档若干题 COMPILE_FAIL**：纯 Python 实现超时走 compiled=False——
  负载分层本来就要淘汰纯 Python 实现，不是缺陷。
- **约 10 题 ArgMax RUNTIME_FAIL**：C1 的 0 维张量缺陷（`shape[d]` 越界）
  被新增的 0 维考法抓住——语料的战果，判罚正确。
- **静默错误（WRONG_RESULT）预期为 0**：2026-10-11 出现的 4 个已全部
  归因为评测方冤案（2×gelu：NPU 默认模式 f32 gelu 实为 tanh 口径；
  1×grouped_matmul：extreme 输入下参考自身 f32 累加噪声；1×bf16 1-ULP
  超过偏紧容差），并已由 tolerance.yaml 1.1 修复（gelu/grouped_matmul
  的 f32 覆盖档 abs=1e-3 + bf16 rel 7.8e-3）。**容差修正后的首轮真机
  复核：静默错误若非 0 即为回归，记下来**。
- **grouped_matmul-f32-S00 等此前被容差错杀的用例判 PASS**——
  容差修复（或语义 + 逐元素 allclose）的真机复核点；再见 WRONG_RESULT
  即回归。2026-10-11 已确认 PASS。

```bash
# 6d-2.（推荐）静默错误分型：误差 1e-5 级 = 疑容差标定问题（出题方的锅）；
#       差几个数量级 = 实现真错（语料考出边界，属预期战果）
python -m triton_eval diagnose --run latest
```

## 6d-3. 融合冒烟与探针实证（v0.5 新增；remote-verify.sh 5c/5d 已自动跑）

remote-verify.sh 从 v0.5 起多两步，随主流程自动执行并归档；**定向复验**
（修复后不必跑全量验证）一条龙命令——**环境刚刷新也能直接粘**（判仓库
存在、不存在就克隆；`runs/` 不进版本库，验证目录由命令自建）：

```bash
cd ~ && { [ -d triton-eval ] || git clone https://gitcode.com/hkhkhkzxk/triton-eval.git; } \
  && cd triton-eval && git pull --rebase --autostash origin main \
  && git log --oneline -1 \
  && source scripts/env-bootstrap.sh \
  && pip install -e . -q \
  && VDIR="runs/verify-manual-$(date +%m%d-%H%M%S)" && mkdir -p "$VDIR" \
  && python -m triton_eval run --profile smoke \
       --operators linear_gelu,rms_norm_linear,grouped_matmul_swiglu,transpose_linear,masked_softmax,masked_gelu,gather_exp,add_rms_norm,apply_rotary_pos_emb,moe_gating_top_k_softmax,unsorted_segment_sum \
       --submission examples/demo-submission \
  && python -m triton_eval score --run latest \
  && python scripts/probe_validation.py --out "$VDIR" \
  && python scripts/dot_precision_probe.py --out "$VDIR" \
  && bash scripts/upload-verify-record.sh "$VDIR"
```

（`git log --oneline -1` 是防"赛跑"的核对——第三轮曾因拉取与修复推送
相差不到一分钟而跑了一轮旧代码；开跑前看一眼提交号再继续。）

两份探针报告（probes-report 与 dot-precision-report）落在同一验证目录，
收尾一次归档全部带走。

真机结论（2026-10-13~14，记录 6844…-1013-192217 / …-manual-225653 /
…-manual-234810 / …-manual-1014-093413 / …-1014-170458）：

- **第四轮结果**：两段式与覆盖登记全部生效——GMS / RML / TL 全 6/6
  PASS；段和 9/10（唯 [4096,64] 第三次撞 UB，2D 路径已改 rank-1 同构
  一维循环待确认）；**探针收益实证收官**：9 条性能负载原子 7 条
  实证达标（1.5×~3.5×）、2 条如实声明性。
- **第五轮结果**：段和 10/10；v0.5 首个全量测速轮冻结 10 算子锚点
  （9 融合 + transpose，0.1352x~11.15x），静态错误 0。
- **第六轮结果（1014-170458，首个含 5e 的全量窗口）**：21 ✓ / 2 ✗。
  两个 ✗ 均已归因——① 5c 的预期口径缺陷（把 MaskedGelu 整算子当全
  显性不支持；实测自第一轮起就是 5 个 tanh 用例 PASS、3 个 erf 用例
  RUNTIME_FAIL，demo 内核行为符合设计，已按清单逐用例模式修正预期，
  见 robustness-findings 第五轮附录）；② 5e 首跑发现
  **cross_entropy_loss 在 npu:0 上同进程两遍结果不一致**（3 个用例：
  f32-L31 / f32-L40 / i64-M19，torch.equal 严格相等判据；与 v0.3 时代
  整型用例教训同类，登记在案；判定不受影响——历轮 CE 用例全部 PASS，
  容差吸收了该噪声；refcheck 诊断已升级为附带 max_abs/max_rel 量级，
  下个窗口实测后按既有覆盖机制登记处置）。跨机可复现 ✓（aarch64 重跑
  生成脚本，清单哈希逐字一致）、探针 7/2 复现、525 项真机测试通过。
- **第七轮结果（1014-224625）**：22 ✓ / 1 ✗——5c 预期口径修正验证
  通过（erf 用例 RUNTIME_FAIL、tanh 用例 PASS 分判生效）；唯一 ✗ 是
  5e 的 CE 量级测量本身（6 例，max_rel 全部 6~9×10⁻⁸，失败集合与
  第六轮部分重叠、总体随机）。**CE 锚点自噪声定案登记**：5e 改为
  「除已登记锚点自噪声（CE，rel≤1e-6）外零失败」，超界或其他算子
  仍报新发现；真机测试 526 通过。
- **第八轮结果（1015-105622）：23 ✓ / 0 ✗——首个全绿全量记录**。5e 的
  登记放行逻辑首次上机验证：本轮抽到 2 例 CE 噪声（第三个与前两轮都
  不同的随机集合，f32-L31 max_rel 1.28e-7 / i32-M27 9.1e-8，均远低于
  登记界 1e-6），全部正确进已登记类、判定 ✓；未知类零失败。真机测试
  539 通过（对应本机基线 509）。跨机可复现与探针 7/2 结论复现。
  **此后全量窗口的常态预期即 23 项全 ✓，任何 ✗ 都是回归。**
- 任何 **ORACLE_FAIL** 都不是提交质量问题，是评测侧问题，必须带
  逐用例日志回报（评测侧异常已按逐用例处理，不应再出现"整轮陪葬"）。
- 探针逐原子结论「实证达标 / 声明性考察（如实降级）」**都是合法结果**：
  降级只影响披露口径（覆盖率四类分列），不算验证失败。
- 探针跑批已自动设 `TRITON_ALL_BLOCKS_PARALLEL=1`（绕开 1 维 grid
  < 65536 上限）并逐形状隔离——单形状失败不拖垮整批；计时已带设备
  同步。

## 6d-4. 审计项真机复核（remote-verify 5e 步，2026-10-14 新增）

审计三件套（生成日志 / 演化对照 / 剔除记录防篡改，见覆盖度报告 2.6）
都是 L0 机制，真机侧由 5e 步补两个证据（随 remote-verify.sh 自动跑）：

1. **NPU 参考自检**：对 v0.5 冻结清单的 289 个用例，在 npu:0 上执行
   冻结门禁同款自检（参考可执行、输出有限、两遍一致）。门禁此前只在
   开发机 CPU 上跑过；参考实现在真机评测里每轮都执行过一次，但
   「同进程两遍一致」未在 NPU 上验证过。**已登记的锚点自噪声**：
   cross_entropy_loss 在 npu:0 存在执行间 ULP 级噪声（两窗口实测
   9 例次、max_rel ≤ 9.3e-8、失败集合逐轮随机；判定容差余量 3 个
   数量级，历轮 CE 用例判定全部 PASS）——5e 对「CE 的两遍不一致且
   max_rel ≤ 1e-6」按已登记类放行并记录，**超界或出现在其他算子仍
   按新发现报 ✗**。量级数据与登记依据见 robustness-findings
   第六/七轮附录。
2. **清单跨机可复现**：在真机（aarch64 + 昇腾 torch）重跑
   `build_corpus_v3.py --no-verify-refs`，预期输出「内容未变，未重写」、
   manifest.json 与冻结版逐字一致——生成只依赖纯逻辑（注册表 + 种子 +
   选择器），与设备无关，这步把该性质从机制断言变成实测事实。
   构建事件行随验证记录归档（step-5e-npubuild-events.jsonl），工作区
   随后还原（仓库日志只积累开发机构建）。

两步都只读清单或走纯逻辑，不改写任何冻结资产；失败即如实亮 ✗。

## 6e. standard 档测速（单独开窗口：一条命令脚本）

冒烟档只验对错、不测速度；真机环境**两小时自动清空**，验证流程（约
1-1.5 小时，v0.5 起含融合冒烟、探针实证与审计真机复核）+ 测速塞不进
同一个窗口。测速单独开一个新窗口，跑专用脚本
（自举环境 → 测速 → 打分 → 归档上传一条龙，无需手动装环境）：

```bash
cd ~ && { [ -d triton-eval ] || git clone https://gitcode.com/hkhkhkzxk/triton-eval.git; } \
  && cd triton-eval \
  && git pull --rebase --autostash origin main \
  && bash scripts/standard-window.sh
```

- **默认跑 demo 提交**（triton 内核，全量约 10-30 分钟）。demo 覆盖的
  算子若全对 → 自锚得 100 分，score 输出**真实的按算子性能分表**；
  未全对的算子列"无分原因"。2026-10-12 首个测速窗口实测基线：
  transpose 20/20 全对、自锚 100（锚点 0.14x——demo 内核比官方框架
  慢约 7 倍，属实测裸值）；exp/gelu/softmax/swi_glu 有声明式未实现
  （erf 模式/非末维 dim/非连续输入），其余算子按批注预期全对。
  **静默错误预期 0**（demo 的非连续输入缺陷已改为声明式守卫，
  仍出现即回归）。
- **2026-10-14 v0.5 首个全量测速轮（记录 …-manual-1014-093413，289 用例，
  静默错误 0）冻结的锚点**——10 个算子当轮全对自锚 100：

  | 算子 | 锚点（对框架加速比 geo） | | 算子 | 锚点 |
  |---|---|---|---|---|
  | Transpose | 0.1352x | | RmsNormLinear | 0.4772x |
  | TransposeLinear | 0.1829x | | GroupedMatmulSwiGlu | 0.5778x |
  | GatherExp | 0.2442x | | ApplyRotaryPosEmb | 0.6134x |
  | MoeGatingTopKSoftmax | 0.2521x | | AddRmsNorm | 1.2502x |
  | MaskedSoftmax | 0.3524x | | UnsortedSegmentSum | 11.1529x |

  读法：多数 demo 内核尚不及框架（亚线性锚点，评分器 0.2.2 的统一
  公式保证方向正确）；段和的朴素全扫参考在框架侧是逐段掩码循环，
  demo 并行内核反超 11 倍（超线性锚点）；transpose 锚点与 10-12 窗口
  的 0.14x 交叉复现（跨窗口漂移 ~5%，测量噪声）。其余 15 个算子
  无分并如实披露原因（6 个 demo 未提交 + 9 个带声明式守卫的原语
  算子——守卫明细与 3 个原语 f16-XL 用例的编译失败属 demo 既有
  边界，见"已知边界"）。
- **锚点连续性（环境清空后）**：锚点由数据库内的历史轮次推导，环境
  两小时清空后数据库随之消失——同语料+同提交的下一个全对轮会重新
  冻结出同量级锚点（transpose 0.1352 vs 0.1419 实证漂移 ~5%）。需要
  精确复用历史锚点时，把归档的测速轮 `results.jsonl` 导回数据库
  （`import` 子命令）即可恢复历史轮次。
- **为什么不是 C1**（2026-10-12 实测算的账）：性能锚点资格要求"全对
  **且都有加速比**"。C1 是纯 Python——XL 用例单次执行就要 110-118 秒
  （120 秒预算擦线过），standard 档测速段要在同一预算里连跑 6 次
  （被测 3 遍 + 参考 3 遍），物理上装不下 → XL 用例永远没有加速比 →
  C1 永远当不了锚点轮。这是负载分层的自然推论：纯 Python 实现过得了
  对错关、过不了性能关。triton 内核实现（XL 档毫秒级）才是锚点的料。
- 可选环境变量：`OPERATORS=exp,softmax`（只测指定算子）、
  `SUBMISSION=examples/adversarial-submissions/ADV-C1-fullhouse`（换
  被测提交——预期全对算子也出不了分，原因列在 score 表里）。
- 两个窗口测过同一提交后，可用 `compare` 出真实的两轮对比（性能显著
  变化 ±10% 清单）。
- **交互式跑任何 triton_eval 命令前，先自举环境**：venv 的激活只活在
  脚本进程里，脚本退出后交互 shell 仍是系统 python（实测踩坑：
  直接 `python -m triton_eval compare` 报 No module named triton_eval）。
  交互命令统一加前缀 `source scripts/env-bootstrap.sh &&`，例：
  `cd ~/triton-eval && source scripts/env-bootstrap.sh && python -m triton_eval compare --run 2 --against 1`
- 窗口快到点（约 1.5 小时）时，**先手动补一次
  `bash scripts/upload-verify-record.sh`** 再干别的——把已跑完的部分
  快照上去，环境清空也不丢。

## 7. 收尾抽查（可选）

```bash
sqlite3 runs/triton-eval.db "SELECT submit_id, package_name, validation_ok
  FROM submission_record ORDER BY id DESC LIMIT 5;"
sqlite3 runs/triton-eval.db "SELECT r.status, COUNT(*) FROM case_result r
  JOIN eval_run e ON e.id=r.run_id WHERE e.status='OK'
  GROUP BY r.status;"
```

## 8. 验证记录自动归档（remote-verify.sh 收尾自动执行）

`remote-verify.sh` 结束时（无论成败）会自动把当轮证据提交并推送 gitcode：
`verify-records/<主机>-<时间戳>/`——`SUMMARY.md`（一页报告：环境指纹、
全量测试计数、逐项判定明细）+ 各步日志 + 每轮的 results.jsonl +
完整性档案 + 生成的报告。环境清空后记录仍在仓库里，**无需人工拷贝日志**。

防丢两道闸（2026-10-12 因此丢过一整轮验证记录后加的）：

- **开跑前 10 秒预检**：gitcode 不可达只警告；本地代码落后于 main、或
  推送凭证无效（`git push --dry-run` 探测，不产生任何提交）则**立即退出**——
  不等整轮跑完（最长约 1.5 小时）才发现证据推不上去。
- **冒烟证据先行归档**：`WITH_STANDARD=1` 时，冒烟阶段（含"静默错误
  归零"核心断言）跑完就先归档上传一次；standard 档跑完收尾再归档覆盖
  同一记录目录。中途环境被清或脚本被杀，前半程证据已经在 gitcode 里。

- **推送凭证自动配置**：克隆出来的 remote 是匿名地址，拉代码不需要凭证、
  推归档需要——脚本开跑前检测到匿名地址会自动写入内置令牌（`GITCODE_TOKEN`
  环境变量可覆盖内置值）。内置令牌随仓库公开可见（仅本项目写权限），
  验证周期结束后应在 gitcode 吊销并换新。
- **任何退出方式都归档**：脚本收尾用退出 trap 保证——跑完、Ctrl-C、断连、
  被终止，都会把已有证据归档上传。checks.log 末尾有 `[完]` 标记即脚本
  跑完了全部步骤；没有它说明中途被断（2026-10-12 连续两轮在第 5 步后
  中断，据此定位）。standard 档开跑前还会先把冒烟阶段证据先归档一次，
  即使测速阶段被 SIGKILL（trap 不可捕），前半程证据已在 gitcode。
- **卡点永远可定位**（2026-10-12 实测教训：git 对断死的连接默认不超时，
  会无限挂起且零输出，脚本"看起来死了"其实卡在 git 里）：每节标题与
  每步的开始/结束都带时间戳——终端最后一行就是卡点；归档里的 git
  操作一律带 5 分钟硬超时，超时则改为本地留档 + tar 包，手动重跑
  `upload-verify-record.sh` 即可补传（幂等）。怀疑卡住时可看：
  `ps -eo pid,etime,command | grep -E "remote-verify|upload-verify|git |triton_eval" | grep -v grep`
  ——`git` 进程挂很久 = 网络断死连接（等超时或手动停掉重来）；
  `triton_eval run --profile standard` 在跑 = 正常测速中，等它跑完。
- 推送失败时：归档提交留在本地，并在 `runs/` 下生成备用归档包
  `verify-record-<时间戳>.tar.gz`（可手动 scp 带走）。
- 无网络/暂不推送：`SKIP_ARCHIVE=1 bash scripts/remote-verify.sh` 跳过归档。

---

## 已知边界（不是缺陷，验证时不要误报）

- 子进程命令可被 `TRITON_EVAL_CHILD_CMD` 覆盖（测试/调试接缝）。
  事件预算可被 `TRITON_EVAL_CASE_TIMEOUT_S / _IDLE_MARGIN_S / _GRACE_S`
  压缩（6c 用到）。正常运行不要设这些变量。
- `run --submission` 指向 `.zip` 时自动摄取到 `runs/ingest/<sha12>/`，
  幂等（同包同地）。`ingest` 子命令可独立完成同一步。
- 提交方后台线程在用例间隙向 stdout 打印伪造事件行：事件通道是
  独立文件而非 stdout，stdout 上的内容一律按诊断对待（进
  child-stdout.log）。
- v0.5 的融合算子里，LinearGelu 的 6 个用例与 MaskedGelu 的 3 个
  erf 精确用例走 demo 内核的显性 NotImplementedError（erf 依赖未验证，
  不凑近似）——这些用例 RUNTIME_FAIL 是预期，不是缺陷；**MaskedGelu
  的 5 个 tanh 近似用例已实现，PASS 是预期**（2026-10-14 修正口径：
  此前文档把 MaskedGelu 整算子写成全部 RUNTIME_FAIL，与第一轮起的
  实测不符，归因见 robustness-findings 第五轮附录）。两个算子的锚点
  待支持 erf 的真实提交出现后自然冻结（按算子锚点机制支持这种状态）。
- tl.sin / tl.cos 已在真机确认编译可用（2026-10-13）；三个 f32 大 K
  用例（grouped_matmul_swiglu-f32-X48、transpose_linear-f32-X48、
  rms_norm_linear-f32-X56）与段和大归约用例 f32-X92 的顺序噪声已按
  C2 机制登记 B 类容差覆盖（tolerance 1.2，重放钉保证 10× 真错仍拦），
  预期 PASS；**其余**用例出错才是回归。
