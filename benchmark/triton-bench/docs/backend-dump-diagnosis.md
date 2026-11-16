# 两个后端模式标签：独立根因诊断

## 已知事实与尚未证实的假设

v15 真机记录 `20261026-120532-223940` 中，Neg、Matmul 共 40 例功能和参考门禁通过，但 `FA-PAT-VEC-LOAD`、`FA-PAT-TILE-GEMM` 的编译认证、执行认证均为 False。这里的 False 是“现有证据不能认证”，不是“没有运行过这两个算子”，也不能解释为编译器必然没有实现这些优化。

30 次实际编译全部绑定了真实输入和返回的二进制；旧版输出前缀问题已解决。当前直接阻塞点是：后端 stdout/stderr、文件树中没有可用于认证的后端 IR。具体复核见 `docs/v15-npu-validation.md`。

需要验证的下一层假设是：厂商打印入口可能过滤 pass 名或使用不同输出作用域，导致通用 After/module-scope 与 `canonicalize` 组合没有输出所需阶段。诊断选择 `hivm-inject-sync`、`hivm-inject-block-sync`、`hivm-graph-sync-solver` 作为一组待验证的厂商打印过滤条件；**不是已经确认这些 pass 在真机安装构建中存在或会执行**。工具接受这些参数也不能证明假设成立。

因此不直接在冻结采集器中替换参数、不放宽数据流识别器、不用普通 TTIR load/dot 代替后端优化证据，也不再次盲跑全量。

## 本轮实现

新增独立入口 `scripts/backend-dump-diagnose.sh` 和工具 `scripts/diagnose_backend_dump.py`，不改 v15 冻结框架、语料、内核、参考、容差或 71 标签分母。

1. 先校验 v15 原报告、manifest、提交资产、参考门禁和证据索引。只从 PASS、实际编译且非缓存复用、输入和二进制已绑定的目标用例中选取输入。原始归档仅只读访问。
2. 固定选择 Neg 的 `official-neg-float16-27-normal-c` 和 Matmul 的 `official-matmul-float16-24-normal-c`。保留它们真实的 `ttadapter`、`npubin`，不重新生成输入或内核。
3. 只复用已审核的六个语义编译选项；不执行归档工具路径或任意归档参数。使用当前安装的 `bishengir-compile`，工具 SHA256 必须等于原始实际编译工具的 `554597c9adc8bb8760c9172bd00421b66c9487893473b78f3056c70b6e3a9505`，否则停止。允许显式 `--compiler`，但不能绕过哈希校验。
4. 有界保存实际 `--help` 和帮助中声明的 `--help-hidden` 原始输出；缺选项、非零退出、超时或输出超限时拒绝继续。预检失败仍保存已校验的源证据及失败原因。
5. 对每个输入重放两个配置，共 **4 次后端编译，不执行 NPU 内核**；另有帮助查询。每次编译上限 120 秒，单独临时目录，显式保留 v15 的 `MLIR_ENABLE_DUMP=1`。

| 组别 | 打印参数 |
| --- | --- |
| `v15-control` | 标准 After-all、module-scope、厂商 after=canonicalize、独立 tree-dir，与 v15 对照 |
| `vendor-whitelist` | 厂商 before/after 同时请求上述三个待验证的 pass；不加通用 module-scope 或 tree-dir |

这个对照改变的是打印配置组合，**不是单因素试验**。即便第二组成功，也只能先定位到这条输出配置路径，不能直接断言唯一原因就是某个 pass 或作用域开关。

6. 保留原始 stdout/stderr、saved MLIR、二进制候选、完整哈希、实际命令、编译工具身份和源环境信息；与归档 `npubin` 逐字节比较。按真实 Before/After、模块/函数作用域及 `hivm.hir.*` 操作建立诊断清单。函数级文本只进入诊断清单，未被正式认证规则接纳。
7. 输出两组对照结论及证据索引，并使用现有结果发布器上传到滚动 `npu-results` 分支；失败诊断同样尝试上传。上传失败时不删除本机原始文件。

## 结果怎样解释

输出位于 `runs/backend-diagnostics/<批次>/`，其中 `evidence/diagnostic-report.json` 是诊断报告，`evidence/artifacts.json` 是文件哈希索引；每个算子、每个配置都有 `replay.json` 和压缩原始产物。

| `comparison.conclusion` | 含义和下一步 |
| --- | --- |
| `vendor_filter_restores_dump_in_this_replay` | 同工具、同二进制的对照无所需 dump，厂商配置有 dump。复核真实作用域、pass 与操作语法后再修正式采集器。 |
| `both_variants_produce_dump` | 两组都能输出。优先比较独立重放与原 JIT 子进程的环境、重定向和嵌套调用，不能归因于过滤条件。 |
| `only_control_produces_dump` | 对照有 dump、厂商配置无 dump；该厂商配置不足以解决问题。 |
| `backend_dump_still_missing` | 两组仍无目标后端文本。继续排查实际安装构建、内部工具及打印转发链路，不声称优化不存在。 |
| `archived_binary_not_reproduced` | 独立编译没有复现原二进制，不可作为原 launch 的补充认证；保留文本排查差异。 |
| `compiler_replay_failed` / `compiler_identity_changed` | 编译失败、超时或工具变化；先修复该问题，不能给出模式通过结论。 |
| `dump_present_but_capture_needs_review` | 观测到目标文本，但仍有采集告警、截断等问题，先复核完整性。 |

`ready_for_review=true` 只表示同工具、逐字节相同二进制、有界采集且有目标函数后端文本，便于人工复核；**不是编译认证或执行认证通过**。对照结论本身也没有独立的认证效力。

报告固定声明 `diagnostic_only=true`、`affects_coverage=false`、`coverage_certified=false`、`execution_validation_performed=false`。进程返回 0 表示两个厂商配置都有可复核产物，或本机 `--prepare-only` 完成；1 表示未获得这些产物/预检失败；2 表示上传失败。这些退出码均不用于修改原覆盖率。

取得真实后端语法后，才能据此调整正式采集器及严格识别规则，并冻结新版本（v16），仅重跑 Neg、Matmul 40 例，重新绑定真实编译产物、正确性及 CPU/NPU 参考门禁。不能把独立编译重放冒充新的执行验证，不能原地修改 v15 历史结果。

## 本机发布到 GitCode

只在发布副本运行下列命令，不在旧工作区将全部历史 dirty 文件一起提交。本轮的新脚本、测试、文档需先同步到该副本；不要使用不加检查的 `git add -A`。

```bash
cd /private/tmp/triton-eval-v09-publish-yg17nfbs
git switch -c codex/backend-dump-diagnosis
git add scripts/diagnose_backend_dump.py scripts/backend-dump-diagnose.sh \
  tests/test_backend_diagnostics.py docs/backend-dump-diagnosis.md \
  docs/v15-npu-validation.md docs/v15-backend-capture-fix.md
git diff --cached --stat
git commit -m "Diagnose missing backend dumps using bound v15 compiler inputs"
git push -u origin HEAD
```

若同名分支已经存在，先检查其内容后使用 `git switch codex/backend-dump-diagnosis`，不要重置或强制覆盖它。

## 真机拉取并运行

```bash
cd "$HOME/triton-eval-v07"
git fetch origin refs/heads/codex/backend-dump-diagnosis
git switch --detach FETCH_HEAD
git log -1 --oneline
bash scripts/backend-dump-diagnose.sh
```

使用 `FETCH_HEAD` 避免只 fetch 分支名却没有生成 `origin/<分支>` 的旧问题。有本地改动时先保留，不要强制切换或清理。入口不接收扩大算子范围的参数，也不会启动全量测试。

入口优先读取真机原有 `runs/ascend-suite/20261026-120532-223940/evidence`；若不存在，强制更新滚动结果引用，并从捕获的结果提交中提取该批次到新的独立目录。取不到旧批次则明确失败，不退回其他批次、不触发全量重跑。可显式指定已下载的原始结果：

```bash
TRITON_EVAL_BACKEND_SOURCE=/实际归档路径/records/20261026-120532-223940/evidence \
  bash scripts/backend-dump-diagnose.sh
```

环境启动沿用原 `env-bootstrap.sh`。若现环境更换了编译器，诊断会拒绝哈希不一致；应恢复原安装构建，而不是删除校验或据新编译器输出追认旧 launch。

## 本机可完成的检查

```bash
PYTHONPATH=src python -m pytest tests/test_backend_diagnostics.py -q
bash -n scripts/backend-dump-diagnose.sh
python scripts/diagnose_backend_dump.py \
  --results /实际归档路径/records/20261026-120532-223940/evidence \
  --prepare-only
```

合成工具测试和本机准备成功只证明诊断机制可用，不计为真机证据。当前两个标签仍未认证；历史分区叠加编译/执行覆盖维持 65/71（91.55%），静态覆盖维持 67/71（94.37%），共同确认和最终任务书验收状态不变。

### 本轮本机验证记录

- 诊断专项 36 passed；包括隐藏帮助、截断/失败留档、环境复现、二进制不一致、超时、工具变化、对照解释，以及一键入口的本地原件/固定结果提交提取/旧批次过期三种路径。
- 全套回归 978 passed、188 skipped；日志为 `/private/tmp/triton-backend-diagnostics-all-tests.log`。跳过的设备相关测试不算真机验证。
- 发布副本中的 36 项诊断测试也通过；Shell 语法、diff 空白检查通过。尚未提交或推送。
- 真实 v15 归档的 `--prepare-only` 通过，校验 290 个索引文件并选出上述两份编译输入；产物在 `/private/tmp/triton-backend-prepare-4NaCst/record`。
- 重新运行静态发布门禁：67/71、1377 例、原 1360 例保留，74 入口源码契约通过；报告在 `/private/tmp/triton-backend-diagnostics-readiness.json`。本机未进行目标环境实际载入或 NPU 执行。
- 再次独立重算 v15 报告，与归档完全相同，两个标签仍未认证；没有写回原始归档。
- 发布前尚未取得本工具的真机对照输出。现已收到批次 `20261026-155402-979553`：新打印配置得到两个目标函数的真实后端 IR，四次编译的二进制均与原件相同，具体独立复核见 `docs/backend-dump-diagnosis-results.md`。正式快照和数据流识别仍需适配，尚未冻结 v16、未新增编译或执行认证标签。
