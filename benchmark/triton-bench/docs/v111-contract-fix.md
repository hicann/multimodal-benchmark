# v0.11.1：入口签名修复与运行前门禁

> 已废止：按 2026-10-26 用户要求，本修复版及当日后续修复版不再使用。本文和旧冻结资产仅留档，命令不再作为当前运行指引。替代方案见 `docs/v12-compatibility-fix.md`，真机只运行两个指定算子。

v0.11 的 Neg 实现入口为 `neg(input)`，注册契约要求 `neg(x)`，实际加载按 `SUB-LOD-007` 拒绝。每个算子分区都会校验整份提交；任一入口不合法会使该分区的提交校验失败，因此单个 Neg 签名错误会反复影响其他算子分区。这是发布前入口校验遗漏，不能解释为 Triton 编译或执行失败。

本版将已审阅模板和新的生成样例入口统一改为 `neg(x)`。未放宽参数名契约、反作弊检查、参考门禁、容差或覆盖判据，也不修改运行器整份提交合法性要求。v0.11 语料与样例保持原样供复现。

新版本为 `corpus/v0.11.1-contract-candidates`，样例为 `examples/coverage-v111-ascend-submission`。全部 74 个算子、1,377 个用例的编号和内容哈希与 v0.11 相同；保留原始 68×20 用例和全部17个补充用例。静态覆盖仍为固定 71 项的 67/71（94.37%），编译和执行覆盖需依据本版真机结果重算。

新增门禁：

1. 生成阶段冻结前、复用已有生成结果时及本机运行入口，均检查完整提交的源码入口签名、调用方式与语料算子清单。源码签名和真实加载复用同一组参数名、个数、仅限位置参数判据。不执行 import、默认值、注解或函数体；仅支持样例使用的直接、未装饰同步函数。其他形式保留明确未通过。
2. 真机套件在创建任何算子分区和参考门禁前，通过独立子进程真实导入整份提交并执行既有加载校验。有错误立即退出，不继续重复分区。该步骤会真实执行 Python 模块导入，不编译或调用内核，不代表设备执行验证。
3. 真机真实加载检查归档为 `evidence/submission-preflight.json` 与 `.log`。发生超时或进程异常时以 `.log` 和退出码为准，不伪造通过记录。生成日志记录本机源码契约检查结果。

本机检查（无需 Triton/NPU）：

```bash
python3 scripts/check_coverage_readiness.py
python3 scripts/check_submission_readiness.py
```

第一条同时验证静态覆盖与全部入口；第二条仅验证入口。真机激活正式环境后可单独执行 `python scripts/check_submission_readiness.py --load`，检查实际导入兼容性。

推送本版本后运行：

```bash
cd "$HOME/triton-eval-v07" &&
git fetch origin refs/heads/codex/v09-generation-validation &&
git switch --detach FETCH_HEAD &&
git log -1 --oneline &&
bash scripts/coverage-suite-verify.sh
```

尚未获取中断运行的原始日志；当前已在本地复现明确的 Neg 签名错误，不能推断其他真机失败均由此引起。新增门禁不能保证未实测的后端编译、向量访存和共享内存适配均能通过。编译和执行覆盖仍为 `NOT_RUN`，性能收益仍未建立，清单共同确认状态未改变。
