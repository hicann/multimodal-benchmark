# v0.8 真机结果审计与 v0.8.1 修复

日期：2026-10-23。结果分支 `npu-results`，提交
`c8ed79288286529ca81112d2f69ce029b485e2df`，记录 `20261023-120902-701382`；
执行代码 `87f1066e6deabdc70b775005fbad7c2472b48046`。
逐算子重新审计、环境、原始证据 SHA256 和恢复凭据见
[`v08-npu-results-audit.json`](v08-npu-results-audit.json)。

## 实际结果

环境：Ascend 910B3、aarch64、CANN 9.0.0、Triton Ascend 3.2.0、
Torch 2.7.1+cpu、torch_npu 2.7.1.post4。运行档位为 `smoke`。

| 口径 | 数量 | 解释 |
|---|---:|---|
| 计划候选 / 用例 | 68 / 1360 | 每算子 20 例 |
| 实际内核评测记录 | 1300 | 其余 60 例由独立参考门禁拦截 |
| PASS | 1273 | 全计划通过比例 93.60%；已执行记录通过比例 97.92% |
| COMPILE_FAIL | 26 | Pow 20；TopKValues 6 |
| 原记录 RUNTIME_FAIL | 1 | TopKValues 513 列用例，MLIR 编译错误包裹 120 秒超时 |
| 整算子 20/20 PASS | 63 | 其中 4 项是独立披露的编译器原语 |
| 可计规模的整算子 | 59 | 排除 Exp2、Log2、Log10、XorSum |
| 最终选择 | 50 / 1000 | 同一环境、来源与资产版本，通过原资格审计 |
| 冻结标签覆盖 | 33/71 = 46.48% | 未达到任务书 90%；不等于 Triton API 执行覆盖率 |

原始全量 summary 的 `complete=false`、`all_pass=false` 与 50/1000 资格成立不矛盾：
前者要求全部 68 项完成通过，后者允许从同一轮通过的冗余候选中选 50 项。
本轮只是 smoke 功能证据，不能替代 standard/official 的性能验收。

## 五个未通过项目

| 项目 | 原始证据 | 分析与修改 | 尚需确认 |
|---|---|---|---|
| ArgMax | NPU 参考门禁：`float32-02` 双参考不一致；20 例均未进入被测内核 | 第 02 组包含重复极值，合同要求最左索引。Torch 主参考改为 amax + 显式最左匹配索引归约，独立 NumPy argmax 保留 | 旧日志没有实际索引，不能直接断言根因一定是并列规则；新诊断及 NPU 复验确认 |
| FloorDiv | NPU 参考门禁：int32 第 02/06/10/14/18 组不一致 | 都是包含 INT_MIN/INT_MAX 的边界模式；原 torch.floor_divide 存在精度风险。主参考改为 FP32 估计 + int64 余数修正，覆盖正负分母，不调用该后端整数除法；NumPy 独立参考保持原样 | 真机 int64 路径及结果须复验；CPU 全 int32 随机与边界检查已通过 |
| Signbit | NPU 参考门禁：float16-03/07、float32-02/06 不一致 | 第 02 模式明确含 -0.0；Torch 主参考用原 dtype IEEE 有符号位视图，Triton 用 fp32 位解释，保留负零；NumPy signbit 不变 | 旧日志无具体值，负零归因仍需新诊断确认 |
| Pow | 全 20 例 COMPILE_FAIL：tensor 无 `__pow__` | 当前 Triton 3.2.0 不支持此 `**` 路径。正基数合同内改用 `exp(y * log(x))`，不缩小输入、不改变容差 | NPU 编译和精度复验；不声称原 `**` 已支持 |
| TopKValues | 13 PASS、6 编译失败、1 超时；小 shape 第 00/01/02 组失败，513 列触发 MLIR 超时 | 原 `tl.topk` 路径兼容性差。使用本轮 SortValues 20/20 已通过的 CANN extension.sort，降序排序后仅写前 k 个值，保留非 2 幂 k 和大尾块 | 复验相同形状及 k；不声称原 `tl.topk` 能力通过 |

未移除失败算子，未放宽浮点容差，整数与布尔仍要求精确一致。
新增 NPU 回归文件 `tests/test_official_npu.py`，覆盖五个项目全部 100 例。

## 远程发布破坏冻结资产的问题

旧 `publish_ascend_results.redact` 用 `https?://[^\s]+` 替换所有链接。
在有格式的 JSON 中，这会吞掉 URL 后的结束引号和逗号。
`frozen-official50/official-catalog.json` 因此不再是合法 JSON，SHA256 也不匹配。
逐例 results、环境、integrity 和 qualification 的证据哈希经原代码独立重算一致，
资格结论未被此问题推翻，但远程冻结目录原样不可校验。

本轮修改：保留公开来源 URL，只去除 URL 认证信息及敏感查询字段；按原字节复制
不需脱敏的资产；发布前逐一核验可移动冻结资产的 SHA256，改变则拒绝发布。
新增真实本地裸仓发布回归，检查 JSON、CRLF 和来源链接逐字节保持一致。
日志入口等待 tee 落盘再上传，保持环境启动失败和评测失败时的归档及退出码。

通过固定候选目录和 qualification 的选中名单，可以确定性重建受损文件。
恢复后的 SHA256 为 `081e73b03e6b32aaf89f74c454fee559a60a12d02a3460dc16d243119c0b68d4`，
与原始冻结指纹完全相同。已在原代码检出目录下核验全部 50 个算子、1000 个用例、
参考文件和框架指纹。原下载记录未改动，恢复件位于 `runs/recovered-v08-audited/frozen-official50`。

可复现恢复，输出目录必须不存在：

```bash
python scripts/repair_official_results.py \
  --candidate-corpus corpus/v0.8-official-candidates \
  --evidence runs/remote-npu-results/records/20261023-120902-701382/evidence \
  --out runs/recovered-v08/frozen-official50
```

恢复脚本核对所有 portable 资产，并拒绝与原 SHA256 不同的重建；完整框架核验
仍须检出原执行代码 `87f1066`，不应在已修改的工作区重写旧冻结指纹。

## 新版本与复验

修复改变了参考与判定资产，所以新建 `corpus/v0.8.1-official-candidates`。
仍为 68 项、1360 例，保留相同参数、seed 与边界输入，不继承 v0.8 的 NPU 资格。
旧 v0.8 指纹和结果保留；旧版本复现应检出对应代码。
构建器遇到已有清单时也会核验完整评测资产，不能只检查 manifest 后返回成功。

在更新后的真机代码目录运行五个项目的 100 例定向复验：

```bash
bash scripts/official-suite-verify.sh --operators ArgMax,FloorDiv,Signbit,Pow,TopKValues
```

局部运行只审计所选项目，跳过要求 50 个算子的正式冻结，不混入旧轮 PASS。
完整复验及生成同版本资格资产：

```bash
bash scripts/official-suite-verify.sh
```

本机无 NPU，不能宣称五项已在真机修复通过。CPU 双参考、输入边界回归、上传与恢复
回归用于验证可本地验证的逻辑，最终 NPU 资格以新一轮同版本记录为准。

## 本地验证记录

2026-10-23，本机 macOS 的可用环境为 Python 3.9.6、Torch 2.2.2、NumPy 1.26.4。
项目声明运行时为 Python >=3.11，真机原运行使用 3.11.4；本地检查不能替代目标环境验收。

- `PYTHONPATH=src /private/tmp/tv5/bin/python -m pytest -q`：600 passed、163 skipped，93.94 秒。
  跳过项目需要 NPU 或 Triton，包含新增的五算子 100 例真机回归。
- `build_official_corpus.py`：1360 个 CPU 主参考/独立参考门禁全部通过，日志 `failures={}`。
- v0.8.1 与 v0.8 的 manifest `cases` 字段逐项相同；新 manifest 为
  `a614114dbc93ca0780378cf4c2c6f606e8055dd353c042890326a5b6e9aa2c10`，
  版本和评测资产独立冻结。
- 额外检查 200000 对随机 int32 除法输入及 11263 对正负分母边界输入，与 NumPy 零差异。
- 恢复的 50/1000 冻结目录已在原代码版本完整校验；经新发布复制流程后，105 个文件逐字节不变。
- Bash 两个运行入口语法校验通过；本轮修改的差异格式检查通过。
  工作区已有的 `docs/requirements-coverage-report.md` 用户修改保持原样，未纳入本轮编辑。
