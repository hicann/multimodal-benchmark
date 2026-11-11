# v0.6 真机失败分析与完整运行方案

跨算子、外部提交与后续版本的治理方案见
`docs/operator-evaluation-hardening-plan.md`。该方案为待实施设计；本文的运行证据
与已经完成的工具改动应分别理解，不能据此认定内核或编译器问题已全部修复。

## 1. 已核实的结论

分析依据是 Gitcode 分支 `verify-records/20261023-093228`，提交
`ad2c548a6320a4f4718048a33829b0d57127ea52`，文件
`verify-records/ascend-ops-20261023-093228.jsonl` 和 `verify-records/report-run-1.md`。
已通过远程分支列表确认：本次检查时这是最新的验证分支。

这轮已经为 **220 个不同用例各产出一条终态记录，覆盖 11/11 个算子**。
报告开始、结束时间为 `2026-10-23T01:33:09Z` 和 `01:36:18Z`，约 189 秒。
所以当前首先要解决的是判定和实现的正确性；这轮没有因两小时窗口而漏跑用例。
“完整产出记录”“实际执行成功”“全部判对”应分别验收。

15 指的是来源教程条目；`tiling`、`load_order`、`ub_overflow`、
`load_care_padding` 四个技巧演示没有进入算子注册和本轮分母，不能解释成四个失败算子。

| 算子 | PASS | 其余终态 |
|---|---:|---|
| VectorAdd | 20 | 无 |
| VectorCmp | 20 | 无；当前实际只校验主输出 |
| DiscreteMemoryAccess | 10 | COMPILE_FAIL 10 |
| InsertSlice | 0 | WRONG_RESULT 10、SHAPE_MISMATCH 10 |
| ExtractSlice | 0 | WRONG_RESULT 20 |
| AssignReqToTokenPool | 20 | 无 |
| DecodeGroupedAttention | 0 | WRONG_RESULT 20 |
| CausalConv1d | 6 | COMPILE_FAIL 10、WRONG_RESULT 2、SUBMISSION_INVALID 1、SHAPE_MISMATCH 1 |
| GatherScatter | 20 | 无 |
| BinnedGatherScatter | 20 | 无 |
| PaddedGatherScatter | 0 | SHAPE_MISMATCH 13、WRONG_RESULT 7 |
| 合计 | 116 | COMPILE_FAIL 20、WRONG_RESULT 59、SHAPE_MISMATCH 24、SUBMISSION_INVALID 1 |

原始 PASS 比例为 52.73%，不能直接称作 OpScore。报告显示“无分数”，运行脚本
原先只执行 `report`，没有先执行 `score`。状态 `OK` 也只表示运行流程完成。

环境为 910B3、CANN 9.0.0、Python 3.11.4、torch 2.7.1+cpu、
torch_npu 2.7.1.post4、Triton 实测导入 3.2.0。早期 v0.5 的
910B4/CANN 8.5.0 记录不能替代这套环境下的验证。

验证分支的代码基点是 `a372532`。当前主分支 `eb45ef3` 后续修了
ExtractSlice 的 BLOCK_SIZE，并补了非有限输出的错误详情；这两项尚未被上述结果验证。

## 2. 逐项归因及处理方式

### P0：先处理越界，避免污染其他用例

**CausalConv1d：固定通道块与输入不匹配。**

教程 `003-fused-cat-slice-conv1d.py` 的 NPU 分支固定 `DIM_BLOCK=1024`，
内核对通道没有边界 mask；语料却包含 `D=256`。它会按 1024 通道读取输入、
权重并写回状态和输出。克隆所有入参只能防止正常的就地修改，不能阻止越界写。
这可以解释小 D 用例出现的极大数、非有限值和输入指纹变化；具体内存损坏范围还需
单独进程和设备诊断确认，不能把这几条结果直接归因给编译器。

另外，`seqlen=8` 的十条用例全部在动态 `extract_slice` 处编译失败。
该分支传的 offset/size 还缺少单元素 tuple 的逗号。即使解决编译问题，
其时间循环的 `tl.store` 地址也没有 `+ i`，会反复覆盖同一个时间位置，
其余位置仍是原输入。因此只改 tuple 或屏蔽编译错误都不足以修复。

建议在新的适配提交内复制并修订该内核，保留来源教程文件：

1. 对本轮通道均为 256 的倍数这一输入域，使用能整除 D 的块，例如 256；
   扩大输入域后必须增加通道 mask。
2. 明确使用单元素 tuple；对需要编译期 offset 的扩展使用 `tl.static_range`，
   再验证这版编译器的扩展签名，不能仅靠异常文本判断修复成功。
3. 给每个时间位置加正确的输出偏移，并确认状态只更新一次。
4. 输出与输入分配独立缓冲，状态在副本上更新，返回 `(out, new_state)`。
5. 首先单独执行 `D=256/1024`、`seqlen=1/8`、有/无 silu 的小矩阵；
   这八类组合通过后再跑 20 例，并在新进程复测其他算子。

**PaddedGatherScatter：缩放语义不一致，并有权重越界读取。**

`ref.py` 定义的是不带权重的重排；提交包装器却把 weights 传给教程，
激活 `SCALE=True`。教程中 `idx_x = token_id * NUM_COLUMNS` 是数据偏移，
但取权重时也用 `weights + idx_x`。权重长度只有 `sl * top_k`，
这个地址对大量 token 会越界。它与巨大的误差、NaN/Inf 形态相符。

当前契约下最小修复是包装器传 `weights=None`，与 GatherScatter、
BinnedGatherScatter 已采用的不加权语义一致，并逐行核对有效区域与零填充区域。
如果要考察带权重版本，应新增明确的 scale 变体，同时给参考实现和内核定义同一个
权重索引（例如原始路由槽 `indices[position]`），不能只改参考让错误输出通过。

### P1：修复教程及包装器的确定性错误

**InsertSlice：原始教程的写回地址错误。**

`001-insert_slice.py` 中的二维写回地址为：

```text
out_start + arange(BLOCK_SIZE)[:, None] + arange(D)[None, :]
```

行偏移缺少乘 D，多个行写到重叠区域，大部分输出没有写入。
另外 `insert_slice(..., [i,D], [1,D], ...)` 的列起点为 D，超出了
形状 `(BLOCK_SIZE, D)` 的列范围，应为 `(i, 0)`。
尾块还需要用有效行约束 gather 和 store，避免无效索引读取或写回。
单纯初始化输出为零只能隐藏未写区域，无法修复上述地址错误。

建议保留 insert_slice 扩展作为被测探针，在适配副本中修正 offsets 和
`row * D + col` 写回，并测试尾块。若另写普通 gather 内核作为正确性对照，
需要单独标识它的实现身份；不能据此宣称 insert_slice 扩展已经通过。

**ExtractSlice：包装器块宽错误。**

教程把二维载入的列宽也写成 BLOCK_SIZE，而包装器原先固定 22；
语料的 D 是 32/64/128/256。当前主分支已经改成 `BLOCK_SIZE=d`，
对本轮 S、D 均能整除的组合可解决这处确定的维度不一致。
但 `D=256` 会生成 256×256 的块，仍需真机检查 UB 占用。
新增非整除 S 时还需修复载入 mask：它目前没有把 `block_start` 算进边界。

**DiscreteMemoryAccess：int32 gather 不被当前编译路径支持。**

十条 int32 用例错误完全一致：

```text
Expected dtype fp16/fp32/bf16/f8E5M2/f8E4M3FN/int8, but got int32
```

float32 的十条全部通过，所以继续缩小 N 不会解决这个 dtype 限制。
若评测目标是该编译器的 `tl.gather(int32)` 支持度，应保留 COMPILE_FAIL 并提交最小复现。
若目标是可交付的正确算子，可在新的适配实现内用 int32 指针 load 完成离散读取，
但要保留原始 gather 探针的结果。不要用 int32→float32→int32 冒充完整支持，
超过 2^24 的整数会失去精度。

### P1：先统一注意力输出契约，再评估数值误差

**DecodeGroupedAttention：空 split 和低精度中间计算是两类问题。**

当前 KV 长度固定 128，最小 split 粒度为 32。当 max_kv_splits=8 时，
只有前四个 split 有效；内核跳过其余四个的写回，而包装器用 `torch.empty`
分配输出，参考却给这些位置填零。未写位置会参与比较，不能用调容差解决。
应约定空 split 返回零并在包装器初始化，或在明确的输出契约中只返回有效位置；
参考和提交必须遵循同一规则。

本轮 max_kv_splits=4 的十条用例最大绝对误差约为 0.0012–0.0019，
与另一组包含空 split 的大误差应分开调查。源码中内核按 16 token 块做
online softmax，并在 `p.to(v.dtype)` 后做 dot；参考把一个 split 的
32 token 一次计算，且概率一直保持 float32。这给出了明确的舍入差异来源，
但还不足以单凭源码认定这十条全部是可接受误差。

修复顺序：

1. 参考按 `ceil(ceil(kv_len / splits) / 32) * 32` 计算 split 长度，
   按 kv_indices 取 KV、按 query/kv head 分组，不能固定 `s*32` 和第 0 个 KV head。
2. 补一个独立 torch 诊断参考，逐块模拟低精度概率转换，和数学参考交叉比较；
   数学参考保留为语义锚点，不可只让判定复刻被测实现。
3. 分别记录 att_out、att_lse 的误差和有效 split；用多个 seed 复测。
4. 在取得误差分布和错误实现的反例之后，按现有算子级容差登记机制决定是否覆盖。
   float32 输出并不意味着内部计算拥有 float32 精度；也不能一律套用 bf16 宽容差。

## 3. 框架与生成侧需要补的门禁

这些是源码审计发现，不是本轮新增的编译器失败结论：

- v0.6 生成脚本采用笛卡尔积 `[:20]` 截断，没有调用现有 `guided_selection`。
  例如 VectorAdd 到不了 4194304 档，GatherScatter 到不了 sl=4096；
  每算子有 20 例不等于已覆盖计划中的全部维度。
- gather 系列把 `(sl, hs, ne)` 存入 case.shape，但实际主输入是 `(sl, hs)`；
  AssignReqToTokenPool 的 shape 是 `(1,)` 占位。这会让通用负载标签与实际工作量不符。
  需要明确 shape 为逻辑参数还是主张量 shape，并针对算子定义负载计算方式。
- 注册表声明 VectorCmp 返回 out/mean/rstd，实际参考和提交均只返回 out。
  必须统一规格与实现；若保留三个输出，就必须逐个校验。
- manifest 中有四个 ref_path 仍按 entry 拼接，指向不存在的文件；
  oracle 运行时已通过 source_name 回退绕开，但清单元数据和实际资产仍不一致。
- 大多数 v0.6 纯 torch 参考在模块顶层无条件加载教程及 Triton；这使本可 CPU
  检查的 oracle 依赖被测编译器。建议拆出输入构造与数学参考，CausalConv1d
  的 torch 参考也单独搬出，并确保自检包含每个输出。目前 refcheck 只取第一个输出。
- AssignReqToTokenPool 的请求行随机采样允许重复；若更新区间重叠，多个 program
  写同一位置可能产生竞争。输入构造应保证请求行唯一，或明确无重叠区间的约束。
- case 哈希不绑定参考源码，教程由 corpus 外部动态加载。归档时应同时保存
  manifest、参考源码、教程版本/哈希、提交快照、运行环境和逐例日志。

生成机制建议复用现有的 `候选池 → 合法性/参考门禁 → label_case → guided_selection`。
按算子保证 dtype、规模、边界、布局和属性组合的代表例，再补多样性。
“合法输入域”由算子语义决定；某份示例内核的资源限制单独记录。
不能在看到失败后删除合法用例、降低规模并沿用原测试集版本。
修改参考、用例选择或判定语义时发布新的语料修订并重建性能锚点。

## 4. 可执行的分阶段验收

1. **先保全证据**：固定以上验证提交和环境，保留 220 条原始结果。
   后续修订提交与原始教程探针分别登记身份，逐算子做前后对照。
2. **先清理越界与契约问题**：CausalConv1d、PaddedGatherScatter、InsertSlice，
   然后 ExtractSlice 和 DecodeGroupedAttention。危险用例按独立子进程/算子隔离，
   设备故障后不能继续把受污染进程的后续误差当作独立证据。
3. **参考门禁**：全部输入构造有效；全部输出非空、有限、确定；输入不被修改；
   索引边界、padding、状态更新和多输出逐项通过独立参考校验。
4. **正确性闭环**：11 算子的小样本冒烟通过后再跑全量。
   合法但尚不支持的编译路径仍产出真实失败，保留复现；适配实现另行验证全 PASS。
5. **性能闭环**：正确性通过后逐算子执行 standard，再做 official。
   smoke 不产性能样本；未锁频的结果继续标为 UNVERIFIED。
   切换实现、语料修订或环境时重新核对锚点，不能跨版本直接比较分数。

本次已补的工具和流程：

- `scripts/check_run_completeness.py` 只依赖标准库，逐个 case_code 对账，
  拒绝漏例、重复、额外用例、算子错配、混合 run_id、损坏行和未知终态。
  `complete=true` 表示记录齐全，`all_pass=true` 才表示全 PASS。
  它不代替运行完整性档案，也不证明每条失败记录都实际执行到了设备内核。
- `ascend-ops-verify.sh` 使用每轮独立的 runs 目录，避免新一轮失败时误拷旧结果；
  归档 completeness JSON，在 report 前执行 score，保留 run/对账/评分/报告的失败退出码，
  并继续走原有上传收尾。

离线复核 Gitcode 上的记录：

```bash
git show origin/verify-records/20261023-093228:verify-records/ascend-ops-20261023-093228.jsonl \
  | python scripts/check_run_completeness.py --results -
```

对一轮新的全量正确性验收：

```bash
python scripts/check_run_completeness.py \
  --manifest corpus/v0.6-ascend-ops/manifest.json \
  --results runs/ascend-ops-<时间戳>/run-<编号>/results.jsonl --require-pass
```

离线工具应使用该轮归档的 manifest；这里只是因为上述验证分支与当前清单一致才可用默认路径。
工具退出码：0=满足对应门禁，1=记录不齐或严格模式下有失败，2=文件/输入无法解析。
以上描述的是初始离线审计阶段。后续实施已新增 v0.7 修订语料与适配提交，
参考算法、用例选择及比较器修正情况见 `docs/ascend-ops-npu-validation.md`。
教程源码及旧 v0.6 语料保持原样；CPU 双参考门禁通过不代表 NPU 220/220 已判对。
