# 测试集发布：corpus-v0.5

> 正式发布。版本规则见 docs/versioning.md；本页由 scripts/release.py 生成，与标签指向的提交对应。

## 版本快照（五轴）

| 轴 | 版本 |
|---|---|
| 测试集 | v0.5-cov20-frozen |
| 评测规范 | 0.1-mvp |
| 评分器 | 0.2.3 |
| 容差表 | 1.2 |
| 能力点清单 | 1.1-fusion |

## 冻结内容

- 语料目录：`corpus/v0.5-cov20-frozen`
- 清单哈希：`2771f8b88970fdbcc23b1f9c91dd11315724f8f2c5747ddf34803bfdf3715688`
- 冻结时间：2026-10-13T09:34:27.876149+00:00
- 用例 289 个 / 算子 25 个 / 门禁剔除 0 条
- 生成器提交：`50b6fe55773d5b86c882e6e5a061d574540110ee`；构建环境：Python 3.11.9 / torch 2.14.1 / macOS-26.2-arm64-arm-64bit（取自 generation-log.jsonl，共 9 条构建事件）
- 清单内容落地提交：`3a9ec1ef8f0569fa8a97ca33defab41d9b891220`

## 与上一版的差异

- 用例数 240 → 289，算子 +11 / −0，用例 +92 / −43 / 改 25
- 逐项差异（算子增删 / 用例级增删改 / 规模与 dtype 再分配 / 覆盖原子得失）见 docs/corpus-evolution.md

## 受影响勘误

- **ERR-005（AMEND）**：MoE 门控参考实现的张量落在 CPU（缺 device=），评测子进程崩溃
- **ERR-006（AMEND）**：跨实现求和顺序噪声：f32 大 K 下 tl.dot 与 torch matmul 的固有分歧
- **ERR-007（ANNOTATE）**：交叉熵参考在 npu:0 上同进程两遍结果不一致（锚点自噪声）
- **ERR-008（ANNOTATE）**：三个原语算子的 f16-XL 用例编译失败（疑统一缓冲区容量同类）
- 勘误全文与处置规则见 docs/errata.md

## 验证状态（发布时点）

- 真机验证记录 15 份（verify-records/，含逐项判定与全量测试日志）
- 验证结论索引：docs/requirements-coverage-report.md「真机验证与证据归档」一节
