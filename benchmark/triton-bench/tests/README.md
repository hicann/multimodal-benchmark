# 测试说明

## 分层

| 目录 | 需要 NPU | 说明 |
|---|---|---|
| `tests/test_*.py`（除 test_npu） | 否 | L0 逻辑，macOS / Linux 通用 |
| `tests/test_npu.py` | 是 | 执行层，自动打 `npu` 标记 |
| `tests/npu_fixtures/` | 是 | **负向用例**，不是 corpus 的一部分 |

## 为什么需要负向用例

全 PASS 只能证明「链路通了」，不能证明「判定是对的」。

`npu_fixtures/` 放的是**故意出错**的 kernel，用于验证系统能否正确检出问题：

| 目录 | 期望终态 | 验证什么 |
|---|---|---|
| `silent_wrong/` | `WRONG_RESULT` | 静默错误能否被检出（最高优先级风险信号） |
| `compile_fail/` | `COMPILE_FAIL` | 编译失败能否被正确归类，而非笼统归为运行失败 |

如果 `silent_wrong` 被判成 PASS，说明整套判定逻辑失效，
此前的「20/20 全 PASS」也就没有意义了。

**这些文件不属于冻结的 corpus**，只用于测试，不参与评分。
