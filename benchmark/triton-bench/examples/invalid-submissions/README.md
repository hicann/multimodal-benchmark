# 非法提交样例集

每个目录对应**一条**规则，是"什么会被拒"的可执行说明。
提交方照着改、评测方照着测，规则表里写的东西才不会停留在文档里。

用法：

```bash
python -m triton_eval validate examples/invalid-submissions/<目录名>
```

每个目录都必须被拒绝，且报出对应的规则编号——
`tests/test_invalid_submissions.py` 把这件事钉死。

## 静态阶段（SUB-STR-\*）：不需要执行，看源码就能判定

| 目录 | 规则 | 反例在做什么 |
|---|---|---|
| `SUB-STR-001` | 内置接口代算 | 直接 `torch.exp` 算完返回（最典型的是同名代算） |
| `SUB-STR-002` | 主机侧往返 | `x.cpu()` → 计算 → 搬回加速器，耗时几乎全是拷贝 |
| `SUB-STR-003` | 地址键缓存 | `dict[x.data_ptr()]`，输入轮换后复用陈旧结果 |
| `SUB-STR-004` | 布局搬运越权 | 非布局类算子用 `permute/contiguous` 绕道 |
| `SUB-STR-005` | 干预计时 | 把 `synchronize` 替换成空操作，计时测不到真实耗时 |
| `SUB-STR-006` | 源码不可解析 | 语法错误；不可分析等价于不可信 |

## 加载阶段（SUB-LOD-\*）：需要导入实现文件

| 目录 | 规则 | 反例在做什么 |
|---|---|---|
| `SUB-LOD-001` | 清单缺失 | 没有 `manifest.yaml` |
| `SUB-LOD-002` | 未登记算子 | 清单声明了注册表里没有的算子 |
| `SUB-LOD-003` | 实现文件不存在 | 清单指向的文件不在目录里 |
| `SUB-LOD-004` | 入口缺失 | 文件能导入，但没有清单声明的入口函数 |
| `SUB-LOD-005` | 入口不可调用 | 同名变量把函数遮蔽成了常量 |
| `SUB-LOD-006` | 参数个数不符 | 比规格少两个参数 |
| `SUB-LOD-007` | 参数名不符 | `scale` 写成 `factor`，属性会静默丢失 |

`SUB-LOD-006/007` 的样例刻意不 import triton：这两条只关乎签名，
不依赖能否真编译——这样在任何装了 torch 的机器上都能验证。

## 执行阶段（SUB-EXE-\*）：只有真跑一遍才暴露

| 规则 | 反例在做什么 | 验证方式 |
|---|---|---|
| `SUB-EXE-001` | 返回未通过 Tensor 实例检查的对象；Tensor 子类不再仅因类型身份被拒收 | `tests/runtime/test_exec_runtime_guard.py` |
| `SUB-EXE-002` | 非布局类算子返回非连续输出 | `tests/test_exec_runtime_guard.py` |

这两条不建目录：它们需要构造返回值对象，不是"一份提交"能表达的。
