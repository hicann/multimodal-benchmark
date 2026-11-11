# 提交样例

## `demo-submission/` — 合法提交

目录契约（零构建，无 wheel / build.sh）：

```
demo-submission/
├── manifest.yaml              声明算子名、入口、实现路径
└── operators/
    ├── exp/kernel.py          @triton.jit kernel + 同名入口函数
    └── softmax/kernel.py
```

用法（昇腾主机）：

```bash
python -m triton_eval run --profile smoke --submission examples/demo-submission
```

## 对照：`tests/npu_fixtures/cheating_submission/` — 非法提交

直接用 `torch.softmax` 代算，会在**执行前**被静态校验拒绝（SUB-STR-001）。

## 写一个新提交要满足什么

1. `manifest.yaml` 的算子名必须在注册表中存在
2. 入口函数名与参数列表必须**与算子规格完全一致**
3. 实现不得把核心计算交给 torch 内置接口
4. 不得做主机侧往返、地址键缓存、伪张量、计时篡改
5. 布局变换仅 `LayoutTransform` 类允许
