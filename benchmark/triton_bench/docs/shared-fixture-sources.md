# 示例提交的共享源与独立物化

## 一、维护方式

重复的示例与模板不再逐份维护。`scripts/example_sources/layout.json` 显式声明逻辑文件、共享源、组装后 AST 摘要和 JIT 函数摘要；共享函数通过构建期的具名导入复用。教程加载器直接复用已有历史源码，不改写该历史文件。

这不是用户 ZIP 的新依赖，也不是删除评测场景。`triton_eval.fixture_sources` 在可信仓库内展开源码，输出仍包含完整函数定义。物化后的 ZIP 不导入 `scripts.example_sources`、`scripts.templates` 或评测仓库的源码。

## 二、构建与运行

现有提交仿真、官方套件入口和示例发现流程已经接入物化。需要手动复制、修改或压缩示例时，应先导出独立目录，不应直接压缩尚未展开的维护目录：

```bash
PYTHONPATH=src python3 -m triton_eval.fixture_sources \
  examples/coverage-v18-ascend-submission /tmp/coverage-v18-export
```

目标目录必须不存在，命令不会覆盖已有内容。导出目录可以修改、压缩并作为普通用户提交。用户上传包的目录约定、签名校验、反作弊判定和评分口径均不改变。

构建器的默认输出仍是仓库中的目标目录，不会向临时只读视图写入。物化缓存不是工作目录，不应直接修改；缓存内生成源码被修改时，后续读取会拒绝复用。

## 三、身份与验证边界

注册表校验全部共享源摘要，拒绝路径穿越、符号链接、缺失文件及未登记的构建导入。只处理可信仓库 `examples` 与 `scripts/templates` 下的逻辑路径；用户目录即使伪造同名注册表也不会启用源码展开。

JIT 函数的 AST 与整改前逐个比对；纯 Python 的 Softmax、Cummin 公共标量框架另做数值与边界回归。缓存、Tensor 子类、错误 dtype、忽略 dim、错误索引和错误返回个数等对抗行为继续保留，不把错误实现改成正确实现。

共享源与注册表纳入当前资产绑定。旧资产索引保留在认证历史中，用例清单、历史参考源码及评分定义不改写。新的资产修订明确标记 `hardware_validation=NOT_RUN`；本机 AST/CPU 检查不构成新的编译认证、NPU 执行认证或 CodeCheck 远程关闭结论。
