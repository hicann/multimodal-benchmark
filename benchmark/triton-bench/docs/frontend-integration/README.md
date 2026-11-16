# 前端联调契约包

开发方案见 [前端与接口联调方案](../frontend-platform-plan.md)。

- `openapi.json`：当前 Python API 生成的 OpenAPI 3.1 快照；运行中的 `/openapi.json` 是现行接口来源。
- `examples.json`：无 Worker 的真实本地服务响应，ID/时间归一化。没有远端执行与成功评分样例，不能充当 NPU 验收证据。HTTP 状态来自现有适配器约定与 ServiceError；该导出本身不执行 HTTP/浏览器请求。
- `submission-valid.zip`（由下述命令生成，不纳入版本控制）：通过静态校验的 Exp 示例，只返回输入；仅用于上传联调，数学实现不正确。

在仓库根目录、已有项目依赖的 Python 环境中运行：

```sh
PYTHONPATH=src python scripts/platform_examples/export_frontend_contract.py
PYTHONPATH=src python scripts/platform_examples/export_frontend_contract.py --check
```

导出只创建临时本地服务，不启用执行器、网络或云实例。不要手改生成文件；后端契约变化后重新生成，并核对前端类型、事件解析与展示口径。缺少的未来接口见方案中的待实施清单。
