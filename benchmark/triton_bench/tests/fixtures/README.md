# 测试夹具

这些文件是昇腾执行机上**真实命令的原始输出**，进版本控制。

作用：把「采集」与「解析」分离之后，L0 层的解析逻辑就能在没有 NPU 的机器上
用这些夹具完整测试。环境一旦采集过一次，就长期可测——这是 CANNLab 这类临时
环境被回收后仍能继续开发的保险。

重新采集：在昇腾主机上执行 `bash env/collect-fixtures.sh`。

| 文件 | 来源 |
|---|---|
| `npu_smi_info.txt` | `npu-smi info` |
| `samples_ms.json` | `triton.testing.do_bench` 的真实返回 |
