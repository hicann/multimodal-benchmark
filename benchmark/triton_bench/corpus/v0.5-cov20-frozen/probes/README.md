# 收益实证探针（评测方资产，v0.5）

## 这是什么

优化手法原子（A17 类）是**性能负载型**标签：用例谁都能算对，标签声明
「使用该手法与不使用存在可测量的性能差距」。声明不等于事实——每条
原子必须有一对探针内核（朴素写法 vs 手法写法）在真机上实测收益，
证据归档；不达标的原子在覆盖率披露中如实降级为「声明性考察」。

## 口径

- 达标：手法版比朴素版快 ≥ 1.5 倍的**用例占比 ≥ 80%**
  （每条原子在其覆盖用例的代表性形状上测 3 个形状）；
- 计时：与评测 standard 档同口径——warmup 后计时取中位数；
- 探针**不进用例清单、不参与语料指纹**（它只服务标签可信度，
  不影响考题）；随测试集版本化存放；
- 探针两版都须算对同一道题（跑批脚本含 torch 对拍），否则实证无效——
  测「少干活」不是测优化。

## 探针对与原子的对应（同载体复用，结论逐原子独立出具）

| 探针对 | 载体形状 | 出具结论的原子 |
|---|---|---|
| tile_gemm.py | [1024,1024]、[3072,128]、[2048,2048] | FA-PAT-TILE-GEMM、FA-PAT-VEC-BROADCAST |
| split_k.py | [64,65536] | FA-PAT-SPLIT-K |
| static_unroll.py | 8 组分组矩阵乘 | FA-CTRL-STATIC-UNROLL |
| vec_load.py | [65536]（对齐）、[99971]（非对齐） | FA-PAT-VEC-LOAD、FA-PAT-MASK-TAIL |
| multipass_reduce.py | [1048576] 归约 | FA-PAT-MULTIPASS-REDUCE |
| coalesce_gather.py | [1048576] gather | FA-PAT-COALESCE-GATHER |
| scan_blocked.py | [1048576] 前缀扫描 | FA-PAT-SCAN-BLOCKED |

FA-CONCUR-ACCUM 是**正确性负载**原子（顺序无关累加是语义要求），
不需要收益实证，不在本目录范围。

## 真机跑批

    source scripts/env-bootstrap.sh
    python scripts/probe_validation.py            # 全部探针 → probes-report.{json,md}

在无 NPU 的机器上运行会直接退出（探针内核需要 triton 编译执行）。
