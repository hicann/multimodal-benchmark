"""用例执行。

    npu_exec.py —— L1：编译与执行被测 kernel，只能在昇腾主机运行
    （终态判定不在本模块，而在 triton_eval/judge.py，属 L0）

说明：全仓库 import triton 的 L1 模块共三处——本模块、bench/measure.py
（do_bench 计时）、envprobe/collect.py（读实测 import 版本，包在 try 中可降级）。
由 tests/core/test_contract.py 用 AST 机器校验，防止分层被侵蚀。
"""
