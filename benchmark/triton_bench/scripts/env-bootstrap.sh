#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 评测环境自举（被 remote-verify.sh 与 standard-window.sh 共用，在仓库根目录 source）。
#
# 做四件事：source 统一环境脚本（注入 CANN、锁单卡、屏蔽用户级包）→
# 定位/创建含 triton+torch_npu 的 venv → 激活 → 处理 PYTHONNOUSERSITE。
# pip install -e . 不在这里做——两个调用方的日志口径不同，各自处理。
# 环境建不起来会 exit 1（sourced 脚本里 exit 即退出调用方，正是想要的）。

source env/env.sh || true

VENV="${TRITON_EVAL_VENV:-}"
if [ -z "$VENV" ]; then
    # find 扫常见根下 4 层内的 venv（activate 是标志），逐个测「triton+torch_npu
    # 都可导入」。平台预装容器里 venv 常在 /opt/<厂商>/venv/<名字>，深两层。
    while IFS= read -r act; do
        d="$(dirname "$(dirname "$act")")"
        if [ -x "$d/bin/python" ] && "$d/bin/python" -c "import triton, torch_npu" 2>/dev/null; then
            VENV="$d"; break
        fi
    done < <(find "$HOME/venv" /opt/atomgit /opt/conda "$HOME/miniconda3" \
                 -maxdepth 4 -path '*/bin/activate' 2>/dev/null)
fi

if [ -z "$VENV" ]; then
    if python3 -c "import triton, torch_npu" 2>/dev/null; then
        # 平台预装容器：系统 python 已带完整编译器环境（torch/torch_npu/triton），
        # 直接建继承式 venv 即可。不走 setup-venv.sh——它的洁净检查会把
        # 「venv 能看到系统 triton」当污染拦下（对预装机器是误报）。
        VENV="${VENV_DIR:-$HOME/venv/te-compiler}"
        echo "[env] 系统 python 已有编译器环境，建继承式 venv: $VENV"
        python3 -m venv --system-site-packages "$VENV" \
            || { echo "[env] venv 创建失败"; exit 1; }
    else
        echo "[env] 没找到含 triton+torch_npu 的现成环境，先用 env/setup-venv.sh 建一个（需要网络）"
        # 关键：判定用的 `2>/dev/null` 会吞掉真实原因。系统 python 明明装了
        # torch_npu / triton 却在 source 过 CANN set_env.sh 之后导入失败时，
        # 不打出来就只能靠猜（换机到 CANN 9.0.0 后正撞上这种情况）。
        echo "[env] 导入失败的真实原因（重要，据此判断能否直接复用系统环境）："
        # 必须 || true：本脚本被 source 进 `set -euo pipefail` 的主脚本，
        # 而这条命令注定非 0（正是因为它失败才走到这里）；配合 pipefail 会让
        # 主脚本当场退出，后续的 setup-venv.sh 根本没机会执行。
        python3 -c "import triton, torch_npu" 2>&1 | tail -8 || true
        echo "[env] 本机已有的候选 torch_npu："
        find /usr /opt "$HOME" -maxdepth 5 -name torch_npu -type d 2>/dev/null | head -5 || true
        # -x 打印每条命令，方便从日志看出究竟卡/失败在哪一步
        bash -x env/setup-venv.sh \
          || { echo "[env] 建环境失败：若上面出现网络/超时报错，说明本机到华为镜像不通，"
               echo "[env] 需离线提供 triton-ascend wheel 或改用可访问的镜像源。"; exit 1; }
        VENV="$HOME/venv/te-compiler"
    fi
fi
echo "[env] 使用 venv: $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 被测编译器必须是 triton-ascend，不能是社区 triton：
# best_practice 的 5 个算子（DecodeGroupedAttention / CausalConv1d /
# GatherScatter / BinnedGatherScatter / PaddedGatherScatter）依赖
# triton.language.extra.cann，社区版 triton 没有这个模块。
# 上面「系统 python 能 import triton」只说明装了 triton，不代表装对了 ——
# 真机曾据此选中系统社区 triton 3.2.0，5 个算子加载失败 → 提交校验不通过 →
# 整轮 0 用例。这里补齐判定，缺则按 env/requirements-compiler.txt 安装。
if ! python -c "import triton.language.extra.cann" 2>/dev/null; then
    echo "[env] 当前 triton 缺 CANN 语言扩展，安装被测编译器 triton-ascend ..."
    # 顺序：本地 wheel（最快、不依赖网络）→ 联网 --no-deps → 联网完整安装。
    # 这么排是因为真机到华为镜像的网络并不可靠：实测 pip 空转 14 分钟
    #（累计仅 12 秒 CPU、缓存零增长）却无任何提示。先离线可彻底绕开网络。
    # wheel 备份由 scripts/save-compiler-wheel.sh 维护。
    WHEEL_DIR="${TRITON_EVAL_WHEEL_DIR:-/opt/atomgit/wheels}"
    if ls "${WHEEL_DIR}"/triton_ascend-*.whl >/dev/null 2>&1; then
        echo "[env] 使用本地 wheel 备份离线安装：${WHEEL_DIR}"
        pip install --no-deps "${WHEEL_DIR}"/triton_ascend-*.whl \
          || echo "[env][WARN] 本地 wheel 安装失败，下面改用联网安装"
    fi
    # --no-deps：torch / torch_npu / CANN 平台已预装，放任 pip 自行解析依赖会去
    # 拉几个 GB 的 torch 轮子。--timeout/--retries：pip 默认无限重试，网络不通时
    # 会静默空转，加了才能快速失败。不用 tail 截断输出，进度是卡死的判断依据。
    if ! python -c "import triton.language.extra.cann" 2>/dev/null; then
        pip install --no-deps --timeout 30 --retries 2 -r env/requirements-compiler.txt \
          || echo "[env][WARN] --no-deps 联网安装未成功，下面退回完整安装"
    fi
    if ! python -c "import triton.language.extra.cann" 2>/dev/null; then
        echo "[env] 仍缺 CANN 扩展，改为完整安装（会解析依赖，较慢）..."
        pip install --timeout 30 --retries 2 -r env/requirements-compiler.txt \
          || echo "[env][ERROR] triton-ascend 安装失败，5 个 best_practice 算子仍会加载失败"
    fi
fi

# 无论本轮是否需要安装，都确保 wheel 备份存在（脚本内部幂等，已有则跳过）。
# 环境每两小时清空一次：备份留在 /opt 下，下一轮即便 venv 被重置也能离线秒装，
# 不必再赌网络——联网安装实测要 20 分钟以上，甚至卡死到超时。
bash "$(dirname "${BASH_SOURCE[0]}")/save-compiler-wheel.sh" 2>/dev/null || true
# PYTHONNOUSERSITE 屏蔽 ~/.local 是防「用户级社区 triton 覆盖被测编译器」。
# 平台容器的 torch_npu 有时恰好在 ~/.local——此时若用户目录里没有 triton，
# 解除屏蔽是安全的（TRITON_EVAL_KEEP_USER_SITE=1 显式授权）。
if [ "${TRITON_EVAL_KEEP_USER_SITE:-0}" = "1" ] \
   && ! ls "$HOME"/.local/lib/python*/site-packages/triton* >/dev/null 2>&1; then
    unset PYTHONNOUSERSITE
    echo "[env] 已解除 ~/.local 屏蔽（用户目录无 triton，需要用户级 torch_npu）"
else
    export PYTHONNOUSERSITE=1
fi
