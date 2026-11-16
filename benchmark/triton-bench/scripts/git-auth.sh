#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# Validate a configured remote without copying credentials into argv or logs.
require_safe_git_remote() {
  local target="${1:-origin}" address
  address="$(git remote get-url --push "$target" 2>/dev/null)" || address="$target"
  printf '%s' "$address" | python3 -c '
import sys
from urllib.parse import urlsplit, parse_qsl
try:
    u = urlsplit(sys.stdin.read())
    bad = u.scheme in ("http", "https") and (u.username or u.password)
    bad = bad or any(k.lower() in {"token","access_token","private_token","password","secret","auth","api_key"} for k,v in parse_qsl(u.query))
except ValueError:
    bad = True
if bad:
    sys.stderr.write("拒绝含凭据的仓库 URL；请使用 SSH 或 Git credential helper。\n")
    sys.exit(1)
'
}
