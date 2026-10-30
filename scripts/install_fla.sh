#!/usr/bin/env bash
# Official worker-import dependency for the pinned aarch64 Ascend stack.
set -euo pipefail
H3_PYTHON=${H3_PYTHON:-python}
H3_DOWNLOAD_ROOT=${1:?Usage: bash scripts/install_fla.sh /absolute/path/to/downloads}
[[ $(uname -m) == aarch64 ]] || { echo 'This wheel is for Linux aarch64.' >&2; exit 1; }
mkdir -p "$H3_DOWNLOAD_ROOT"
H3_FLA_WHEEL=flash_linear_attention_npu_a2-26.9.1+dev5cc6990-py3-none-manylinux_2_34_aarch64.whl
H3_FLA_PATH="$H3_DOWNLOAD_ROOT/$H3_FLA_WHEEL"
if [[ ! -f "$H3_FLA_PATH" ]]; then
  curl --fail --location --retry 2 --connect-timeout 15 \
    "https://github.com/flashserve/flash-linear-attention-npu/releases/download/v26.9.1-20261020-020004/$H3_FLA_WHEEL" \
    --output "$H3_FLA_PATH"
fi
echo "2577eca3f1015ca83897bf638046f41cb54691a0c15c433d9e43946fa32d985b  $H3_FLA_PATH" | sha256sum --check
"$H3_PYTHON" -m pip install --no-deps "$H3_FLA_PATH"
