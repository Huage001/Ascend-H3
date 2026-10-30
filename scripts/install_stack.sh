#!/usr/bin/env bash
# Prepare the pinned H3 framework sources in an EXISTING Ascend environment.
# No driver/toolkit, weights, system libraries or other environments are changed.
set -euo pipefail
H3_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
H3_STACK_ROOT=${1:?Usage: bash scripts/install_stack.sh /absolute/path/to/framework-sources}
H3_PYTHON=${H3_PYTHON:-python}
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh first}"
"$H3_PYTHON" - <<'PY'
import sys
import torch
import torch_npu
import triton
assert sys.version_info >= (3, 12), 'Python 3.12+ is required'
assert torch.__version__.split('+')[0] == '2.10.0', torch.__version__
assert torch_npu.__version__.startswith('2.10.0'), torch_npu.__version__
print('Using existing Ascend Torch stack', torch.__version__, torch_npu.__version__, triton.__version__)
PY
mkdir -p "$H3_STACK_ROOT"
H3_STACK_ROOT=$(cd -- "$H3_STACK_ROOT" && pwd)
checkout() {
  local name=$1 url=$2 revision=$3 folder="$H3_STACK_ROOT/$1"
  if [[ ! -e "$folder" ]]; then
    git clone --filter=blob:none "$url" "$folder"
    git -C "$folder" checkout --detach "$revision"
  fi
  [[ $(git -C "$folder" rev-parse HEAD) == "$revision" ]] || {
    echo "Unexpected checkout: $folder (expected $revision)" >&2
    exit 1
  }
}
checkout vllm https://github.com/vllm-project/vllm.git 5df63a7203cdb9328d824a27b682e77c7b9928be
checkout vllm-omni https://github.com/vllm-project/vllm-omni.git 45603043133fda3fe486d8e558c226d3391940f0
checkout vllm-ascend https://github.com/vllm-project/vllm-ascend.git 0875da8bc578ab4bd65fa5aa544b212349d0214d
"$H3_PYTHON" "$H3_REPO_ROOT/scripts/apply_framework_patches.py" --omni "$H3_STACK_ROOT/vllm-omni"
# Freeze the existing accelerator distributions for dependency resolution.
"$H3_PYTHON" - "$H3_STACK_ROOT/accelerator-constraints.txt" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
names=['torch','torch-npu','triton-ascend','torchvision','torchaudio']
Path(sys.argv[1]).write_text('\n'.join(f'{n}=={version(n)}' for n in names)+'\n')
PY
"$H3_PYTHON" -m pip install 'setuptools>=78.1.1,<81' 'setuptools-scm>=8' wheel 'cmake>=3.26' pybind11 ninja
(
  cd "$H3_STACK_ROOT/vllm"
  "$H3_PYTHON" tools/use_existing_torch.py --prefix
  "$H3_PYTHON" -m pip install -c "$H3_STACK_ROOT/accelerator-constraints.txt" -r requirements/common.txt
  VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0 \
    "$H3_PYTHON" -m pip install --no-build-isolation --no-deps -e .
)
"$H3_PYTHON" -m pip install -c "$H3_STACK_ROOT/accelerator-constraints.txt" -r "$H3_REPO_ROOT/requirements/h3-runtime.txt"
SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0 \
  "$H3_PYTHON" -m pip install --no-build-isolation --no-deps -e "$H3_STACK_ROOT/vllm-omni"
# H3 uses Torch-NPU primitives plus this repository's separately built CANN
# libraries. Optional framework kernels for other model families can be built
# with H3_FULL_FRAMEWORK_BUILD=1; this does not replace `h3-ascend build`.
if [[ ${H3_FULL_FRAMEWORK_BUILD:-0} == 1 ]]; then
  git -C "$H3_STACK_ROOT/vllm-ascend" submodule update --init csrc/third_party/catlass
  git -C "$H3_STACK_ROOT/vllm-ascend/csrc/third_party/catlass" checkout 41bf90da655bba3c66d0acd7e00abe33960ecfd6
fi
COMPILE_CUSTOM_KERNELS=${H3_FULL_FRAMEWORK_BUILD:-0} SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0rc1 \
  "$H3_PYTHON" -m pip install --no-build-isolation --no-deps -e "$H3_STACK_ROOT/vllm-ascend"
"$H3_PYTHON" -m pip install --no-build-isolation --no-deps -e "$H3_REPO_ROOT"
H3_PYTHON="$H3_PYTHON" bash "$H3_REPO_ROOT/scripts/install_fla.sh" "$H3_STACK_ROOT/downloads"
echo 'Framework sources installed. Run h3-ascend build --method all before inference.'
