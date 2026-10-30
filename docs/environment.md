# Environment qualification

The integration is being qualified with these exact source revisions:

| Package | Revision |
| --- | --- |
| vLLM | tag `v0.32.0` |
| vLLM-Omni | `45603043133fda3fe486d8e558c226d3391940f0` |
| vLLM-Ascend | `0875da8bc578ab4bd65fa5aa544b212349d0214d` |
| CANN Catlass dependency | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |

The private Ascend qualification environment has Python 3.12.14, Torch
2.10.0+cpu, Torch-NPU 2.10.0.post2 and CANN 9.2.0. The operators were also
built and smoke-tested with CANN 9.0.1, but the selected modern framework
needs the newer runtime device-mapping API. The selected vLLM-Ascend revision declares
Torch-NPU 2.10.0.post4 and Triton-Ascend 3.2.2. The current video tests use
2.10.0.post2, constrained explicitly instead of silently upgrading it.
vLLM-Ascend is installed as `0.32.0rc1` from the pinned source above;
this is a qualified H3 subset of that framework, not qualification of all
model families or of every Torch-NPU patch release.

Apply the explicit Omni compatibility patches after checking out the
revision above:

```bash
python scripts/apply_framework_patches.py --omni /path/to/vllm-omni
```

The first preserves dataclass construction with the pinned vLLM scheduler.
The second permits ordinary layerwise FastH3 fusion while still rejecting
distributed host-weight loading that bypasses the fusion stream. The script
checks the source revision and safely recognizes already-applied patches.
The third places a selected, unquantized text encoder on CPU during online
DiT quantized loading, preventing its initial BF16 allocation from consuming
the accelerator before offload hooks exist. It preserves the encoder's
runtime target device and leaves quantized encoder loading unchanged.

The upstream Ascend Dockerfile additionally pins
`flash-linear-attention-npu-a2==26.9.1+dev5cc6990` from its official release,
SHA256 `2577eca3f1015ca83897bf638046f41cb54691a0c15c433d9e43946fa32d985b`
for the aarch64 wheel. It is imported by the shared worker patch set even
when H3 does not use the package's linear kernels. The wheel requires
`GLIBCXX_3.4.29`; use a compatible C++ runtime in this environment rather
than replacing the host's system library.

On Ascend, build vLLM with `VLLM_TARGET_DEVICE=empty` using the upstream
`tools/use_existing_torch.py --prefix` helper. This keeps the Ascend Torch
version instead of installing the CUDA Torch requirement from the standard
vLLM build. Initialize vLLM-Ascend's fixed Catlass submodule before compiling
native extensions. The setup script defaults to `COMPILE_CUSTOM_KERNELS=0` for the framework.
H3 uses Torch-NPU primitives plus this repository's separately built attention
extensions. Full INT8 TP2 video inference and VAE decoding have passed in
this mode; see the method-specific qualification matrix before selecting a
sparse path. `H3_FULL_FRAMEWORK_BUILD=1` additionally builds the framework's
operator collection, including kernels for unrelated model families.

Install all framework changes in an independent environment. Do not upgrade
another experiment's shared environment in place. Source the selected CANN
`set_env.sh` before native builds and inference; the test host's older default
CANN installation is not suitable for the same Torch-NPU kernels.
When adding this checkout or framework sources to `PYTHONPATH`, prepend
paths without discarding the values installed by `set_env.sh`. CANN's
`python/site-packages` contains `tbe`/`te`, which VAE graph initialization
needs even when precompiled DiT attention has already run successfully.

The historical NVIDIA references used Torch 2.12.0+cu130 on an RTX PRO 6000
Blackwell Server Edition. Their official CUDA source revisions are listed in
`THIRD_PARTY_NOTICES.md`; they are separate from the Ascend runtime stack.

The private CANN 9.2.0 toolkit and 910B operator packages were obtained from
[Huawei's official package mirror](https://mirrors.huaweicloud.com/ascend/repos/conda/linux-aarch64/)
and verified against its SHA256 metadata before offline installation.
The installation does not change the host driver or shared CANN directories.

## Reproduce the source installation

Start from a dedicated Python 3.12 Ascend environment with the Torch stack
above. The script checks that stack, constrains dependency resolution to the
existing accelerator packages, checks out exact framework revisions, applies
the three reviewed patches and installs the project and fixed FLA wheel:

```bash
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

The installation script has syntax and source-revision checks. A clean-room
installation of this consolidated script is not yet qualified; the running
video environment was assembled using the equivalent individual steps.
It does not install the Ascend driver/toolkit, models or a C++ runtime.
An existing checkout at a different revision is rejected rather than reset.
The FLA wheel is checked against its fixed SHA256 before installation.
