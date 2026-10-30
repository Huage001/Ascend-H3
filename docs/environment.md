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
Torch-NPU 2.10.0.post4 and Triton-Ascend 3.2.2; resolving and validating that
version boundary is still in progress. Do not interpret the source revision
list as a tested installation recipe until qualification completes.

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
native extensions. The source-only `COMPILE_CUSTOM_KERNELS=0` option alone
does not qualify runtime inference.

Install all framework changes in an independent environment. Do not upgrade
another experiment's shared environment in place. Source the selected CANN
`set_env.sh` before native builds and inference; the test host's older default
CANN installation is not suitable for the same Torch-NPU kernels.

The historical NVIDIA references used Torch 2.12.0+cu130 on an RTX PRO 6000
Blackwell Server Edition. Their official CUDA source revisions are listed in
`THIRD_PARTY_NOTICES.md`; they are separate from the Ascend runtime stack.

The private CANN 9.2.0 toolkit and 910B operator packages were obtained from
[Huawei's official package mirror](https://mirrors.huaweicloud.com/ascend/repos/conda/linux-aarch64/)
and verified against its SHA256 metadata before offline installation.
The installation does not change the host driver or shared CANN directories.
