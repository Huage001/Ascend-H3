"""Initialize the pinned official SM120 FA4 provider before Omni imports it."""

import faulthandler
import os
import runpy
import sys


def main():
    if os.getenv("H3_ASCEND_CUDA_REFERENCE") != "1":
        raise RuntimeError("This launcher is only for the explicit single-card CUDA reference")
    faulthandler.enable()
    sys.path.insert(0, os.environ["H3_CUDA_REFERENCE_CODE"])
    from dense_backend import _load_fa4

    fa4, _, _ = _load_fa4(os.environ["MIOWTION_ROOT"])
    if fa4._modules() is None or not fa4.fa4_sm8x.installed():
        raise RuntimeError(f"Official SM120 FA4 initialization failed: {fa4.unavailable_reason()}")
    from h3_ascend.integrations.host_memory import install_initialization_cleanup

    install_initialization_cleanup()
    runpy.run_module("vllm_omni.entrypoints.cli.main", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
