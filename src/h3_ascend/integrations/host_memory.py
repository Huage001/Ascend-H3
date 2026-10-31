"""Release unused loading allocations before building pinned CPU masters.

This only runs while constructing the layerwise offloader. It does not change
master dtype, resident layers, prefetching or any timed forward execution.
"""

import ctypes
import gc


def install_initialization_cleanup():
    from vllm_omni.diffusion.offloader.layerwise_backend import (
        LayerwiseOffloadHook,
        LayerWiseOffloadBackend,
    )

    if getattr(LayerWiseOffloadBackend.enable, "_h3_initialization_cleanup", False):
        return
    libc = ctypes.CDLL(None)
    trim = getattr(libc, "malloc_trim", None)
    if trim is not None:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int

    def cleanup():
        gc.collect()
        if trim is not None:
            trim(0)

    original_enable = LayerWiseOffloadBackend.enable
    original_to_cpu = LayerwiseOffloadHook._to_cpu

    def enable(self, pipeline):
        cleanup()
        return original_enable(self, pipeline)

    def to_cpu(*args, **kwargs):
        # The preceding block's original CPU storage has just been released.
        # Return those unused pages before allocating the next pinned master.
        cleanup()
        return original_to_cpu(*args, **kwargs)

    enable._h3_initialization_cleanup = True
    LayerWiseOffloadBackend.enable = enable
    LayerwiseOffloadHook._to_cpu = staticmethod(to_cpu)
