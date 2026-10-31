"""File-backed masters for the input-only vision tower, initialized before timing.

T2VA never executes vision blocks. All tensor values and native metadata remain
available; conditioned requests may page them in, but are not performance tested.
Active text and DiT blocks retain the original pinned transfer/prefetch policy.
"""

import ctypes
import json
import os
import tempfile
from pathlib import Path


def mapped_master(size, dtype, directory):
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="dormant-vision-", suffix=".bin", dir=directory)
    try:
        tensor = torch.from_file(name, shared=True, size=size, dtype=dtype)
        return tensor, fd, Path(name)
    except BaseException:
        os.close(fd)
        os.unlink(name)
        raise


def release_file_pages(tensor, fd, path):
    # Publish dirty bytes to the file, unmap physical pages from RSS, then
    # discard clean file cache. The virtual tensor and exact data persist.
    libc = ctypes.CDLL(None, use_errno=True)
    address = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    for name, flag in [("msync", 4), ("madvise", 4)]:
        call = getattr(libc, name)
        call.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        call.restype = ctypes.c_int
        if call(address, nbytes, flag):
            raise OSError(ctypes.get_errno(), name)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    path.unlink()  # mapping keeps its private inode until server exit


def vision_to_cpu(params, bufs, lb, directory):
    weights = {}
    metadata = {}
    targets = []
    files = []
    try:
        for dtype, named in lb.group_named_tensors_by_dtype(params, bufs).items():
            specs = lb.describe_tensor_storage(named)
            size = sum(s.storage_numel for s in specs)
            tensor, fd, path = mapped_master(size, dtype, directory)
            offset = 0
            for spec in specs:
                tensor[offset : offset + spec.storage_numel].copy_(
                    lb.flatten_physical_storage(spec.value, spec.storage_numel)
                )
                metadata.setdefault(dtype, []).append(
                    lb.tensor_storage_metadata(spec, offset, include_device=True)
                )
                targets.append(spec.target)
                offset += spec.storage_numel
            weights[dtype] = tensor
            files.append((tensor, fd, path))
        # Finish I/O before touching original storage, preserving rollback.
        for tensor, fd, path in files:
            release_file_pages(tensor, fd, path)
        lb.clear_tensor_storage(targets)
    except BaseException:
        for _, fd, path in files:
            try:
                os.close(fd)
            except OSError:
                pass
            path.unlink(missing_ok=True)
        raise
    return weights, metadata


def install():
    if os.getenv("H3_ASCEND_DORMANT_VISION_MMAP") != "1":
        return
    from vllm_omni.diffusion.offloader import layerwise_backend as lb

    original = lb.LayerwiseOffloadHook._to_cpu
    if getattr(original, "_h3_dormant_vision_mmap", False):
        return
    from h3_ascend.paths import cache_root

    directory = cache_root() / "dormant-vision-masters"
    total = [0]
    count = [0]

    def to_cpu(params, bufs, pin_memory=True):
        # Native Qwen3-VL vision block signature; text uses self_attn.qkv_proj
        # and H3 DiT uses attn.qkv_proj. Fail closed on unfamiliar names.
        if "attn.qkv.weight" not in params or "mlp.linear_fc1.weight" not in params:
            return original(params, bufs, pin_memory)
        size = sum(
            v.numel() * v.element_size() for v in list(params.values()) + list(bufs.values())
        )
        result = vision_to_cpu(params, bufs, lb, directory)
        total[0] += size
        count[0] += 1
        audit = os.getenv("H3_ASCEND_AUDIT_DIR")
        if audit:
            p = Path(audit) / f"dormant-vision-{os.getpid()}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    dict(
                        vision_blocks=count[0],
                        file_backed_bytes=total[0],
                        active_text_and_dit_offload_unchanged=True,
                        exact_storage_values=True,
                    )
                )
            )
        return result

    to_cpu._h3_dormant_vision_mmap = True
    lb.LayerwiseOffloadHook._to_cpu = staticmethod(to_cpu)
