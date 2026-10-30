"""vLLM-Omni plugin: reuse upstream loaders, INT8 linears and TP/Ulysses.

The plugin is inert unless H3_ASCEND_METHOD is explicitly selected.
"""

import importlib.util
import os
import re
from collections import OrderedDict

_PLANS = OrderedDict()
_REGISTERED = False


def register():
    global _REGISTERED
    method = os.getenv("H3_ASCEND_METHOD")
    if _REGISTERED or not method:
        return
    if method not in ("full", "vsa", "veda", "vdn", "spark"):
        raise ValueError(f"unknown H3_ASCEND_METHOD={method}")
    if importlib.util.find_spec("torch_npu") is None:
        if os.getenv("H3_ASCEND_CUDA_REFERENCE") != "1":
            return
        from h3_ascend.audit import install as install_audit
        from h3_ascend.integrations.cuda_reference import register as register_cuda
        from h3_ascend.integrations.lora import install as install_lora
        from h3_ascend.profiling import install as install_profiling

        register_cuda(method)
        install_lora()
        install_audit()
        install_profiling()
        _REGISTERED = True
        return
    import torch
    import torch_npu  # noqa: F401 -- registers torch.npu before it is used

    torch.npu.matmul.allow_hf32 = False
    from vllm_omni.diffusion.attention.backends.registry import (
        DiffusionAttentionBackendEnum as E,
    )
    from vllm_omni.diffusion.attention.backends.registry import (
        register_diffusion_backend,
    )

    if method == "vsa":
        register_diffusion_backend(
            E.FASTVIDEO_VSA, "h3_ascend.integrations.vllm_omni.AscendVSABackend"
        )
        from vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3 import (
            MiniMaxH3VSAImpl,
        )

        MiniMaxH3VSAImpl.forward_npu = _vsa_forward
    elif method == "vdn":
        register_diffusion_backend(
            E.VDNH3_ATTN, "h3_ascend.integrations.vllm_omni.AscendVDNBackend"
        )
        from h3_ascend.backends.vdn import install

        install()
    else:
        register_diffusion_backend(
            E.TORCH_SDPA, "h3_ascend.integrations.vllm_omni.AscendSparseBackend"
        )
    from h3_ascend.integrations.lora import install as install_lora_fusion

    install_lora_fusion()
    from h3_ascend.audit import install as install_audit

    install_audit()
    from h3_ascend.profiling import install as install_profiling

    install_profiling()
    _REGISTERED = True


def _head_shard(local):
    from h3_ascend.context import HeadShard

    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
    except (AssertionError, RuntimeError):
        tp, rank = 1, 0
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_ulysses_parallel_rank,
            get_ulysses_parallel_world_size,
        )

        up, ur = get_ulysses_parallel_world_size(), get_ulysses_parallel_rank()
    except (AssertionError, RuntimeError):
        up, ur = 1, 0
    if 56 % (tp * up) or local != 56 // (tp * up):
        raise ValueError(f"head shard does not match TP={tp}, Ulysses={up}, local={local}")
    return HeadShard(rank * (56 // tp) + ur * local, local)


def _valid_length(q, metadata):
    if metadata is not None and metadata.packed_padding is not None:
        return int(metadata.packed_padding.q_length)
    return q.shape[1]


def _layout(q, metadata):
    from h3_ascend.context import VideoLayout

    vl = metadata.video_layout if metadata is not None else None
    if vl is None:
        return None
    targets = [s for s in vl.video_spans if s.role == "target"]
    if len(targets) != 1 or len(vl.video_spans) != 1:
        raise ValueError("this release supports one T2VA target, without reference video spans")
    span = targets[0]
    f, h, w = map(int, span.latent_grid)
    prefix = int(span.start)
    if prefix + f * h * w != _valid_length(q, metadata):
        raise ValueError("target must cover the end of the valid packed T2VA document")
    frames = {37: 124, 102: 345}.get(f)
    if frames is None:
        raise ValueError(f"unsupported native latent frames {f}")
    segments = tuple(metadata.extra.get("vsa_h3_prefix_segments", ()))
    return VideoLayout((f, h, w), prefix, prefix_segments=segments, frames=frames)


def _dense(q, k, v, metadata):
    import torch

    from h3_ascend.backends.full import full_attention

    if metadata is not None and metadata.attn_mask is not None:
        # Token refiners can still carry a real Q/K mask. Preserve it through
        # the upstream implementation rather than treating it as padding.
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

        return SDPAImpl(q.shape[2], q.shape[3], q.shape[3] ** -0.5).forward_npu(q, k, v, metadata)
    n = _valid_length(q, metadata)
    if q.shape[0] != 1:
        raise ValueError("this release uses batch=1; do not co-pack independent prompts")
    nk = (
        int(metadata.packed_padding.kv_length)
        if metadata is not None and metadata.packed_padding is not None
        else k.shape[1]
    )
    y = full_attention(q[0, :n].contiguous(), k[0, :nk].contiguous(), v[0, :nk].contiguous())
    if n == q.shape[1]:
        return y[None]
    out = torch.zeros_like(q)
    out[0, :n] = y
    return out


def _vsa_forward(self, q, k, v, attn_metadata=None):
    import torch

    from h3_ascend.backends.vsa import attention

    layout = _layout(q, attn_metadata)
    if layout is None:
        return _dense(q, k, v, attn_metadata)
    if attn_metadata.extra.get("vsa_h3_tile_size", 64) != 64:
        raise ValueError("Ascend paired VSA uses logical tile64, not tile128")
    gate = attn_metadata.extra.get("gate_compress")
    if gate is None:
        raise ValueError("VSA checkpoint gate is absent; refusing a dense fallback")
    n = layout.sequence_length
    sparsity = attn_metadata.extra.get("vsa_h3_sparsity")
    # The exported Ascend VSA path uses the historical 90% route budget when
    # a legacy student supplies no ratio. Do not inherit Omni's fixed top-64
    # default, whose effective sparsity changes with the video resolution.
    if sparsity is None:
        sparsity = 0.9
    y = attention(
        q[0, :n].contiguous(),
        k[0, :n].contiguous(),
        v[0, :n].contiguous(),
        gate[0, :n].contiguous(),
        layout,
        keep_rate=1.0 - float(sparsity),
    )
    out = torch.zeros_like(q)
    out[0, :n] = y
    return out


# Backend classes are imported lazily by the upstream registry.
def __getattr__(name):
    if name == "AscendVSABackend":
        from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
            FastVideoVSABackend,
        )

        class AscendVSABackend(FastVideoVSABackend):
            supported_platforms = ("npu",)

            @classmethod
            def supports_packed_mask_free(cls):
                return True

            @classmethod
            def validate_available(cls):
                from h3_ascend.native import library_path

                if not library_path("vsa_paired64").is_file():
                    raise ImportError("build Ascend VSA with h3-ascend build --method vsa")

        globals()[name] = AscendVSABackend
        return AscendVSABackend
    if name == "AscendVDNBackend":
        from vllm_omni.diffusion.attention.backends.vdnh3_attn import (
            VDNAttentionBackend,
        )

        class AscendVDNBackend(VDNAttentionBackend):
            supported_platforms = ("npu",)

            @classmethod
            def supports_packed_mask_free(cls):
                return True

            @classmethod
            def validate_available(cls):
                return None

        globals()[name] = AscendVDNBackend
        return AscendVDNBackend
    if name == "AscendSparseBackend":
        import torch
        from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

        class AscendSparseImpl(SDPAImpl):
            def __init__(self, *args, prefix="", **kwargs):
                super().__init__(*args, prefix=prefix, **kwargs)
                self.prefix = prefix
                matches = re.findall(r"(?:blocks|transformer_blocks)\.(\d+)", prefix)
                self.layer = int(matches[-1]) if matches else 0

            @torch.compiler.disable
            def forward_npu(self, q, k, v, attn_metadata=None):
                method = os.environ["H3_ASCEND_METHOD"]
                if method == "full":
                    return _dense(q, k, v, attn_metadata)
                layout = _layout(q, attn_metadata)
                if layout is None:
                    return _dense(q, k, v, attn_metadata)
                if attn_metadata.attn_mask is not None:
                    raise ValueError(
                        "sparse H3 requires the validated mask-free single-document path"
                    )
                shard = _head_shard(q.shape[2])
                key = (
                    method,
                    layout,
                    shard,
                    str(q.device),
                    int(torch.npu.current_stream(q.device).npu_stream),
                )
                n = layout.sequence_length
                if key not in _PLANS:
                    if method == "spark":
                        from h3_ascend.backends.spark import SparkAttention

                        _PLANS[key] = SparkAttention(layout)
                    elif method == "veda":
                        from h3_ascend.backends.veda import VedaAttention

                        _PLANS[key] = VedaAttention(layout, shard, device=str(q.device))
                    else:
                        raise ValueError(method)
                _PLANS.move_to_end(key)
                while len(_PLANS) > int(os.getenv("H3_ASCEND_PLAN_CACHE_SIZE", "1")):
                    _PLANS.popitem(last=False)
                y = _PLANS[key](
                    q[0, :n].contiguous(),
                    k[0, :n].contiguous(),
                    v[0, :n].contiguous(),
                    layer=self.layer,
                )
                out = torch.zeros_like(q)
                out[0, :n] = y
                return out

        class AscendSparseBackend(SDPABackend):
            supported_platforms = ("npu",)

            @staticmethod
            def get_name():
                return "H3_ASCEND"

            @staticmethod
            def get_impl_cls():
                return AscendSparseImpl

            @classmethod
            def supports_multi_doc_packed_varlen(cls):
                return False

        globals()[name] = AscendSparseBackend
        return AscendSparseBackend
    raise AttributeError(name)
