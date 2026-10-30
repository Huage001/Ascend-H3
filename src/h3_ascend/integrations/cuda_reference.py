"""Optional benchmark bridge to pinned official CUDA attention sources.

No NPU kernels or new CUDA compute kernels are used here. The reference uses
the same Omni model, loading, quantization and Offload policy as Ascend.
Set H3_CUDA_REFERENCE_CODE to a directory containing the hash-pinned NVIDIA
experiment's dense_backend.py, veda_backend.py and spark_backend.py adapters;
their original source/dependency pins remain enforced by those adapters.
"""

import os
import re
import sys
from dataclasses import replace

_PLANS = {}


def register(method):
    import torch
    from vllm_omni.diffusion.attention.backends.registry import (
        DiffusionAttentionBackendEnum as E,
    )
    from vllm_omni.diffusion.attention.backends.registry import register_diffusion_backend

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA reference requires a real NVIDIA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    root = os.environ["H3_CUDA_REFERENCE_CODE"]
    sys.path.insert(0, root)
    from dense_backend import _load_fa4

    fa4, _, _ = _load_fa4(os.environ["MIOWTION_ROOT"])
    if fa4._modules() is None:
        raise RuntimeError("Pinned Miowtion FA4 provider could not initialize")
    if method in ("full", "veda", "spark"):
        register_diffusion_backend(
            E.TORCH_SDPA, "h3_ascend.integrations.cuda_reference.CUDAReferenceBackend"
        )
    elif method == "vsa":
        from vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3 import MiniMaxH3VSAImpl

        original = MiniMaxH3VSAImpl.forward_cuda

        def vsa(self, q, k, v, attn_metadata=None):
            if attn_metadata is not None:
                attn_metadata.extra.setdefault("vsa_h3_sparsity", .9)
            return original(self, q, k, v, attn_metadata)

        MiniMaxH3VSAImpl.forward_cuda = vsa


def __getattr__(name):
    if name != "CUDAReferenceBackend":
        raise AttributeError(name)
    import torch
    from vllm_omni.diffusion.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionImpl,
    )

    class CUDAReferenceImpl(FlashAttentionImpl):
        def __init__(self, *args, prefix="", **kwargs):
            super().__init__(*args, prefix=prefix, **kwargs)
            match = re.findall(r"(?:blocks|transformer_blocks)\.(\d+)", prefix)
            self.layer = int(match[-1]) if match else 0

        @torch.compiler.disable
        def forward_cuda(self, q, k, v, attn_metadata=None):
            from h3_ascend.integrations.vllm_omni import _layout

            method = os.environ["H3_ASCEND_METHOD"]
            layout = _layout(q, attn_metadata)
            if layout is None:
                return super().forward_cuda(q, k, v, attn_metadata)
            if attn_metadata.attn_mask is not None:
                raise ValueError("Reference T2VA video requires a mask-free packed document")
            n = layout.sequence_length
            key = (method, layout, str(q.device))
            if key not in _PLANS:
                f, h, w = layout.grid
                if method == "spark":
                    from spark_backend import Spark

                    backend = Spark(dict(seq=n, G=layout.prefix, F=f, grid_h=h, grid_w=w), .1)
                    backend.config = replace(backend.config, sol_log_density=False)
                    backend.controller.config = backend.config
                else:
                    from miowtion.h3.geometry import geometry_from_latent_t

                    geo = geometry_from_latent_t("16:9", f)
                    audio = geo.num_audio_rows
                    config = dict(width=w * 32, height=h * 32, frames=layout.frames,
                                  raw_latent_shape=(f, h * 2, w * 2), video_grid=(f, h, w),
                                  prefix_tokens=layout.prefix, text_tokens=layout.prefix - audio,
                                  audio_tokens=audio, video_tokens=f * h * w, seq_len=n)
                    if method == "full":
                        from dense_backend import DenseBackend

                        backend = DenseBackend(config, source_root=os.environ["MIOWTION_ROOT"],
                                               implementation="fa4")
                    else:
                        from veda_backend import VedaBackend

                        backend = VedaBackend(config, source_root=os.environ["MIOWTION_ROOT"],
                                              predictor_path=os.environ["H3_ASCEND_VEDA_PREDICTOR"],
                                              precision="bf16")
                _PLANS[key] = backend
            backend = _PLANS[key]
            inputs = [x[:, :n] for x in (q, k, v)]
            if method == "spark":
                result = backend(*inputs, layer=self.layer)
            else:
                result = backend.attention(*inputs, x=None, q_pre=None, k_pre=None,
                                           rope=None, layer=self.layer)
            if n == q.shape[1]:
                return result
            out = torch.zeros_like(q)
            out[:, :n] = result
            return out

    class CUDAReferenceBackend(FlashAttentionBackend):
        supported_platforms = ("cuda",)

        @classmethod
        def get_impl_cls(cls):
            return CUDAReferenceImpl

    globals()[name] = CUDAReferenceBackend
    return CUDAReferenceBackend
