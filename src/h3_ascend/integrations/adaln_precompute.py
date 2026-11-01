"""Exact startup-schedule AdaLN preparation; keep projection weights on CPU.

Opt-in until the NPU integration and complete-request checks have passed.
Uses the original projection math, scheduler and versioned projection cache.
"""

import json
import os
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace


class _HostProjectionCache:
    offload_weights = True

    def __init__(self, cache):
        self.cache = cache

    def __getattr__(self, name):
        return getattr(self.cache, name)


class _CountedStager:
    def __init__(self, stager):
        self.stager = stager
        self.loads = 0

    @property
    def device(self):
        return self.stager.device

    def load(self):
        self.stager.load()
        self.loads += 1

    def offload(self):
        self.stager.offload()


def _versioned_parameters(module):
    """Retain version checks even if the loader created inference tensors."""
    import torch
    from vllm_omni.quantization._copy_missing_attrs import copy_missing_attrs

    for name, previous in list(module.named_parameters(recurse=False)):
        if not previous.is_inference():
            continue
        with torch.inference_mode(False):
            value = previous.detach().clone()
            parameter = torch.nn.Parameter(value, requires_grad=previous.requires_grad)
            copy_missing_attrs(previous, parameter)
            assert torch.equal(value.view(torch.uint8), previous.view(torch.uint8))
            module.register_parameter(name, parameter)


def _cache_time_embeddings(model):
    """Memoize native TP time embeddings from their exact original inputs.

    FP32 reductions on four/eight ranks need not return identical low bits on
    repeated evaluations. Bind reuse before that reduction, with the same
    native version/settings checks and all-rank hit vote as leaf projections.
    """
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.cache.exact_projection_cache import ExactProjectionCache

    if get_tensor_model_parallel_world_size() <= 2:
        return None
    embedder = model.time_embedder
    existing = getattr(embedder, "_h3_input_cache", None)
    if existing is not None:
        existing.clear()
        return existing
    for child in embedder.modules():
        _versioned_parameters(child)
    cache = ExactProjectionCache(max_bytes=1024**2)
    original = embedder.forward

    def cached_forward(timesteps):
        cache.prepare(timesteps)
        return cache.project("time_embedder", embedder, timesteps, lambda: original(timesteps))

    embedder._h3_input_cache = cache
    embedder.forward = cached_forward
    return cache


def resolve_startup_schedule(pipeline, requested_steps=None):
    """Resolve the loaded model/merged student's real native denoise schedule."""
    if requested_steps is not None and (type(requested_steps) is not int or requested_steps < 1):
        raise ValueError("Startup DiT steps must be a positive integer")
    sampling = SimpleNamespace(num_inference_steps=requested_steps, lora_request=None, lora_scale=1.0)
    positions, steps = pipeline._resolve_sigma_positions("t2va", sampling)
    if requested_steps is not None and steps != requested_steps:
        raise ValueError(f"Loaded student requires {steps} DiT updates; requested {requested_steps}")
    return dict(mode="t2va", num_steps=steps, base_schedule=positions,
                flow_shift=pipeline.default_video_shift, audio_flow_shift=pipeline.default_audio_shift)


def prepare(model, contract, *, projection_override=None):
    """Load each projection once, compute each native timestep shape, unload."""
    import torch
    from vllm_omni.diffusion.models.minimax_h3.adaln_cache import timestep_plans

    if projection_override is None:
        projections = [block.adaln_proj for block in model.blocks]
        projections.append(model.final_layer.adaln_proj)
        assert len(projections) == 51
    else:
        projections = list(projection_override)
    assert all(getattr(p, "_h3_host_adaln", False) for p in projections)
    cache = model.adaln_cache
    cache.clear()
    device = next(model.time_embedder.parameters()).device
    assert device.type == "npu"
    with torch.inference_mode():
        time_cache = _cache_time_embeddings(model)
        timesteps = [t.to(device) for t in timestep_plans(contract)]
        if time_cache is not None:
            time_cache.max_bytes = max(time_cache.max_bytes, sum(t.numel() for t in timesteps)
                                       * model.time_embedder.proj_out.output_size * 4)
        embeddings = [model.time_embedder(t) for t in timesteps]
        assert embeddings
        if time_cache is not None:
            before_hits, before_misses = time_cache.hits, time_cache.misses
            for t, reference in zip(timesteps, embeddings, strict=True):
                value = model.time_embedder(t)
                assert torch.equal(value.view(torch.int32), reference.view(torch.int32))
            assert time_cache.hits - before_hits == len(timesteps)
            assert time_cache.misses == before_misses
        records = []
        for projection in projections:
            _versioned_parameters(projection.linear)
            projection._prepare_weight_stager(device)
            stager = projection._weight_stager
            before = cache._signature(projection.linear)
            loads_before_preparation = stager.loads
            stager.load()
            try:
                # Separate M=1/M=2 calls preserve the serving GEMM shape and
                # SiLU/downcast order. Do not combine steps into a larger GEMM.
                outputs = [
                    projection.linear(torch.nn.functional.silu(t).to(torch.bfloat16))[0]
                    for t in embeddings
                ]
            finally:
                stager.offload()
            assert cache._signature(projection.linear) == before
            # Retain every resolved plan, including longer teacher schedules.
            cache.max_bytes = max(cache.max_bytes, cache._bytes + sum(
                output.numel() * output.element_size() for output in outputs
            ))
            for embedding, output in zip(embeddings, outputs, strict=True):
                assert bool(torch.isfinite(output).all())
                cache.prepare(embedding)
                value = cache.project(
                    projection._cache_name, projection.linear, embedding,
                    lambda output=output: output,
                )
                assert torch.equal(value.view(torch.int16), output.view(torch.int16))
            loads = stager.loads
            hits = cache.hits
            for embedding, output in zip(embeddings, outputs, strict=True):
                cache.prepare(embedding)
                value = projection._project(embedding)
                assert torch.equal(value.view(torch.int16), output.view(torch.int16))
            assert stager.loads == loads and cache.hits - hits == len(embeddings)
            assert all(p.device.type == "cpu" for p in projection.linear.parameters())
            records.append({
                "name": projection._cache_name, "plans": len(embeddings),
                "weight_bytes_cpu": sum(p.numel() * p.element_size() for p in projection.linear.parameters()),
                "preparation_loads": stager.loads - loads_before_preparation,
                "verification_cache_hits": cache.hits - hits,
                "bitexact": True,
            })
            del outputs
            if len(records) % 10 == 0 or len(records) == len(projections):
                print("H3_ADALN_PREPARED", len(records), "/", len(projections), flush=True)
        torch.npu.synchronize(device)
    result = {
        "status": "prepared", "contract": contract, "records": records,
        "cache_bytes_npu": cache._bytes, "cache_entries": len(cache._entries),
        "cache_budget_bytes_npu": cache.max_bytes,
        "weight_bytes_cpu": sum(r["weight_bytes_cpu"] for r in records),
        "all_projection_outputs_bitexact": True,
        "cache_hits_at_preparation": cache.hits, "cache_misses_at_preparation": cache.misses,
        "time_embedding_shapes": [list(t.shape) for t in embeddings],
        "time_input_cache": ({"entries": len(time_cache._entries), "bytes_npu": time_cache._bytes,
                              "hits": time_cache.hits, "misses": time_cache.misses}
                             if time_cache is not None else None),
    }
    model._h3_adaln_preparation = result
    cache._input = None
    cache._key = None
    directory = os.getenv("H3_ASCEND_ADALN_RECEIPTS")
    if directory:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"prepared-{os.getpid()}.json").write_text(json.dumps(result, indent=2) + "\n")
        if projection_override is None:
            _idle_snapshots(model, root)
    return result


def runtime_stats(model):
    if not hasattr(model, "_h3_adaln_preparation"):
        return None
    projections = [b.adaln_proj for b in model.blocks] + [model.final_layer.adaln_proj]
    parameters = [p for projection in projections for p in projection.linear.parameters()]
    excluded = {id(p) for p in parameters}
    cache = model.adaln_cache
    time_cache = getattr(model.time_embedder, "_h3_input_cache", None)
    result = {
        "cache_hits": cache.hits, "cache_misses": cache.misses,
        "cache_bytes_npu": cache._bytes, "cache_entries": len(cache._entries),
        "weight_stage_loads": sum(p._weight_stager.loads for p in projections),
        "adaln_weights_cpu": all(p.device.type == "cpu" for p in parameters),
        "adaln_weight_bytes_cpu": sum(p.numel() * p.element_size() for p in parameters),
        "other_dit_parameters_npu": all(p.device.type == "npu" for p in model.parameters() if id(p) not in excluded),
        "other_dit_parameter_bytes_npu": sum(p.numel() * p.element_size() for p in model.parameters() if id(p) not in excluded),
        "time_input_cache": ({"entries": len(time_cache._entries), "bytes_npu": time_cache._bytes,
                              "hits": time_cache.hits, "misses": time_cache.misses}
                             if time_cache is not None else None),
    }
    state = getattr(model, "_h3_input_rope_state", None)
    if state is not None:
        result["input_projection"] = {
            "forward": state["forward"], "rope_builds": state["rope_builds"],
            "rope_gathers": state["rope_gathers"],
            "full_rope_per_rank_cached": state["full_rope"] is not None,
            "projection_calls": [block.attn._h3_input_calls for block in model.blocks],
            "all_layers_live_qkv_bitexact": all(block.attn._h3_input_audit_steps == 8 for block in model.blocks),
        }
    return result


def _idle_snapshots(model, directory):
    """Read counters only when the harness requests an idle-time snapshot."""
    reference = weakref.ref(model)

    def watch():
        previous = None
        request = directory / "snapshot-request.json"
        while True:
            time.sleep(0.25)
            instance = reference()
            if instance is None:
                return
            try:
                modified = request.stat().st_mtime_ns
                if modified == previous:
                    continue
                payload = json.loads(request.read_text())
                nonce = payload["nonce"]
                if not isinstance(nonce, str) or not nonce.replace("-", "").replace("_", "").isalnum():
                    raise ValueError("Invalid idle snapshot nonce")
                result = dict(pid=os.getpid(), nonce=nonce, time=time.time(), **runtime_stats(instance))
                path = directory / f"snapshot-{nonce}-{os.getpid()}.json"
                temporary = path.with_suffix(".json.new")
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(path)
                previous = modified
            except FileNotFoundError:
                pass

    threading.Thread(target=watch, daemon=True, name="adaln-idle-snapshots").start()


def install():
    if os.getenv("H3_ASCEND_PRECOMPUTE_ADALN") != "1":
        return
    import torch
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.models.minimax_h3.adaln_cache import schedule_contract
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3AdalnProj
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.offloader.module_residency import (
        BoundedAllocatorCache,
        PinnedModuleStager,
    )
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

    if getattr(MiniMaxH3AdalnProj, "_h3_precompute_installed", False):
        return
    original_init = MiniMaxH3AdalnProj.__init__
    original_stager = MiniMaxH3AdalnProj._prepare_weight_stager
    original_load = DiffusionModelRunner.load_model

    def initialize(self, arch, out_features, quant_config, *, expand_ratio, modality_num, prefix, adaln_cache=None):
        if adaln_cache is None or not adaln_cache.max_bytes:
            raise ValueError("Exact AdaLN preparation requires the native projection cache")
        if quant_config is not None and quant_config.get_name() != "int8":
            raise ValueError("NPU AdaLN preparation currently supports BF16 and native INT8")
        local_width = out_features // get_tensor_model_parallel_world_size()
        host_cache = _HostProjectionCache(adaln_cache)
        # Preserve TP1's existing BF16 fallback; do not change its quantization.
        unquantized = quant_config is None or local_width > 65535
        original_init(
            self, arch, out_features, None if unquantized else quant_config,
            expand_ratio=expand_ratio, modality_num=modality_num, prefix=prefix,
            adaln_cache=host_cache if unquantized else adaln_cache,
        )
        if not unquantized:
            method = self.linear.quant_method
            if not getattr(method, "supports_offload_after_quant", False):
                raise ValueError("AdaLN quantizer cannot finalize its weights on CPU")
            method.enable_offload_after_quant()
            self._offload_weights = True
            self._adaln_cache = host_cache
        self._h3_host_adaln = True

    def prepare_stager(self, device):
        if not getattr(self, "_h3_host_adaln", False) or device.type != "npu":
            return original_stager(self, device)
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            raise RuntimeError("NPU AdaLN preparation requires eager inference")
        _versioned_parameters(self.linear)
        if self._adaln_cache._has_forward_hooks(self.linear):
            raise RuntimeError("AdaLN projection hooks prevent exact memoization")
        signature = self._offload_signature()
        if signature is None:
            raise RuntimeError("AdaLN weights lack a usable immutable signature")
        if self._weight_stager is None or self._weight_stager.device != device or self._host_signature != signature:
            if any(p.device.type != "cpu" for p in self.linear.parameters()):
                raise RuntimeError("AdaLN master weights must finish loading on CPU")
            self._weight_stager = _CountedStager(
                PinnedModuleStager(self.linear, device, cache_retention=BoundedAllocatorCache(device))
            )
            self._host_signature = self._offload_signature()

    def load(runner, *args, **kwargs):
        result = original_load(runner, *args, **kwargs)
        pipeline = runner.pipeline
        if isinstance(pipeline, MiniMaxH3Pipeline):
            value = os.getenv("H3_ASCEND_ADALN_STEPS")
            requested = int(value) if value is not None else None
            contract = schedule_contract(**resolve_startup_schedule(pipeline, requested))
            prepare(pipeline.transformer, contract)
        return result

    MiniMaxH3AdalnProj.__init__ = initialize
    MiniMaxH3AdalnProj._prepare_weight_stager = prepare_stager
    DiffusionModelRunner.load_model = load
    MiniMaxH3AdalnProj._h3_precompute_installed = True
