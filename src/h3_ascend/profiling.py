"""Idle-time, opt-in device-event profiling, separate from formal latency runs.

Create CONTROL/start only after all timed requests finish. Each DiT worker
writes CONTROL/ready-PID.json after installing hooks; CONTROL/ready remains
for single-worker compatibility. No profiling hook runs before that.
"""

import json
import os
import threading
import time
import weakref
from pathlib import Path

_FORWARD_MODELS = []


def register_live_model(model):
    """Remember an actually executed model, including deserialized TP workers."""
    if not any(reference() is model for reference in _FORWARD_MODELS):
        _FORWARD_MODELS.append(weakref.ref(model))


def _select_live_models(constructed):
    executed = [reference() for reference in _FORWARD_MODELS if reference() is not None]
    candidates = executed or [reference() for reference in constructed if reference() is not None]
    return list({id(model): model for model in candidates}.values())


def install():
    directory = os.getenv("H3_ASCEND_PROFILE_CONTROL")
    if not directory:
        return
    import torch
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    if getattr(MiniMaxH3DiTModel, "_h3_profile_control", False):
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    instances = []
    original = MiniMaxH3DiTModel.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        instances.append(weakref.ref(self))

    MiniMaxH3DiTModel.__init__ = initialize
    MiniMaxH3DiTModel._h3_profile_control = True

    def watch():
        while not (root / "start").is_file():
            time.sleep(1)
        # Timed requests are finished. Release unreachable response objects
        # before the independent profiling request; no weights are offloaded
        # differently and no cleanup is inserted in a timed forward.
        import ctypes
        import gc

        unreachable = gc.collect()
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        trimmed = trim(0) if trim is not None else None
        (root / "pre-profile-gc.json").write_text(
            json.dumps(
                {
                    "unreachable": unreachable,
                    "malloc_trim": trimmed,
                    "formal_measurement_already_complete": True,
                }
            )
        )
        models = _select_live_models(instances)
        if not models:
            # TP launchers and helpers can import the class without hosting
            # a DiT. The harness requires readiness from every audited worker.
            (root / f"ignored-no-dit-{os.getpid()}.json").write_text(
                json.dumps({"pid": os.getpid(), "model_count": 0, "ignored": True})
            )
            return
        if len(models) != 1:
            (root / "error.json").write_text(
                json.dumps({"pid": os.getpid(), "model_count": len(models)})
            )
            return
        model = models[0]
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        rank_metadata = {
            "pid": os.getpid(),
            "tp_rank": get_tensor_model_parallel_rank(),
            "tp_world_size": get_tensor_model_parallel_world_size(),
            "local_attention_heads": [block.attn.num_heads for block in model.blocks],
        }
        kind = next(
            (p.device.type for p in model.parameters() if p.device.type in ("cuda", "npu")), None
        )
        if kind is None:
            raise RuntimeError("No accelerator-resident model state found for profiling")
        device = torch.get_device_module(kind)
        rows = []
        handles = []
        started = [None]
        forward_index = [0]

        def span(module, category, layer):
            pending = []

            @torch.compiler.disable
            def before(module, args):
                event = device.Event(enable_timing=True)
                event.record()
                pending.append(event)

            @torch.compiler.disable
            def after(module, args, result):
                event = device.Event(enable_timing=True)
                event.record()
                rows.append((category, layer, pending.pop(), event))

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))

        for index, block in enumerate(model.blocks):
            span(block.attn, "attention", index)
            span(block.mlp, "ffn", index)
            for name, category in (
                ("qkv_proj", "qkv_projection"),
                ("attention", "attention_dispatch"),
                ("out_proj", "output_projection"),
            ):
                child = getattr(block.attn, name, None)
                if child is not None:
                    span(child, category, index)

        @torch.compiler.disable
        def begin(module, args):
            device.synchronize()
            started[0] = time.perf_counter()

        handles.append(model.register_forward_pre_hook(begin))

        @torch.compiler.disable
        def flush(module, args, result):
            device.synchronize()
            forward_index[0] += 1
            values = [
                dict(category=c, layer=i, milliseconds=a.elapsed_time(b)) for c, i, a, b in rows
            ]
            rows.clear()
            with (root / f"events-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            **rank_metadata,
                            "profile_forward_index": forward_index[0],
                            "spans": values,
                            "instrumented": True,
                            "dit_wall_seconds": time.perf_counter() - started[0],
                        }
                    )
                    + "\n"
                )

        handles.append(model.register_forward_hook(flush))
        # Keep handles alive; the server exits after the separate profiling run.
        model._h3_profile_handles = handles
        ready = {**rank_metadata, "device": str(device)}
        ready_file = root / f"ready-{os.getpid()}.json"
        staged = ready_file.with_suffix(".tmp")
        staged.write_text(json.dumps(ready))
        staged.replace(ready_file)
        (root / "ready").write_text(json.dumps(ready))

    threading.Thread(target=watch, daemon=True, name="h3-profile-idle-control").start()
