"""Idle-time, opt-in device-event profiling, separate from formal latency runs.

Create CONTROL/start only after all timed requests finish. The worker writes
CONTROL/ready after installing hooks. No profiling hook runs before that.
"""

import json
import os
import threading
import time
import weakref
from pathlib import Path


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
        models = [ref() for ref in instances if ref() is not None]
        if len(models) != 1:
            (root / "error.json").write_text(json.dumps({"model_count": len(models)}))
            return
        model = models[0]
        kind = next((p.device.type for p in model.parameters() if p.device.type in ("cuda", "npu")), None)
        if kind is None:
            raise RuntimeError("No accelerator-resident model state found for profiling")
        device = torch.get_device_module(kind)
        rows = []
        handles = []
        started = [None]

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
                ("qkv_proj", "qkv_projection"), ("attention", "attention_dispatch"),
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
            values = [dict(category=c, layer=i, milliseconds=a.elapsed_time(b))
                      for c, i, a, b in rows]
            rows.clear()
            with (root / f"events-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps({"spans": values, "instrumented": True,
                                         "dit_wall_seconds": time.perf_counter() - started[0]}) + "\n")

        handles.append(model.register_forward_hook(flush))
        # Keep handles alive; the server exits after the separate profiling run.
        model._h3_profile_handles = handles
        (root / "ready").write_text(json.dumps({"pid": os.getpid(), "device": str(device)}))

    threading.Thread(target=watch, daemon=True, name="h3-profile-idle-control").start()
