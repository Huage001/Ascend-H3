def attention(q, k, v, gate, layout, keep_rate=0.1, topk=None, **kwargs):
    if gate is None:
        raise ValueError("VSA requires the trained compression gate; use a VSA checkpoint/LoRA")
    if q.shape != gate.shape:
        raise ValueError("VSA gate must match this rank's Q head shard")
    if not layout.prefix_segments:
        raise ValueError("VSA requires separate text/audio prefix segment lengths")
    from h3_ascend._vendor.vsa.npu_vsa64_ascendc import forward

    return forward(
        q,
        k,
        v,
        gate,
        layout.prefix_segments,
        layout.grid,
        group=2,
        prefix_fa=True,
        options={"route_stack_tiles": 4, "route_layout": "BNSD", "route_joint": False},
        keep_rate=keep_rate,
        topk=topk,
    )
