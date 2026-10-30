def full_attention(q, k, v, scale=None):
    """[tokens,local_heads,dim], no padding tokens among valid rows."""
    import torch

    scale = scale if scale is not None else q.shape[-1] ** -0.5
    if q.device.type in ("npu", "privateuseone"):
        import torch_npu

        # Include the layout conversions in caller measurements.
        qs, ks, vs = [x.transpose(0, 1)[None].contiguous() for x in (q, k, v)]
        y = torch_npu.npu_fused_infer_attention_score(
            qs,
            ks,
            vs,
            num_heads=q.shape[1],
            num_key_value_heads=k.shape[1],
            input_layout="BNSD",
            scale=scale,
        )[0]
        return y[0].transpose(0, 1).contiguous()
    with torch.nn.attention.sdpa_kernel(
        [
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        ]
    ):
        y = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            scale=scale,
        )
    return y[0].transpose(0, 1).contiguous()
