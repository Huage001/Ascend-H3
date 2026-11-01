"""Reuse the existing W8A8 input quantizer before head-local QKV projection."""


def gather_input(x, group, world, *, packet="separate"):
    import torch
    import torch.distributed as dist
    import torch_npu

    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Expected contiguous BF16 [local_tokens, hidden]")
    quantized, scale = torch_npu.npu_dynamic_quant(x)
    scale = scale.reshape(-1)
    if scale.dtype != torch.float32 or scale.numel() != x.shape[0]:
        raise ValueError("Expected native FP32 per-token scales")
    if packet == "separate":
        gathered = torch.empty((world * x.shape[0], x.shape[1]), device=x.device, dtype=torch.int8)
        scales = torch.empty(world * x.shape[0], device=x.device, dtype=torch.float32)
        work_q = dist.all_gather_into_tensor(gathered, quantized, group=group, async_op=True)
        work_s = dist.all_gather_into_tensor(scales, scale, group=group, async_op=True)
        work_q.wait()
        work_s.wait()
        return gathered, scales
    if packet != "joint512":
        raise ValueError(packet)
    data_bytes = quantized.numel()
    scale_bytes = scale.numel() * 4
    size = ((data_bytes + scale_bytes + 511) // 512) * 512
    send = torch.zeros(size, device=x.device, dtype=torch.uint8)
    send[:data_bytes].copy_(quantized.reshape(-1).view(torch.uint8))
    send[data_bytes : data_bytes + scale_bytes].copy_(scale.view(torch.uint8))
    received = torch.empty((world, size), device=x.device, dtype=torch.uint8)
    dist.all_gather_into_tensor(received.reshape(-1), send, group=group)
    gathered = (
        received[:, :data_bytes]
        .contiguous()
        .view(torch.int8)
        .reshape(world * x.shape[0], x.shape[1])
    )
    scales = (
        received[:, data_bytes : data_bytes + scale_bytes]
        .contiguous()
        .view(torch.float32)
        .reshape(-1)
    )
    return gathered, scales


def shard_weight(weight, scale, rank, world, *, heads=56, width=128):
    """Prepare once, after loading, LoRA fusion and weight quantization."""
    import torch

    if weight.shape != (5376, 3 * heads * width) or scale.numel() != 3 * heads * width:
        raise ValueError("Expected post-loader Q-all/K-all/V-all INT8 QKV weight")
    count = heads // world
    if heads % world or not 0 <= rank < world:
        raise ValueError((heads, rank, world))
    sections = [
        slice((part * heads + rank * count) * width, (part * heads + (rank + 1) * count) * width)
        for part in range(3)
    ]
    return (
        torch.cat([weight[:, section] for section in sections], dim=1).contiguous(),
        torch.cat([scale.reshape(-1)[section] for section in sections]).contiguous(),
    )


def project(quantized, scales, weight, weight_scale, heads):
    import torch
    import torch_npu

    value = torch_npu.npu_quant_matmul(
        quantized,
        weight,
        weight_scale,
        pertoken_scale=scales,
        output_dtype=torch.bfloat16,
    )
    return tuple(t.reshape(value.shape[0], heads, 128) for t in value.split(heads * 128, dim=-1))
