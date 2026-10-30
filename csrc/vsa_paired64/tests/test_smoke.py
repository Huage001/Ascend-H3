import os

import pytest
import torch
import torch_npu

from block_sparse_attention import block_sparse_attention


@pytest.mark.parametrize("kv_block_size", [64, 128, 192, 256])
@pytest.mark.parametrize("input_layout", ["BNSD", "TND"])
def test_forward_kv_block_multiple_of_64(kv_block_size, input_layout):
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    batch, heads, seq_len, head_dim = 1, 2, 576, 128
    q_block_size = 128
    options = dict(dtype=torch.bfloat16, device=device)
    input_shape = (
        (batch, heads, seq_len, head_dim)
        if input_layout == "BNSD"
        else (seq_len, heads, head_dim)
    )
    q = torch.randn(input_shape, **options)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    q_blocks = (seq_len + q_block_size - 1) // q_block_size
    kv_blocks = (seq_len + kv_block_size - 1) // kv_block_size
    mask_cpu = torch.zeros(batch, heads, q_blocks, kv_blocks, dtype=torch.bool)
    for q_block in range(q_blocks):
        mask_cpu[:, :, q_block, q_block % kv_blocks] = True
        mask_cpu[:, :, q_block, (q_block + 2) % kv_blocks] = True
    mask = mask_cpu.to(dtype=torch.int8, device=device)

    sequence_arguments = (
        {}
        if input_layout == "BNSD"
        else {
            "actual_seq_lengths": [seq_len],
            "actual_seq_lengths_kv": [seq_len],
        }
    )
    output = block_sparse_attention(
        q,
        k,
        v,
        mask,
        block_shape=(q_block_size, kv_block_size),
        input_layout=input_layout,
        num_key_value_heads=heads,
        scale_value=head_dim**-0.5,
        **sequence_arguments,
    )

    token_mask = (
        mask_cpu.repeat_interleave(q_block_size, dim=2)
        .repeat_interleave(kv_block_size, dim=3)[..., :seq_len, :seq_len]
    )
    if input_layout == "BNSD":
        q_cpu = q.float().cpu()
        k_cpu = k.float().cpu()
        v_cpu = v.float().cpu()
        output_cpu = output.float().cpu()
    else:
        q_cpu = q.permute(1, 0, 2).unsqueeze(0).float().cpu()
        k_cpu = k.permute(1, 0, 2).unsqueeze(0).float().cpu()
        v_cpu = v.permute(1, 0, 2).unsqueeze(0).float().cpu()
        output_cpu = output.permute(1, 0, 2).unsqueeze(0).float().cpu()
    scores = torch.matmul(q_cpu, k_cpu.transpose(-1, -2)) * head_dim**-0.5
    expected = torch.matmul(
        scores.masked_fill(~token_mask, float("-inf")).softmax(-1), v_cpu
    )

    torch.testing.assert_close(
        output_cpu, expected, rtol=8e-3, atol=8e-3
    )


def test_forward_and_backward():
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    shape = (1, 8, 512, 128)
    options = dict(dtype=torch.bfloat16, device=device)
    q = torch.randn(shape, **options, requires_grad=True)
    k = torch.randn(shape, **options, requires_grad=True)
    v = torch.randn(shape, **options, requires_grad=True)
    mask = torch.randint(0, 2, (1, 8, 4, 4), dtype=torch.int8)
    mask[..., torch.arange(4), torch.arange(4)] = 1
    mask = mask.to(device)
    grad = torch.randn(shape, **options)

    scale = 128**-0.5
    q_ref = q.detach().clone()
    k_ref = k.detach().clone()
    v_ref = v.detach().clone()
    out = block_sparse_attention(
        q, k, v, mask, input_layout="BNSD", num_key_value_heads=8,
        scale_value=scale,
    )
    out_ref = torch_npu.npu_block_sparse_attention(
        q_ref,
        k_ref,
        v_ref,
        mask,
        [128, 128],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=8,
        scale_value=scale,
        inner_precise=0,
        softmax_lse_flag=1,
    )[0]
    q_cpu = q.detach().float().cpu().requires_grad_(True)
    k_cpu = k.detach().float().cpu().requires_grad_(True)
    v_cpu = v.detach().float().cpu().requires_grad_(True)
    token_mask = mask.bool().cpu().repeat_interleave(128, 2).repeat_interleave(128, 3)
    scores = torch.matmul(q_cpu, k_cpu.transpose(-1, -2)) * scale
    out_cpu = torch.matmul(scores.masked_fill(~token_mask, float("-inf")).softmax(-1), v_cpu)
    out.backward(grad)
    out_cpu.backward(grad.float().cpu())
    torch.npu.synchronize()

    torch.testing.assert_close(out, out_ref, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(q.grad.float().cpu(), q_cpu.grad, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(k.grad.float().cpu(), k_cpu.grad, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(v.grad.float().cpu(), v_cpu.grad, rtol=8e-3, atol=8e-3)
