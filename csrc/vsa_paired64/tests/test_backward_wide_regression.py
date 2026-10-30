"""Check wide backward tiles, sequence tails, and shared KV heads."""

import os

import pytest
import torch
import torch_npu

from block_sparse_attention import block_sparse_attention


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_block_size,kv_block_size", [(128, 128), (256, 128), (64, 64)])
@pytest.mark.parametrize("seq_len", [640, 641, 1536, 1537])
@pytest.mark.parametrize("optimized", [False, True, None])
def test_tiled_gradients_against_fp32(dtype, head_dim, q_block_size, kv_block_size, seq_len,
                                    optimized, monkeypatch):
    # Include full internal tiles, a short user block at Q=256, and
    # a one-token tail to check partial vector slabs and padded L1 reads.
    # Longer sequences cross several cyclic two-Q work items and exercise
    # UB/L1 reuse. Cover both forced paths and automatic dispatch with fewer
    # KV heads and shorter sequences than the former optimization thresholds.
    if optimized is None:
        monkeypatch.delenv("BSA_BACKWARD_OPT", raising=False)
    else:
        monkeypatch.setenv("BSA_BACKWARD_OPT", "1" if optimized else "0")
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    torch.manual_seed(20260905)
    torch.npu.manual_seed_all(20260905)
    batch, heads, kv_heads = 2, 4, 2
    options = dict(device=device, dtype=dtype, requires_grad=True)
    q = torch.randn(batch, heads, seq_len, head_dim, **options)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, **options)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, **options)
    grad = torch.randn_like(q)
    q_blocks = (seq_len + q_block_size - 1) // q_block_size
    kv_blocks = (seq_len + kv_block_size - 1) // kv_block_size
    mask = torch.rand(batch, heads, q_blocks, kv_blocks, device=device) > 0.6
    mask[..., 0] = True
    mask[..., -1] = False  # Untouched KV gradients must remain zero.
    output = block_sparse_attention(
        q, k, v, mask.to(torch.int8), (q_block_size, kv_block_size),
        input_layout="BNSD", num_key_value_heads=kv_heads,
        scale_value=head_dim**-0.5,
    )
    output.backward(grad)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    expanded_k = k_ref.repeat_interleave(heads // kv_heads, dim=1)
    expanded_v = v_ref.repeat_interleave(heads // kv_heads, dim=1)
    token_mask = mask.repeat_interleave(q_block_size, dim=2).repeat_interleave(kv_block_size, dim=3)
    token_mask = token_mask[:, :, :seq_len, :seq_len]
    scores = q_ref @ expanded_k.transpose(-1, -2) * head_dim**-0.5
    expected = scores.masked_fill(~token_mask, -torch.inf).softmax(-1) @ expanded_v
    expected.backward(grad.float())
    for actual, reference in (
        (output, expected), (q.grad, q_ref.grad),
        (k.grad, k_ref.grad), (v.grad, v_ref.grad),
    ):
        torch.testing.assert_close(actual.float(), reference, rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("q_seq,kv_seq", [(1, 129), (129, 1), (129, 65537)])
@pytest.mark.parametrize("block_size", [128, 64])
def test_automatic_rectangular_gradients_against_fp32(dtype, q_seq, kv_seq, block_size, monkeypatch):
    """Exercise automatic MQA dispatch on short inputs and KV lengths above 65536."""
    monkeypatch.delenv("BSA_BACKWARD_OPT", raising=False)
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    torch.manual_seed(20260906)
    torch.npu.manual_seed_all(20260906)
    options = dict(device=device, dtype=dtype, requires_grad=True)
    q = torch.randn(1, 2, q_seq, 128, **options)
    k = torch.randn(1, 1, kv_seq, 128, **options)
    v = torch.randn_like(k, requires_grad=True)
    grad = torch.randn_like(q)
    mask = torch.ones(1, 2, (q_seq + block_size - 1) // block_size,
                      (kv_seq + block_size - 1) // block_size,
                      dtype=torch.bool, device=device)
    mask[..., ::2, 1::2] = False
    output = block_sparse_attention(
        q, k, v, mask.to(torch.int8), (block_size, block_size), input_layout="BNSD",
        num_key_value_heads=1, scale_value=128**-0.5,
    )
    output.backward(grad)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    token_mask = mask.repeat_interleave(block_size, 2).repeat_interleave(block_size, 3)
    token_mask = token_mask[..., :q_seq, :kv_seq]
    scores = (q_ref @ k_ref.transpose(-1, -2)) * 128**-0.5
    reference = scores.masked_fill(~token_mask, -torch.inf).softmax(-1) @ v_ref
    reference.backward(grad.float())
    for name, actual, expected in (
        ("output", output, reference), ("dQ", q.grad, q_ref.grad),
        ("dK", k.grad, k_ref.grad), ("dV", v.grad, v_ref.grad),
    ):
        torch.testing.assert_close(
            actual.float(), expected, rtol=8e-3, atol=8e-3,
            msg=lambda message: f"{name}: {message}",
        )
