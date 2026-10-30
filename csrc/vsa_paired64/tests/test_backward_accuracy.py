import math
import os

import pytest
import torch
import torch_npu

from block_sparse_attention import block_sparse_attention


def make_block_mask(
    seq_len: int, q_block_size: int, kv_block_size: int, sparsity: float, seed: int
):
    q_blocks = math.ceil(seq_len / q_block_size)
    kv_blocks = math.ceil(seq_len / kv_block_size)
    total = q_blocks * kv_blocks
    keep_count = round(total * (1.0 - sparsity))
    if keep_count < q_blocks:
        raise ValueError("sparsity must leave at least one KV block per Q block")

    keep = torch.zeros(total, dtype=torch.bool)
    rows = torch.arange(q_blocks, dtype=torch.int64)
    keep[rows * kv_blocks + rows.remainder(kv_blocks)] = True
    extra = keep_count - q_blocks
    if extra:
        candidates = torch.arange(total, dtype=torch.int64)[~keep]
        generator = torch.Generator().manual_seed(seed)
        selected = candidates[
            torch.randperm(candidates.numel(), generator=generator)[:extra]
        ]
        keep[selected] = True
    return keep.view(q_blocks, kv_blocks)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    threshold = 8e-3 + 8e-3 * expected.abs()
    return {
        "mean_abs": error.abs().mean().item(),
        "max_abs": error.abs().max().item(),
        "relative_l2": (error.norm() / expected.norm().clamp_min(1e-12)).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
        "close_ratio": (error.abs() <= threshold).float().mean().item(),
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_block_size", [1, 17, 126, 127, 128, 129])
@pytest.mark.parametrize("kv_block_size", [64, 128, 192, 256])
def test_backward_against_dense_fp32(dtype, head_dim, q_block_size, kv_block_size):
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    torch.manual_seed(1234)
    torch.npu.manual_seed_all(1234)

    batch, heads, seq_len = 1, 2, 257
    scale = head_dim**-0.5
    shape = (batch, heads, seq_len, head_dim)

    block_mask_cpu = make_block_mask(
        seq_len, q_block_size, kv_block_size, 0.50, 1234
    )
    q_blocks, kv_blocks = block_mask_cpu.shape
    block_mask = (
        block_mask_cpu.to(torch.int8)
        .view(1, 1, q_blocks, kv_blocks)
        .expand(batch, heads, -1, -1)
        .contiguous()
        .to(device)
    )
    token_mask = (
        block_mask_cpu.repeat_interleave(q_block_size, 0)
        .repeat_interleave(kv_block_size, 1)
        [:seq_len, :seq_len]
        .view(1, 1, seq_len, seq_len)
        .to(device)
    )

    q = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(shape, dtype=dtype, device=device)

    output = block_sparse_attention(
        q,
        k,
        v,
        block_mask,
        block_shape=(q_block_size, kv_block_size),
        input_layout="BNSD",
        num_key_value_heads=heads,
        scale_value=scale,
        inner_precise=0,
    )
    output.backward(grad_output)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale
    probabilities = scores.masked_fill(~token_mask, float("-inf")).softmax(-1)
    reference = torch.matmul(probabilities, v_ref)
    reference.backward(grad_output.float())
    torch.npu.synchronize()

    name = "bf16" if dtype == torch.bfloat16 else "fp16"
    failures = []
    for gradient_name, actual, expected in (
        ("output", output, reference),
        ("dQ", q.grad, q_ref.grad),
        ("dK", k.grad, k_ref.grad),
        ("dV", v.grad, v_ref.grad),
    ):
        metrics = error_metrics(actual, expected)
        print(f"{name} {gradient_name}: {metrics}")
        if metrics["close_ratio"] < 1.0:
            for row_begin in range(0, seq_len, q_block_size):
                row_end = min(seq_len, row_begin + q_block_size)
                row_metrics = error_metrics(
                    actual[:, :, row_begin:row_end],
                    expected[:, :, row_begin:row_end],
                )
                print(
                    f"{name} {gradient_name} rows[{row_begin}:{row_end}]: "
                    f"{row_metrics}"
                )
                if row_metrics["close_ratio"] < 0.5 and row_end - row_begin <= 17:
                    print(
                        f"  actual sample={actual[0, 0, row_begin, :8].float().cpu().tolist()}\n"
                        f"  expect sample={expected[0, 0, row_begin, :8].float().cpu().tolist()}"
                    )
        try:
            torch.testing.assert_close(
                actual.float(), expected, rtol=8e-3, atol=8e-3
            )
        except AssertionError as error:
            failures.append((gradient_name, error))
    if failures:
        gradient_name, error = failures[0]
        raise AssertionError(f"{gradient_name} failed:\n{error}") from error


def test_grouped_multisegment_backward_matches_fa():
    """Guard FA rounding parity and L1 cache lifetime across KV segments."""
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    torch.manual_seed(20260904)
    torch.npu.manual_seed_all(20260904)

    batch, heads, seq_len, head_dim = 1, 24, 1024, 128
    shape = (batch, heads, seq_len, head_dim)
    scale = head_dim**-0.5
    options = dict(dtype=torch.bfloat16, device=device)
    q_data = torch.randn(shape, **options)
    k_data = torch.randn(shape, **options)
    v_data = torch.randn(shape, **options)
    grad_output = torch.randn(shape, **options)
    block_count = math.ceil(seq_len / 128)
    block_mask = torch.ones(
        (batch, heads, block_count, block_count),
        dtype=torch.int8,
        device=device,
    )

    q_bsa = q_data.detach().requires_grad_(True)
    k_bsa = k_data.detach().requires_grad_(True)
    v_bsa = v_data.detach().requires_grad_(True)
    output_bsa = block_sparse_attention(
        q_bsa,
        k_bsa,
        v_bsa,
        block_mask,
        block_shape=(128, 128),
        input_layout="BNSD",
        num_key_value_heads=heads,
        scale_value=scale,
        inner_precise=0,
    )
    output_bsa.backward(grad_output)

    q_fa = q_data.detach().requires_grad_(True)
    k_fa = k_data.detach().requires_grad_(True)
    v_fa = v_data.detach().requires_grad_(True)
    output_fa = torch_npu.npu_fusion_attention(
        q_fa,
        k_fa,
        v_fa,
        head_num=heads,
        input_layout="BNSD",
        scale=scale,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=0,
    )[0]
    output_fa.backward(grad_output)
    torch.npu.synchronize()

    for actual, expected in (
        (output_bsa, output_fa),
        (q_bsa.grad, q_fa.grad),
        (k_bsa.grad, k_fa.grad),
        (v_bsa.grad, v_fa.grad),
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=8e-3, atol=8e-3
        )


def test_wide_grad_macro_boundaries_match_fa():
    """Exercise several 64-edge gradient macros in one wide work item."""
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    torch.manual_seed(20260905)
    torch.npu.manual_seed_all(20260905)

    batch, heads, seq_len, head_dim = 1, 2, 2048, 128
    shape = (batch, heads, seq_len, head_dim)
    scale = head_dim**-0.5
    options = dict(dtype=torch.bfloat16, device=device)
    q_data = torch.randn(shape, **options)
    k_data = torch.randn(shape, **options)
    v_data = torch.randn(shape, **options)
    grad_output = torch.randn(shape, **options)
    block_count = math.ceil(seq_len / 128)
    block_mask = torch.ones(
        (batch, heads, block_count, block_count),
        dtype=torch.int8,
        device=device,
    )

    q_bsa = q_data.detach().requires_grad_(True)
    k_bsa = k_data.detach().requires_grad_(True)
    v_bsa = v_data.detach().requires_grad_(True)
    output_bsa = block_sparse_attention(
        q_bsa,
        k_bsa,
        v_bsa,
        block_mask,
        block_shape=(128, 128),
        input_layout="BNSD",
        num_key_value_heads=heads,
        scale_value=scale,
        inner_precise=0,
    )
    output_bsa.backward(grad_output)

    q_fa = q_data.detach().requires_grad_(True)
    k_fa = k_data.detach().requires_grad_(True)
    v_fa = v_data.detach().requires_grad_(True)
    output_fa = torch_npu.npu_fusion_attention(
        q_fa,
        k_fa,
        v_fa,
        head_num=heads,
        input_layout="BNSD",
        scale=scale,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=0,
    )[0]
    output_fa.backward(grad_output)
    torch.npu.synchronize()

    for actual, expected in (
        (output_bsa, output_fa),
        (q_bsa.grad, q_fa.grad),
        (k_bsa.grad, k_fa.grad),
        (v_bsa.grad, v_fa.grad),
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=8e-3, atol=8e-3
        )


def test_active_edge_packet_backward_against_dense_fp32():
    """Exercise a packet with more than four unique KV blocks."""
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    torch.manual_seed(20260904)
    torch.npu.manual_seed_all(20260904)

    batch, heads, seq_len, head_dim = 1, 2, 1024, 128
    shape = (batch, heads, seq_len, head_dim)
    options = dict(dtype=torch.bfloat16, device=device)
    q = torch.randn(shape, **options, requires_grad=True)
    k = torch.randn(shape, **options, requires_grad=True)
    v = torch.randn(shape, **options, requires_grad=True)
    grad_output = torch.randn(shape, **options)

    # Every group of four Q blocks has 16 active edges spread over eight KV
    # blocks.  The active-edge packet therefore takes the wide (>4 unique KV)
    # dQ path rather than the dense four-slot CachedB path.
    rows = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (0, 1, 4, 5),
        (2, 3, 6, 7),
        (0, 3, 4, 7),
        (1, 2, 5, 6),
    )
    block_mask_cpu = torch.zeros((8, 8), dtype=torch.bool)
    for q_block, kv_blocks in enumerate(rows):
        block_mask_cpu[q_block, list(kv_blocks)] = True
    block_mask = (
        block_mask_cpu.to(torch.int8)
        .view(1, 1, 8, 8)
        .expand(batch, heads, -1, -1)
        .contiguous()
        .to(device)
    )
    token_mask = (
        block_mask_cpu.repeat_interleave(128, 0)
        .repeat_interleave(128, 1)
        .view(1, 1, seq_len, seq_len)
        .to(device)
    )

    output = block_sparse_attention(
        q,
        k,
        v,
        block_mask,
        block_shape=(128, 128),
        input_layout="BNSD",
        num_key_value_heads=heads,
        scale_value=head_dim**-0.5,
        inner_precise=0,
    )
    output.backward(grad_output)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * head_dim**-0.5
    probabilities = scores.masked_fill(~token_mask, float("-inf")).softmax(-1)
    reference = torch.matmul(probabilities, v_ref)
    reference.backward(grad_output.float())
    torch.npu.synchronize()

    for actual, expected in (
        (output, reference),
        (q.grad, q_ref.grad),
        (k.grad, k_ref.grad),
        (v.grad, v_ref.grad),
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=8e-3, atol=8e-3
        )


def test_wide_packet_split_column_across_work_items_against_dense_fp32():
    """Cover a KV column split across packets and a logical-work boundary."""
    device = torch.device(f"npu:{os.environ.get('BSA_TEST_DEVICE', '7')}")
    torch.npu.set_device(device)
    torch.manual_seed(20260904)
    torch.npu.manual_seed_all(20260904)

    # Forty-nine Q tasks cross the 48-task logical-work boundary. Keeping KV
    # column zero for every Q also forces that column to resume across six
    # physical cap-8 packets in the first work item, without constructing a
    # block-size-specific kernel path.
    batch, heads, seq_len, head_dim = 1, 1, 49 * 128, 128
    shape = (batch, heads, seq_len, head_dim)
    options = dict(dtype=torch.bfloat16, device=device)
    q = torch.randn(shape, **options, requires_grad=True)
    k = torch.randn(shape, **options, requires_grad=True)
    v = torch.randn(shape, **options, requires_grad=True)
    grad_output = torch.randn(shape, **options)

    block_count = seq_len // 128
    block_mask_cpu = torch.eye(block_count, dtype=torch.bool)
    block_mask_cpu[:, 0] = True
    block_mask = block_mask_cpu.to(torch.int8).view(
        batch, heads, block_count, block_count
    ).to(device)
    token_mask = (
        block_mask_cpu.repeat_interleave(128, 0)
        .repeat_interleave(128, 1)
        .view(batch, heads, seq_len, seq_len)
        .to(device)
    )

    output = block_sparse_attention(
        q,
        k,
        v,
        block_mask,
        block_shape=(128, 128),
        input_layout="BNSD",
        num_key_value_heads=heads,
        scale_value=head_dim**-0.5,
        inner_precise=0,
    )
    output.backward(grad_output)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * head_dim**-0.5
    probabilities = scores.masked_fill(~token_mask, float("-inf")).softmax(-1)
    reference = torch.matmul(probabilities, v_ref)
    reference.backward(grad_output.float())
    torch.npu.synchronize()

    for actual, expected in (
        (output, reference),
        (q.grad, q_ref.grad),
        (k.grad, k_ref.grad),
        (v.grad, v_ref.grad),
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=8e-3, atol=8e-3
        )
