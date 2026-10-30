from __future__ import annotations

from collections.abc import Sequence

import torch


def _as_optional_int_list(value, name):
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers or None")
    return [int(item) for item in value]


def _normalize_arguments(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    q_input_layout,
    kv_input_layout,
    num_key_value_heads,
    actual_seq_lengths,
    actual_seq_lengths_kv,
):
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("value", value),
        ("block_sparse_mask", block_sparse_mask),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "npu":
            raise ValueError(f"{name} must be on an NPU")
    if not (query.device == key.device == value.device == block_sparse_mask.device):
        raise ValueError("query, key, value and block_sparse_mask must share one NPU")
    if q_input_layout != kv_input_layout:
        raise ValueError("q_input_layout and kv_input_layout must be identical")
    if q_input_layout not in ("TND", "BNSD"):
        raise ValueError("only TND and BNSD layouts are supported")
    if len(block_shape) != 2:
        raise ValueError("block_shape must contain exactly two integers")
    normalized_shape = (int(block_shape[0]), int(block_shape[1]))
    if normalized_shape[0] <= 0 or normalized_shape[1] <= 0:
        raise ValueError("block_shape values must be positive")
    if normalized_shape[1] % 64:
        raise ValueError("block_shape[1] must be a multiple of 64")

    q_lengths = _as_optional_int_list(actual_seq_lengths, "actual_seq_lengths")
    kv_lengths = _as_optional_int_list(
        actual_seq_lengths_kv, "actual_seq_lengths_kv"
    )
    if q_input_layout == "TND" and (q_lengths is None or kv_lengths is None):
        raise ValueError("TND layout requires both actual sequence length lists")
    if q_input_layout == "BNSD" and (q_lengths is not None or kv_lengths is not None):
        raise ValueError("BNSD layout does not accept actual sequence length lists")

    if num_key_value_heads is None:
        num_key_value_heads = int(key.shape[1])
    return (
        normalized_shape,
        q_lengths,
        kv_lengths,
        int(num_key_value_heads),
    )


class _BlockSparseAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        input_layout,
        num_key_value_heads,
        scale_value,
        inner_precise,
    ):
        output, softmax_lse, _sparse_metadata = torch.ops.miowtion128schedule2_attention_ext.forward(
            query,
            key,
            value,
            block_sparse_mask,
            list(block_shape),
            actual_seq_lengths,
            actual_seq_lengths_kv,
            input_layout,
            num_key_value_heads,
            scale_value,
            inner_precise,
            True,
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            output,
            softmax_lse,
            block_sparse_mask,
        )
        ctx.block_shape = block_shape
        ctx.actual_seq_lengths = actual_seq_lengths
        ctx.actual_seq_lengths_kv = actual_seq_lengths_kv
        ctx.input_layout = input_layout
        ctx.num_key_value_heads = num_key_value_heads
        ctx.scale_value = scale_value
        ctx.mark_non_differentiable(softmax_lse)
        return output, softmax_lse

    @staticmethod
    def backward(ctx, grad_output, _grad_lse):
        query, key, value, output, softmax_lse, block_sparse_mask = (
            ctx.saved_tensors
        )
        dq, dk, dv = torch.ops.miowtion128schedule2_attention_ext.backward(
            grad_output.contiguous(),
            query,
            key,
            value,
            output,
            softmax_lse,
            block_sparse_mask,
            None,
            list(ctx.block_shape),
            ctx.actual_seq_lengths,
            ctx.actual_seq_lengths_kv,
            ctx.input_layout,
            ctx.num_key_value_heads,
            0,
            ctx.scale_value,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def npu_miowtion128schedule2_attention(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    *,
    q_input_layout="TND",
    kv_input_layout="TND",
    num_key_value_heads=None,
    scale_value=0.0,
    inner_precise=0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    softmax_lse_flag=0,
):
    """Compatibility-style BlockSparseAttention returning ``(output, lse)``."""
    block_shape, q_lengths, kv_lengths, num_key_value_heads = _normalize_arguments(
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        q_input_layout,
        kv_input_layout,
        num_key_value_heads,
        actual_seq_lengths,
        actual_seq_lengths_kv,
    )
    if inner_precise not in (0, 1):
        raise ValueError("inner_precise must be 0 or 1")
    if query.dtype == torch.bfloat16 and inner_precise != 0:
        raise ValueError("bfloat16 requires inner_precise=0")
    if softmax_lse_flag not in (0, 1):
        raise ValueError("softmax_lse_flag must be 0 or 1")

    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if needs_grad:
        return _BlockSparseAttentionAutograd.apply(
            query,
            key,
            value,
            block_sparse_mask,
            block_shape,
            q_lengths,
            kv_lengths,
            q_input_layout,
            num_key_value_heads,
            float(scale_value),
            int(inner_precise),
        )

    output, softmax_lse, _sparse_metadata = torch.ops.miowtion128schedule2_attention_ext.forward(
        query,
        key,
        value,
        block_sparse_mask,
        list(block_shape),
        q_lengths,
        kv_lengths,
        q_input_layout,
        num_key_value_heads,
        float(scale_value),
        int(inner_precise),
        bool(softmax_lse_flag),
    )
    return output, softmax_lse


def miowtion128schedule2_attention(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape=(128, 128),
    *,
    input_layout="BNSD",
    num_key_value_heads=None,
    scale_value=0.0,
    inner_precise=0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    return_lse=False,
):
    """Run BlockSparseAttention and automatically use its native backward."""
    output, softmax_lse = npu_miowtion128schedule2_attention(
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        q_input_layout=input_layout,
        kv_input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        scale_value=scale_value,
        inner_precise=inner_precise,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        softmax_lse_flag=int(return_lse),
    )
    return (output, softmax_lse) if return_lse else output


def miowtion128schedule2_attention_backward(
    dout,
    query,
    key,
    value,
    attention_out,
    softmax_lse,
    block_sparse_mask,
    block_shape=(128, 128),
    *,
    input_layout="BNSD",
    num_key_value_heads=None,
    scale_value=0.0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
):
    """Call the migrated native gradient kernel explicitly."""
    block_shape, q_lengths, kv_lengths, num_key_value_heads = _normalize_arguments(
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        input_layout,
        input_layout,
        num_key_value_heads,
        actual_seq_lengths,
        actual_seq_lengths_kv,
    )
    return torch.ops.miowtion128schedule2_attention_ext.backward(
        dout.contiguous(),
        query,
        key,
        value,
        attention_out,
        softmax_lse,
        block_sparse_mask,
        None,
        list(block_shape),
        q_lengths,
        kv_lengths,
        input_layout,
        num_key_value_heads,
        0,
        float(scale_value),
    )


__all__ = [
    "miowtion128schedule2_attention",
    "miowtion128schedule2_attention_backward",
    "npu_miowtion128schedule2_attention",
]
