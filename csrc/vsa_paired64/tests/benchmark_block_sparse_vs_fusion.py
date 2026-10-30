#!/usr/bin/env python3
"""Benchmark migrated block-sparse attention against fusion attention.

The default input matches ``run_v6a_q128_with_bsa.py``: B=1, H=24, D=128,
53223 natural multimodal tokens padded to S=53248, and a V6A routing mask
constructed from preserved condition/keyframe blocks plus video Top-K blocks.
``npu_fusion_attention`` uses the same padded Q/K/V tensors and no mask.

Use ``--msprof`` to launch a profiling worker, parse op_summary_*.csv, and
report forward and backward device-side Task Duration statistics and the
resulting speedups. Both operators always use ``inner_precise=0``.
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch


V6A_BATCH = 1
V6A_HEADS = 24
V6A_HEAD_DIM = 128
V6A_TEXT_TOKENS = 300
V6A_AUDIO_TOKENS = 123
V6A_VIDEO_GRID = (60, 22, 40)
V6A_VIDEO_TOKENS = math.prod(V6A_VIDEO_GRID)
V6A_NATURAL_LENGTH = V6A_TEXT_TOKENS + V6A_AUDIO_TOKENS + V6A_VIDEO_TOKENS
V6A_PADDED_LENGTH = 53248
V6A_VIDEO_FRAME_TOKENS = V6A_VIDEO_GRID[1] * V6A_VIDEO_GRID[2]
V6A_KEYFRAME_START_FRAMES = 1
V6A_KEYFRAME_END_FRAMES = 1
V6A_BLOCK_Q = 128
V6A_BLOCK_K = 128
V6A_TOPK_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-mode",
        choices=("v6a", "random"),
        default="v6a",
        help="input construction: aligned V6A routing (default) or legacy random sparsity",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=V6A_PADDED_LENGTH,
        help="padded sequence length (V6A requires 53248)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=V6A_HEADS)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.86,
        help="requested sparsity for --input-mode=random; ignored by V6A routing",
    )
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=128,
        help="BSA KV sparse-block size (V6A default: 128)",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="number of measured operator calls (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="benchmark forward only (required when the BSA backward kernel does not support the block shape)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="deprecated: sparse and dense-full outputs are intentionally different",
    )
    parser.add_argument(
        "--msprof",
        action="store_true",
        help="launch this benchmark under msprof and report device task durations",
    )
    parser.add_argument(
        "--msprof-output",
        type=Path,
        default=Path("msprof_attention_results"),
        help="parent directory for msprof output",
    )
    parser.add_argument("--msprof-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def make_shared_block_keep_mask(
    q_blocks: int,
    kv_blocks: int,
    sparsity: float,
    seed: int,
) -> torch.Tensor:
    """Create an exact-density CPU bool mask, with at least one KV per Q row."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    total = q_blocks * kv_blocks
    keep_count = round(total * (1.0 - sparsity))
    if keep_count < q_blocks:
        raise ValueError(
            f"{sparsity=:.6f} keeps only {keep_count} blocks, fewer than the "
            f"{q_blocks} required to keep every Q-block row valid"
        )

    # Keeping one diagonal/wrapped block per row prevents fully masked softmax rows.
    flat_keep = torch.zeros(total, dtype=torch.bool)
    rows = torch.arange(q_blocks, dtype=torch.int64)
    mandatory = rows * kv_blocks + rows.remainder(kv_blocks)
    flat_keep[mandatory] = True

    extra_count = keep_count - q_blocks
    if extra_count:
        candidates = torch.arange(total, dtype=torch.int64)[~flat_keep]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:extra_count]]
        flat_keep[chosen] = True

    return flat_keep.view(q_blocks, kv_blocks)


def quantize_v6a_topk(real_topk: int) -> int:
    """Use the same TOPK buckets as the V6A test helper."""
    buckets = sorted(set(range(1, 65)) | {value * 4 for value in range(17, 65)})
    return next((bucket for bucket in buckets if bucket >= real_topk), real_topk)


def make_v6a_qkv_and_mask(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Reproduce the multimodal input and routing used by the V6A Q128 test."""
    natural_shape = (V6A_BATCH, V6A_HEADS, V6A_NATURAL_LENGTH, V6A_HEAD_DIM)
    options = dict(dtype=dtype, device=device)

    with torch.no_grad():
        query = torch.randn(natural_shape, **options)
        key = torch.randn(natural_shape, **options)
        value = torch.randn(natural_shape, **options)
        key_centered = key - key.mean(dim=2, keepdim=True)

        padding = V6A_PADDED_LENGTH - V6A_NATURAL_LENGTH
        query = torch.nn.functional.pad(query, (0, 0, 0, padding)).contiguous()
        key = torch.nn.functional.pad(key, (0, 0, 0, padding)).contiguous()
        value = torch.nn.functional.pad(value, (0, 0, 0, padding)).contiguous()
        key_centered = torch.nn.functional.pad(
            key_centered, (0, 0, 0, padding)
        ).contiguous()

        q_blocks = V6A_PADDED_LENGTH // V6A_BLOCK_Q
        kv_blocks = V6A_PADDED_LENGTH // V6A_BLOCK_K
        video_start = (V6A_TEXT_TOKENS + V6A_BLOCK_K - 1) // V6A_BLOCK_K
        video_end = (V6A_TEXT_TOKENS + V6A_VIDEO_TOKENS) // V6A_BLOCK_K

        first_keyframe_token_end = (
            V6A_TEXT_TOKENS
            + V6A_KEYFRAME_START_FRAMES * V6A_VIDEO_FRAME_TOKENS
        )
        last_keyframe_token_start = (
            V6A_TEXT_TOKENS
            + V6A_VIDEO_TOKENS
            - V6A_KEYFRAME_END_FRAMES * V6A_VIDEO_FRAME_TOKENS
        )
        first_keyframe_end = min(
            video_end,
            (first_keyframe_token_end + V6A_BLOCK_K - 1) // V6A_BLOCK_K,
        )
        last_keyframe_start = max(
            first_keyframe_end,
            last_keyframe_token_start // V6A_BLOCK_K,
        )

        candidate_start = first_keyframe_end
        candidate_end = last_keyframe_start
        candidate_blocks = candidate_end - candidate_start
        candidate_topk = max(
            1,
            min(candidate_blocks, int(V6A_TOPK_RATIO * candidate_blocks)),
        )

        q_pooled = query.view(
            V6A_BATCH, V6A_HEADS, q_blocks, V6A_BLOCK_Q, V6A_HEAD_DIM
        ).float().mean(dim=3)
        k_pooled = key_centered.view(
            V6A_BATCH, V6A_HEADS, kv_blocks, V6A_BLOCK_K, V6A_HEAD_DIM
        ).float().mean(dim=3)
        scores = q_pooled @ k_pooled.transpose(-1, -2)
        candidate_indices = torch.topk(
            scores[..., candidate_start:candidate_end],
            candidate_topk,
            dim=-1,
            sorted=False,
        ).indices + candidate_start

        forced = torch.cat(
            (
                torch.arange(0, video_start, device=device),
                torch.arange(video_start, first_keyframe_end, device=device),
                torch.arange(last_keyframe_start, video_end, device=device),
                torch.arange(video_end, kv_blocks, device=device),
            )
        )
        forced = forced.view(1, 1, 1, -1).expand(
            V6A_BATCH, V6A_HEADS, q_blocks, -1
        )
        lut = torch.cat((forced, candidate_indices), dim=-1).to(torch.int32)

        real_topk = lut.shape[-1]
        kernel_topk = quantize_v6a_topk(real_topk)
        if kernel_topk > real_topk:
            lut = torch.cat(
                (
                    lut,
                    lut[..., -1:].expand(*lut.shape[:-1], kernel_topk - real_topk),
                ),
                dim=-1,
            )
        lut = lut.contiguous()

        block_mask = torch.zeros(
            (V6A_BATCH, V6A_HEADS, q_blocks, kv_blocks),
            dtype=torch.int8,
            device=device,
        )
        block_mask.scatter_(-1, lut.long(), 1)
        block_mask = block_mask.contiguous()

    selected = int(block_mask.sum().item())
    total = block_mask.numel()
    info: dict[str, object] = {
        "lut_shape": tuple(lut.shape),
        "real_topk": real_topk,
        "kernel_topk": kernel_topk,
        "selected": selected,
        "total": total,
        "text_blocks": video_start,
        "keyframe_start_blocks": first_keyframe_end - video_start,
        "keyframe_end_blocks": video_end - last_keyframe_start,
        "audio_blocks": kv_blocks - video_end,
        "candidate_blocks": candidate_blocks,
        "candidate_topk": candidate_topk,
    }
    return query, key, value, block_mask, info


def effective_token_density(
    block_keep: torch.Tensor,
    seq_len: int,
    q_block_size: int,
    kv_block_size: int,
) -> float:
    q_sizes = torch.full((block_keep.shape[0],), q_block_size, dtype=torch.int64)
    kv_sizes = torch.full((block_keep.shape[1],), kv_block_size, dtype=torch.int64)
    q_sizes[-1] = seq_len - (block_keep.shape[0] - 1) * q_block_size
    kv_sizes[-1] = seq_len - (block_keep.shape[1] - 1) * kv_block_size
    kept_tokens = (
        block_keep.to(torch.int64) * q_sizes[:, None] * kv_sizes[None, :]
    ).sum()
    return kept_tokens.item() / (seq_len * seq_len)


def benchmark(
    op: Callable[[], torch.Tensor], warmup: int, iterations: int
) -> tuple[list[float], float]:
    for _ in range(warmup):
        output = op()
    torch.npu.synchronize()
    del output

    latencies_ms: list[float] = []
    last_output = None
    for _ in range(iterations):
        torch.npu.synchronize()
        start = time.perf_counter()
        last_output = op()
        torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    # Materialize one value after timing so the returned tensor remains observable.
    checksum = float(last_output.reshape(-1)[0].float().cpu())
    del last_output
    return latencies_ms, checksum


def clear_gradients(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def benchmark_backward(
    op: Callable[[], torch.Tensor],
    grad_output: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    warmup: int,
    iterations: int,
) -> tuple[list[float], float]:
    """Time backward only; each forward graph is built before timing starts."""
    for _ in range(warmup):
        clear_gradients(*inputs)
        output = op()
        output.backward(grad_output)
    torch.npu.synchronize()
    del output

    latencies_ms: list[float] = []
    for _ in range(iterations):
        clear_gradients(*inputs)
        output = op()
        torch.npu.synchronize()
        start = time.perf_counter()
        output.backward(grad_output)
        torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        del output

    checksum = float(inputs[0].grad.reshape(-1)[0].float().cpu())
    return latencies_ms, checksum


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(name: str, values: list[float]) -> None:
    print(
        f"{name:28s} "
        f"mean={statistics.fmean(values):9.3f} ms  "
        f"p50={statistics.median(values):9.3f} ms  "
        f"p90={percentile(values, 0.90):9.3f} ms"
    )


def worker_arguments(args: argparse.Namespace) -> list[str]:
    result = [
        "--msprof-worker",
        "--input-mode",
        args.input_mode,
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--num-heads",
        str(args.num_heads),
        "--head-dim",
        str(args.head_dim),
        "--sparsity",
        str(args.sparsity),
        "--q-block-size",
        str(args.q_block_size),
        "--kv-block-size",
        str(args.kv_block_size),
        "--dtype",
        args.dtype,
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
    ]
    if args.forward_only:
        result.append("--forward-only")
    return result


def find_op_summary(profile_dir: Path) -> Path:
    candidates = sorted(
        profile_dir.rglob("op_summary_*.csv"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise FileNotFoundError(f"no op_summary_*.csv found below {profile_dir}")
    return candidates[-1]


def get_csv_field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row:
            return row[name]
    lowered = {key.strip().lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"none of columns {names} are present; got {list(row)}")


def parse_msprof_durations(
    csv_path: Path, warmup: int, iterations: int, forward_only: bool = False
) -> tuple[
    list[float], list[float], list[float], list[float], list[float], list[float]
]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    records: list[tuple[float, str, float]] = []
    type_counts: dict[str, int] = {}
    for position, row in enumerate(rows):
        op_type = get_csv_field(row, "OP Type", "Op Type").strip()
        type_counts[op_type] = type_counts.get(op_type, 0) + 1
        normalized = "".join(character for character in op_type.lower() if character.isalnum())
        if "blocksparseattentiongradfinalize" in normalized:
            kind = "bsa_backward_finalize"
        elif "blocksparseattentiongrad" in normalized:
            kind = "bsa_backward_main"
        elif "blocksparseattention" in normalized:
            kind = "bsa_forward"
        elif "flashattentionscoregrad" in normalized or "fusionattentiongrad" in normalized:
            kind = "fa_backward"
        elif "flashattentionscore" in normalized or "fusionattention" in normalized:
            kind = "fa_forward"
        else:
            continue

        duration_us = float(get_csv_field(row, "Task Duration(us)").strip())
        try:
            start_us = float(get_csv_field(row, "Task Start Time(us)").strip())
        except (KeyError, ValueError):
            start_us = float(position)
        records.append((start_us, kind, duration_us / 1000.0))

    records.sort(key=lambda item: item[0])
    durations = {
        kind: [duration for _, record_kind, duration in records if record_kind == kind]
        for kind in (
            "bsa_forward",
            "bsa_backward_main",
            "bsa_backward_finalize",
            "fa_forward",
            "fa_backward",
        )
    }
    expected = warmup + iterations
    # A forward operator is also executed to build every backward graph. The
    # worker profiles backward groups first and pure-forward groups last, so
    # the last `iterations` forward records are the requested forward samples.
    minimum_counts = (
        {
            "bsa_forward": expected,
            "fa_forward": expected,
        }
        if forward_only
        else {
            "bsa_forward": 2 * expected,
            "fa_forward": 2 * expected,
            "bsa_backward_main": expected,
            "fa_backward": expected,
        }
    )
    if any(len(durations[kind]) < count for kind, count in minimum_counts.items()):
        common_types = sorted(type_counts.items(), key=lambda item: item[1], reverse=True)[:20]
        raise RuntimeError(
            "insufficient target operators in op_summary: "
            f"counts={ {kind: len(values) for kind, values in durations.items()} }, "
            f"minimum={minimum_counts}; "
            f"most common OP Types={common_types}"
        )
    if any(len(durations[kind]) != count for kind, count in minimum_counts.items()):
        print(
            "Warning: target operator counts differ from warmup+iters; "
            f"using the last {iterations} records for each kind; "
            f"counts={ {kind: len(values) for kind, values in durations.items()} }."
        )
    bsa_main = [] if forward_only else durations["bsa_backward_main"][-iterations:]
    bsa_finalize = [] if forward_only else durations["bsa_backward_finalize"][-iterations:]
    bsa_total = (
        [main + finalize for main, finalize in zip(bsa_main, bsa_finalize)]
        if bsa_finalize
        else bsa_main.copy()
    )
    return (
        durations["bsa_forward"][-iterations:],
        bsa_total,
        bsa_main,
        bsa_finalize,
        durations["fa_forward"][-iterations:],
        [] if forward_only else durations["fa_backward"][-iterations:],
    )


def run_with_msprof(args: argparse.Namespace) -> None:
    msprof = shutil.which("msprof")
    if msprof is None:
        raise RuntimeError("msprof was not found in PATH; source the CANN set_env.sh first")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.msprof_output.expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    command = [
        msprof,
        f"--output={output_dir}",
        "--aic-mode=task-based",
        "--task-time=l1",
        "--aic-metrics=PipeUtilization",
        "--summary-format=csv",
        "--export=on",
        sys.executable,
        str(script),
        *worker_arguments(args),
    ]
    print("Running msprof:")
    print("  " + shlex.join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"msprof failed with exit code {completed.returncode}")

    csv_path = find_op_summary(output_dir)
    (
        bsa_fwd_ms,
        bsa_bwd_ms,
        bsa_bwd_main_ms,
        bsa_bwd_finalize_ms,
        fa_fwd_ms,
        fa_bwd_ms,
    ) = parse_msprof_durations(
        csv_path, args.warmup, args.iters, forward_only=args.forward_only
    )
    print("msprof device task results")
    summarize("BSA forward", bsa_fwd_ms)
    summarize("FA forward", fa_fwd_ms)
    if not args.forward_only:
        summarize("BSA backward", bsa_bwd_ms)
        summarize("  BSA backward main", bsa_bwd_main_ms)
        if bsa_bwd_finalize_ms:
            summarize("  BSA backward finalize", bsa_bwd_finalize_ms)
        summarize("FA backward", fa_bwd_ms)
    print(
        "Forward speedup (FA / BSA): "
        f"{statistics.fmean(fa_fwd_ms) / statistics.fmean(bsa_fwd_ms):.3f}x"
    )
    if not args.forward_only:
        print(
            "Backward speedup (FA / BSA): "
            f"{statistics.fmean(fa_bwd_ms) / statistics.fmean(bsa_bwd_ms):.3f}x"
        )
    print(f"op_summary: {csv_path}")


def main() -> None:
    args = parse_args()
    if args.msprof and not args.msprof_worker:
        run_with_msprof(args)
        return

    # Prefer the repository's in-place build over an installed wheel,
    # independently of the working directory (including msprof workers).
    local_package_root = Path(__file__).resolve().parents[1]
    if (local_package_root / "block_sparse_attention" / "__init__.py").is_file():
        sys.path.insert(0, str(local_package_root))

    import torch_npu
    import block_sparse_attention as bsa_package

    print(f"BSA package: {Path(bsa_package.__file__).resolve()}")
    print(f"BSA native library: {bsa_package._LIBRARY.resolve()}")

    if args.seq_len <= 0 or args.batch_size <= 0 or args.num_heads <= 0:
        raise ValueError("seq-len, batch-size, and num-heads must be positive")
    if not args.forward_only and args.head_dim not in (64, 128):
        raise ValueError("head-dim must be 64 or 128 for the backward benchmark")
    if args.q_block_size <= 0:
        raise ValueError("q-block-size must be positive")
    if args.kv_block_size <= 0 or args.kv_block_size % 64:
        raise ValueError("kv-block-size must be a positive multiple of 64")
    if args.warmup <= 0 or args.iters <= 0:
        raise ValueError("warmup and iters must be positive")
    if args.input_mode == "v6a":
        required = {
            "seq_len": V6A_PADDED_LENGTH,
            "batch_size": V6A_BATCH,
            "num_heads": V6A_HEADS,
            "head_dim": V6A_HEAD_DIM,
            "q_block_size": V6A_BLOCK_Q,
            "kv_block_size": V6A_BLOCK_K,
            "dtype": "bf16",
        }
        mismatches = {
            name: (getattr(args, name), expected)
            for name, expected in required.items()
            if getattr(args, name) != expected
        }
        if mismatches:
            raise ValueError(
                f"V6A input requires {required}; mismatches={mismatches}. "
                "Use --input-mode=random for other configurations."
            )

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    inner_precise = 0
    scale = args.head_dim**-0.5
    q_blocks = math.ceil(args.seq_len / args.q_block_size)
    kv_blocks = math.ceil(args.seq_len / args.kv_block_size)

    v6a_info = None
    if args.input_mode == "v6a":
        query, key, value, block_sparse_mask, v6a_info = make_v6a_qkv_and_mask(
            device, dtype
        )
        block_density = int(v6a_info["selected"]) / int(v6a_info["total"])
        token_density = block_density
    else:
        shape = (args.batch_size, args.num_heads, args.seq_len, args.head_dim)
        query = torch.randn(shape, dtype=dtype, device=device)
        key = torch.randn(shape, dtype=dtype, device=device)
        value = torch.randn(shape, dtype=dtype, device=device)
        block_keep_cpu = make_shared_block_keep_mask(
            q_blocks, kv_blocks, args.sparsity, args.seed
        )
        block_density = block_keep_cpu.float().mean().item()
        token_density = effective_token_density(
            block_keep_cpu, args.seq_len, args.q_block_size, args.kv_block_size
        )
        # BSA: 1 means that a block participates in attention. The legacy
        # random pattern is shared by all heads; FA does not consume it.
        block_sparse_mask = (
            block_keep_cpu.to(torch.int8)
            .view(1, 1, q_blocks, kv_blocks)
            .expand(args.batch_size, args.num_heads, -1, -1)
            .contiguous()
            .to(device)
        )

    requires_grad = not args.forward_only
    query.requires_grad_(requires_grad)
    key.requires_grad_(requires_grad)
    value.requires_grad_(requires_grad)
    grad_output = None if args.forward_only else torch.randn_like(query)
    inputs = (query, key, value)
    shape = tuple(query.shape)

    elem_bytes = torch.empty((), dtype=dtype).element_size()
    qkv_bytes = 3 * math.prod(shape) * elem_bytes
    output_bytes = math.prod(shape) * elem_bytes
    block_mask_bytes = args.batch_size * args.num_heads * q_blocks * kv_blocks

    print("Configuration")
    print(
        f"  B={args.batch_size}, H={args.num_heads}, S={args.seq_len}, "
        f"D={args.head_dim}, dtype={args.dtype}, inner_precise={inner_precise}, "
        f"device={device}, input_mode={args.input_mode}"
    )
    if v6a_info is not None:
        print(
            f"  V6A natural/padded tokens={V6A_NATURAL_LENGTH}/{V6A_PADDED_LENGTH}, "
            f"video_grid={V6A_VIDEO_GRID}, LUT={v6a_info['lut_shape']}"
        )
        print(
            "  forced blocks: "
            f"text={v6a_info['text_blocks']}, "
            f"keyframe={v6a_info['keyframe_start_blocks']}+"
            f"{v6a_info['keyframe_end_blocks']}, "
            f"audio={v6a_info['audio_blocks']}; "
            f"candidate={v6a_info['candidate_topk']}/"
            f"{v6a_info['candidate_blocks']}; "
            f"real/kernel topk={v6a_info['real_topk']}/"
            f"{v6a_info['kernel_topk']}"
        )
    else:
        print(f"  requested random block sparsity={args.sparsity:.4%}")
    print(
        f"  BSA block={args.q_block_size}x{args.kv_block_size}, "
        f"blocks={q_blocks}x{kv_blocks}, FA mask=None (full dense)"
    )
    print(
        f"  actual block sparsity={1.0 - block_density:.4%}, "
        f"actual token sparsity={1.0 - token_density:.4%}"
    )
    print("Estimated persistent tensor memory (operator workspaces excluded)")
    print(
        f"  Q/K/V={gib(qkv_bytes):.3f} GiB, output={gib(output_bytes):.3f} GiB, "
        "FA mask=0 GiB, "
        f"BSA block mask={gib(block_mask_bytes):.6f} GiB"
    )
    if not args.forward_only:
        print(
            f"  Q/K/V gradients={gib(qkv_bytes):.3f} GiB, "
            f"output gradient={gib(output_bytes):.3f} GiB"
        )

    torch.npu.synchronize()

    def run_bsa() -> torch.Tensor:
        output, _softmax_lse = bsa_package.npu_block_sparse_attention(
            query,
            key,
            value,
            block_sparse_mask,
            (args.q_block_size, args.kv_block_size),
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=args.num_heads,
            scale_value=scale,
            inner_precise=inner_precise,
            softmax_lse_flag=1,
        )
        return output

    def run_fa() -> torch.Tensor:
        return torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=args.num_heads,
            input_layout="BNSD",
            scale=scale,
            keep_prob=1.0,
            inner_precise=inner_precise,
            sparse_mode=0,
        )[0]

    if args.msprof_worker:
        # Profile backward groups first. Each call also emits one forward op;
        # pure-forward groups are placed last for unambiguous CSV selection.
        if not args.forward_only:
            for _ in range(args.warmup):
                clear_gradients(*inputs)
                bsa_output = run_bsa()
                bsa_output.backward(grad_output)
            torch.npu.synchronize()
            for _ in range(args.iters):
                clear_gradients(*inputs)
                bsa_output = run_bsa()
                bsa_output.backward(grad_output)
            torch.npu.synchronize()

            for _ in range(args.warmup):
                clear_gradients(*inputs)
                fa_output = run_fa()
                fa_output.backward(grad_output)
            torch.npu.synchronize()
            for _ in range(args.iters):
                clear_gradients(*inputs)
                fa_output = run_fa()
                fa_output.backward(grad_output)
            torch.npu.synchronize()

        with torch.inference_mode():
            for _ in range(args.warmup):
                bsa_output = run_bsa()
            torch.npu.synchronize()
            for _ in range(args.iters):
                bsa_output = run_bsa()
            torch.npu.synchronize()
            for _ in range(args.warmup):
                fa_output = run_fa()
            torch.npu.synchronize()
            for _ in range(args.iters):
                fa_output = run_fa()
            torch.npu.synchronize()
        print(
            "MSPROF_WORKER_DONE "
            f"BSA_sample={float(bsa_output.reshape(-1)[0].float().cpu()):.6g} "
            f"FA_sample={float(fa_output.reshape(-1)[0].float().cpu()):.6g}"
        )
        return

    with torch.inference_mode():
        bsa_ms, bsa_checksum = benchmark(run_bsa, args.warmup, args.iters)
        fa_ms, fa_checksum = benchmark(run_fa, args.warmup, args.iters)

    if not args.forward_only:
        bsa_bwd_ms, bsa_grad_checksum = benchmark_backward(
            run_bsa, grad_output, inputs, args.warmup, args.iters
        )
        fa_bwd_ms, fa_grad_checksum = benchmark_backward(
            run_fa, grad_output, inputs, args.warmup, args.iters
        )

    print("Results")
    summarize("BSA forward", bsa_ms)
    summarize("FA forward", fa_ms)
    if not args.forward_only:
        summarize("BSA backward", bsa_bwd_ms)
        summarize("FA backward", fa_bwd_ms)
    print(f"Forward speedup (FA / BSA): {statistics.fmean(fa_ms) / statistics.fmean(bsa_ms):.3f}x")
    if not args.forward_only:
        print(
            "Backward speedup (FA / BSA): "
            f"{statistics.fmean(fa_bwd_ms) / statistics.fmean(bsa_bwd_ms):.3f}x"
        )
    print(f"Output samples: BSA={bsa_checksum:.6g}, FA={fa_checksum:.6g}")
    if not args.forward_only:
        print(f"Q-gradient samples: BSA={bsa_grad_checksum:.6g}, FA={fa_grad_checksum:.6g}")

    if args.check:
        print(
            "Correctness comparison skipped: BSA uses a sparse mask while "
            "FA intentionally computes unmasked full attention."
        )


if __name__ == "__main__":
    main()
