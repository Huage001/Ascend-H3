# Ascend Block Sparse Attention

Standalone PyTorch/NPU wheel for the `BlockSparseAttention` and
`BlockSparseAttentionGrad` kernels migrated from `ops-transformer` revision
`f1cde5a4aae9ad6c04d92899a7115ed91c62e84a`.

The native launch and wheel layout follow `flex-attention/csrc`: AscendC
kernels are compiled into a static library, linked into one PyTorch operator
library, and loaded with `torch.ops.load_library` when the Python package is
imported. The extension currently targets Ascend 910B and supports TND/BNSD.

## Build a wheel

```bash
source /usr/local/Ascend/cann/set_env.sh
cd miowtion128pipe3_attention
./build.sh build
```

The wheel is written to `dist/`. The build also refreshes the in-place native
library under `miowtion128pipe3_attention/lib/`, so the source package can be used
directly without installing the wheel. Run `./build.sh clean` to remove build
outputs, generated native libraries, egg-info, and Python caches. With no
command, `./build.sh` defaults to `build`.

## Use

```python
import torch
from miowtion128pipe3_attention import miowtion128pipe3_attention

B, H, S, D = 1, 8, 2048, 128
q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="npu", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)
mask = torch.ones(B, H, S // 128, S // 128, dtype=torch.int8, device="npu")

out = miowtion128pipe3_attention(
    q, k, v, mask, block_shape=(128, 128), input_layout="BNSD",
    num_key_value_heads=H, scale_value=D**-0.5, inner_precise=0,
)
out.sum().backward()
```

`block_sparse_mask == 1` means that the corresponding Q/KV block participates
in attention. The forward kernel accepts `block_shape[1]` values that are
multiples of 64. Its internal KV compute tile remains 128 tokens, so adjacent
64-token sparse blocks may be gathered into one compute tile.
`scale_value` is passed through exactly like the `torch_npu` API; pass
`head_dim ** -0.5` for the conventional scaled dot-product attention value.

The migrated forward and backward kernels support head dimensions 64 and 128.
Q sparse blocks may have any positive size; KV sparse blocks must be positive
multiples of 64. Both kernels target Ascend 910B and accept FP16/BF16 tensors
in TND or BNSD layout.

On BNSD, backward uses a capacity-driven grouped pipeline whenever one KV
sparse block fits the 128-token internal tile. The kernel reads the ordinary
Q-major uint8 `[B, H, Q_blocks, KV_blocks]` mask directly; it does not build or
consume a packed bitmask. It groups Q tasks, skips empty KV columns, keeps
reusable operands in L1, accumulates gradients in L0C, and pipelines AIV
softmax work. In the packed path, dQ and dK share the same dS tile in L1: dK
uses a fractal-aware L1-to-L0 transpose instead of reading dS from GM again.
The BNSD post stage also writes each core's contiguous output interval directly.
These decisions are based on layout and internal tile capacity rather than a
special case for one block value; larger KV blocks retain the general tiled
fallback.

The accuracy matrix includes FP16/BF16, head dimensions 64/128, Q block sizes
1/17/128/129, and KV block sizes 64/128/192/256:

```bash
BSA_TEST_DEVICE=0 ASCEND_DEVICE_ID=0 pytest -q tests/test_backward_accuracy.py
```

## Benchmark

Run [tests/benchmark_block_sparse_vs_fusion.py](tests/benchmark_block_sparse_vs_fusion.py)
from the repository root after building the native library. The script requires
PyTorch, `torch_npu`, and a configured CANN environment. It prefers the local
in-place build and prints the Python package and native library paths used.
Adjust the CANN installation path and device ID for your machine.

```bash
source /usr/local/Ascend/cann/set_env.sh
ASCEND_DEVICE_ID=6 python3 tests/benchmark_block_sparse_vs_fusion.py \
  --device 6 --warmup 3 --iters 10
```

The default V6A workload uses BF16, B=1, H=24, S=53248, D=128, 128x128
sparse blocks, and seed 1234. It pads 53223 natural tokens and selects
58 of 416 KV blocks per query block (86.0577% sparsity). This is a large
workload; use the random-input example below for a smaller smoke run.

The script reports forward/backward mean, p50, and p90 latency, plus speedup
as **FA time / BSA time**. Normal timing synchronizes the NPU around each
call; backward timing excludes construction of the forward graph but includes
the host launch and synchronization overhead. Mask construction is not timed.
FA uses the same Q/K/V but **no mask**, so this is a sparse-versus-dense
performance comparison, not an accuracy comparison. Both use `inner_precise=0`;
use the accuracy tests for correctness (`atol=rtol=0.008`).

### Device profiling

Add `--msprof` to collect device task durations and pipeline utilization:

```bash
ASCEND_DEVICE_ID=6 python3 tests/benchmark_block_sparse_vs_fusion.py \
  --device 6 --warmup 3 --iters 10 --msprof \
  --msprof-output ./msprof_attention_results
```

`msprof` must be on `PATH` (provided by CANN). Each run creates a timestamped
directory, prints the exported `op_summary_*.csv` path, and reports statistics
with warmup calls excluded. These are device task times, not the synchronized
host-side timings of the normal benchmark. Profiling output defaults to
`./msprof_attention_results` relative to the working directory and is ignored
by Git. See [the final profiling report](docs/FINAL_PROFILE.md) and
[optimization notes](docs/BACKWARD_PERFORMANCE.md) for archived results.

### Smaller or custom workloads

Use `--input-mode random` to change the shape or sparsity; V6A mode requires
its fixed shape, block sizes, and BF16 dtype.

```bash
ASCEND_DEVICE_ID=6 python3 tests/benchmark_block_sparse_vs_fusion.py \
  --device 6 --input-mode random --batch-size 1 --num-heads 8 \
  --seq-len 2048 --head-dim 128 --dtype bf16 --sparsity 0.75 \
  --q-block-size 128 --kv-block-size 128 --warmup 3 --iters 10
```

`--forward-only` skips backward. `--seed` controls input generation;
`--warmup` and `--iters` default to 3 and 5. Use `--help` for all options.
For eligible BNSD inputs (KV block=128 or block shape=64x64),
`BSA_BACKWARD_OPT=0` forces the legacy
backward schedule and `BSA_BACKWARD_OPT=1` forces the optimized schedule;
leave it unset to benchmark automatic dispatch. Automatic optimization requires
BNSD, D=128, and either KV block=128 or block shape=64x64;
it has no additional KV-head-count or Q/KV
sequence-length thresholds. KV=192/256 backward has a
known timeout in the accuracy matrix; use KV=128 for the validated benchmark.
For 64x64, the legacy override (and default D=64) retains the original grouped
path; automatic D=128 and the optimized override use the wide packet pipeline.
Other Q block sizes with KV=64, TND, and KV blocks larger than 128 are unchanged.
