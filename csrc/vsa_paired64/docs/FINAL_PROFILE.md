# Final optimized BSA profiling — 2026-09-06 02:13 UTC

Current installed final library, SHA256 checked before and after collection:
`9d46dea68cb0d02d9ca503cd5766f56315bf0259a2dead015d3d8ba6bc49e114`.

Ascend 910B3, NPU 6, CANN 9.1.0. NPU 6 had no running processes before
collection. No `BSA_BACKWARD_OPT` override: automatic optimized dispatch.
Default V6A: BF16, B=1, H=24, S=53248, D=128, 128x128 blocks, seed 1234,
58 of 416 KV blocks selected per query block. FA is unmasked dense attention.

Reproduction command, from the repository root (the script was moved into
`tests/` after this collection):

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
env -u BSA_BACKWARD_OPT ASCEND_DEVICE_ID=6 python3 -u tests/benchmark_block_sparse_vs_fusion.py \
  --device 6 --warmup 3 --iters 10 --msprof \
  --msprof-output ./msprof_attention_results
```

Device task durations from msprof, not synchronized host-side backward timing:

| Operator | Mean (ms) | p50 (ms) | p90 (ms) |
| --- | ---: | ---: | ---: |
| BSA forward | 46.612 | 46.602 | 46.967 |
| BSA backward | 105.700 | 105.671 | 105.836 |
| FA forward | 317.936 | 317.900 | 318.382 |
| FA backward | 586.968 | 587.035 | 588.171 |

FA/BSA: 6.821x forward, 5.553x backward. The 100 ms backward target is
still not reached. The earlier 108.302 ms result was a separate normal
benchmark of synchronized `backward()`, not a device kernel timing.

CSV: [op_summary_20260906021343.csv](op_summary_20260906021343.csv).

The 13 backward records comprise three warmups (CSV lines 44, 47, 50)
and ten measured calls (lines 53, 56, 59, 62, 65, 68, 71, 74, 77, 80).
Measured backward durations in milliseconds:
105.690732, 105.646490, 105.926496, 105.548189, 105.649731,
105.651911, 105.835935, 105.814574, 105.514508, 105.717912.
Mean: 105.6996478 ms; min/max: 105.514508/105.926496 ms.

Collection console log: `/tmp/bsa_final_profiling.log`.
