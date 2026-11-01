# Serving guide

## Launch examples

Use only devices that have no other compute processes. For one card:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Other backends use the same server API:

```bash
h3-ascend serve --method vsa --model /models/FastH3-V2 \
  --precision int8 --devices 1 --offload text
h3-ascend serve --method veda --model /models/MiniMax-H3 \
  --predictor /models/veda/predictor.safetensors \
  --precision int8 --devices 1 --offload text
h3-ascend serve --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 --precision int8 --devices 1 --offload text
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Run one server at a time on the selected devices. Add `--dry-run` to inspect
the generated native vLLM-Omni command without loading a model.

## Precision and memory

The single-card examples stream the text encoder while keeping the INT8 DiT
resident. The native recipe quantizes the DiT; the text encoder remains BF16,
so its memory footprint still needs to be included in the launch policy.
For **BF16**, use `--precision bf16`; if necessary select
`--offload layer --no-precompute-adaln` explicitly. INT8 uses the native recipe:

```json
{"transformer":{"method":"int8"}}
```

INT8 projections feed BF16 Q/K/V into attention. Attention accumulators,
Veda predictions and VDN state keep their prescribed precision. This is
INT8 model inference, not a claim that every attention operation is INT8.
The native recipe retains unsupported layers in BF16. Single-card H3 has a
96,768-wide projection that exceeds the NPU quantized matmul output limit
of 65,535; TP can reduce its per-rank width into range.

## Multi-card inference

For **four-card inference**, use native tensor parallel head sharding:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--parallel tp` supports 1, 2, 4, 7 and 8 devices, which divide 56 heads.
`--parallel sp` supports 1, 2, 4 and 8 devices for all five methods. Veda retains
global predictor head IDs after sharding. Seven-card DiT TP uses one text
encoder rank because its 64 query / 8 KV heads cannot be divided by seven.
Launches default to `--offload none` (resident weights). If memory is insufficient,
`--offload text` streams only the text encoder while retaining the DiT on device;
`--offload module` swaps complete components (experimental qualification);
`--offload layer` streams both components, keeping one active and one prefetched
block. The completed [matched-Offload reference](archive/matched-offload-single-card.md)
explicitly uses
`--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) on both hardware
platforms. New end-to-end measurements use **INT8 only**, starting with resident
execution and marking every necessary Offload fallback. The isolated attention
and DiT-block comparison uses BF16 on both devices: layers 0/24/49, three runs
each, summing the nine timings and dividing by nine. `--parallel ulysses` is an alias for `sp`; VDN is supported too. TP remains
the default.
Native tiled VAE decoding now defaults to all selected cards
(`--vae-parallel-size 0`); use `--vae-parallel-size 1` for a single decoder rank.
The INT8 Full / 124-frame pipeline has passed with one, two, four and eight
VAE ranks. Other method/frame-count combinations remain in the E2E matrix.

## AdaLN preparation

Resident NPU launches prepare AdaLN results before server readiness. Preparation
resolves the loaded model and merged LoRA's actual sigma positions, DiT update
count, and video/audio shifts; it is not hard-coded to eight steps. Use
`serve --steps N` to specify a teacher/custom-student launch, and use the same
count in `generate --steps N`. A conflicting distilled checkpoint schedule is
rejected. Cache capacity grows with the required plans. The original projection
math is retained; AdaLN master weights stay on CPU, while prepared results stay
on NPU. TP4/TP8 also memoize native time embeddings from exact timestep inputs
to avoid low-bit FP32 reduction variation invalidating downstream caches.
Weights, numerical settings and all-rank hit agreement still guard reuse.
Restart with the new configuration to precompute a different schedule; an
unprepared input uses the original computation and may transfer weights.
`--no-precompute-adaln` disables preparation; DiT module/layer streaming keeps it
disabled by default.

## Long-video requests

For single-card 345-frame Spark/VSA/VDN, use `--offload layer --no-precompute-adaln`
with the original validated operators. The [E2E table](../README.md#performance)
marks these native layer
Offload cells with †L; ‡R identifies matching measurements reused from the
previous run. All other cells retain their individually recorded residency policy.

The launcher sets the native worker request deadline to 7200 seconds so a long
video is not aborted by the upstream 600-second default. Set
`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` before launch to override this limit; it changes
waiting behavior, not model computation or the measured request duration.


## Sequence parallel inference

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel sp --steps 8 --offload text

# VDN keeps its official student and original 128-dimensional linear branch.
h3-ascend serve --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 \
  --precision int8 --devices 4 --parallel sp --steps 8 --offload text
```

SP distributes token rows through FFN and residual operations while retaining
replicated DiT master weights. The default resident INT8 SP2/SP4 QKV path exchanges
the input activation and FP32 scale already required by native W8A8, then projects
only this rank's heads. Head-local weight caches are prepared after LoRA fusion
and quantization. Attention cores and BF16 output redistribution are unchanged;
there is no additional lossy communication quantization. On this path every rank
prepares a complete RoPE table once per request and shares it across layers and steps.
SP8 retains BF16 QKV redistribution because input AllGather was slower at that
degree. BF16 inference and DiT streaming also retain the original QKV path.
Set `H3_ASCEND_INPUT_EXCHANGE=0` to request that reference path on SP2/SP4.

VSA preserves the original gate projection and redistributes its gate separately.
On the input-exchange path, VDN reuses the full-sequence head-local raw QKV for its original linear branch,
retaining BF16 branch inputs, original features/inverse/scans, and readout.
VDN comparisons qualify against the same SP head partition; head partitioning
itself can have floating-point differences from a single full-head branch call.


### Measured SP memory settings

1344×768, INT8, eight updates; VAE decoding uses all selected cards by default.
Choose `--offload` for the requested frame count:

| SP cards | Frames and methods | Offload |
| ---: | --- | --- |
| 2 | 124/345, all five methods | `text` |
| 4 | 124, all five methods | `none` |
| 4 | 345, Full / VSA / Veda | `none` |
| 4 | 345, VDN / Spark | `text` |
| 8 | 124/345, all five methods | `none` |

All thirty selected SP records keep the DiT resident, with prepared AdaLN
results. Text Offload does not stream the DiT. The examples above use `text`
so that both supported frame counts fit; the short four-card cases were timed
with `none` as shown in the [fresh SP table](../README.md#sp-performance).

VDN SP2 / 345 retains the original BF16 QKV exchange to avoid the additional
head-projection weight cache. Its DiT still uses the native INT8 recipe:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
H3_ASCEND_INPUT_EXCHANGE=0 h3-ascend serve \
  --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 \
  --precision int8 --devices 2 --parallel sp --steps 8 --offload text
```

The other resident SP2/SP4 cases use the pre-QKV input exchange. SP8 keeps
the measured faster BF16 QKV path. No extra lossy communication quantization
is enabled.
