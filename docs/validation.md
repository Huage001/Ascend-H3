# Validation status

Current reporting uses **eight actual forwards for all five methods**, with
Full + official LightX2V8 as baseline. Common seed **7** was visually selected using VSA V2 at 1344×768 / 345
frames; the complete single-card matrix is recorded below; the separately requested TP8 study is ongoing. Previous Full4-reference and seed42 results are historical.

| Check | Status |
| --- | --- |
| CPU launch/schedule/asset/figure and profiler model registry | 21 checks and 8 subtests pass; all default students use eight forwards |
| Source CANN operators | Previously compiled and verified; operators are unchanged by the step-count switch |
| LightX2V 8-step 768p source | Fixed revision and SHA256 in `configs/lightx2v8.json`; source SHA256 and static preparation verified on both platforms |
| VSA V2 full checkpoint | All 87 downloaded/reused files hash verified on both platforms |
| VSA V2 BF16, single-card 124 and 345 frames | Current seed7 cases pass on both platforms; earlier seed42 results remain archived |
| VSA V2 BF16, TP2, 124 frames | Passed at seed42; this establishes functional head sharding, not single-card timing |
| Native attention-head TP support | Native launch support for 1/2/4/7/8 ranks is retained. Full BF16 eight-step TP2 passed at seed7; further TP2 tests were cancelled by the user. The new TP8 performance study is separate and ongoing; no complete new TP4 qualification is claimed |
| VSA V2 online INT8 loader | Derived-scale validation bug fixed without weakening trained-weight or serialized-scale checks; new eight-forward long videos pass on both platforms |
| Matched dormant-vision initialization | Exact file-backed CPU storage for 27 unused input-vision blocks in VSA/VDN BF16 on both platforms; active DiT/text offload unchanged. Both long-video methods pass all four requests on both platforms |
| NPU VDN temporary memory | Complete-window KV gather batching bounds workspace without excluding keys; complete eight-forward BF16/INT8 long requests pass. Native bitwise equivalence for changed CANN call shapes is not asserted |
| Unified8 single-card matrix | All 40 cases accepted: 20 Ascend and 20 PRO 6000, five methods × BF16/INT8 × 124/345 frames at 1344×768. Every row uses eight actual forwards and seed7 |
| Current qualitative gallery | Ten Ascend videos (seed7 for Full/VDN/Spark; seed42 retakes for Veda/VSA) at 1344×768 / 124 native frames / 24 FPS / eight updates; five methods × BF16/INT8. Original/transfer SHA256 matches; showcase retakes are separate from timing and full audio/video decode pass. Eight sampled frames including final frame123 reviewed. All 21 language READMEs show the same ten native MP4 players in five rows and two columns |

A source build, unit check or server launch does not establish a complete
video result. Accept a row only after actual eight-forward/all-50-block audit,
finite outputs, VSA gate checks when applicable, full MP4 frame/audio checks,
uninstrumented request timing, an independent eight-forward profile and a
clean selected-device ownership guard. LoRAs merge before quantization and TP.

Both platforms use the same explicit Offload policy:
`omni-layer-prefetch1-vae-resident-v1`. Neither failed attempts nor visually
screened candidates are formal timing samples. Both BF16 and INT8 speedups use the same hardware's BF16 Full8 denominator. The higher relative Full8
speedup in each completed hardware pair is bold. No old four-step result is
rescaled into an eight-step claim.

[Current raw matrix](../benchmarks/single-card-highres.json) ·
[Detailed timing protocol](benchmark-protocol.md) ·
[Historical Full4-reference measurements](../benchmarks/archive/pre-unified8-full4-reference.json)
