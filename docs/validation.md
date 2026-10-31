# Validation status

Current reporting uses **eight actual forwards for all five methods**, with
Full + official LightX2V8 as baseline. Common seed **7** was visually selected using VSA V2 at 1344×768 / 345
frames; the entire matrix is being remeasured. Previous Full4-reference and seed42 results are historical.

| Check | Status |
| --- | --- |
| CPU launch/schedule/asset/figure contracts | 16 checks pass; all default students use eight forwards |
| Source CANN operators | Previously compiled and verified; operators are unchanged by the step-count switch |
| LightX2V 8-step 768p source | Fixed revision and SHA256 in `configs/lightx2v8.json`; source SHA256 and static preparation verified on both platforms |
| VSA V2 full checkpoint | All 87 downloaded/reused files hash verified on both platforms |
| VSA V2 BF16, single-card 124 and 345 frames | Passed at seed42; archived because the new matrix requires a newly selected common seed |
| VSA V2 BF16, TP2, 124 frames | Passed at seed42; this establishes functional head sharding, not single-card timing |
| Full / Veda / Spark / VDN TP2, both precisions | Previous released student configurations passed real video/audio requests; new default LoRA8 qualification remains explicit |
| VSA V2 online INT8 loader | Derived-scale validation bug fixed without weakening trained-weight or serialized-scale checks; new-seed full requests pending |
| CUDA initialization memory cleanup | Initial unused host pages are released before constructing pinned CPU masters; full VSA V2 completion still pending |
| New unified8 single-card matrix | 20 cases per hardware platform (five methods × two precisions × two frame counts); seed7 long-video Full8 baseline jobs are running |
| New qualitative gallery | Seed7 VSA V2 BF16 long video complete; one cyclist with shot changes. Other eight-step samples are being regenerated. Screening is excluded from timing |

A source build, unit check or server launch does not establish a complete
video result. Accept a row only after actual eight-forward/all-50-block audit,
finite outputs, VSA gate checks when applicable, full MP4 frame/audio checks,
uninstrumented request timing, an independent eight-forward profile and a
clean selected-device ownership guard. LoRAs merge before quantization and TP.

Both platforms use the same explicit Offload policy:
`omni-layer-prefetch1-vae-resident-v1`. Neither failed attempts nor visually
screened candidates are formal timing samples. The higher relative Full8
speedup in each completed hardware pair is bold. No old four-step result is
rescaled into an eight-step claim.

[Current raw matrix](../benchmarks/single-card-highres.json) ·
[Detailed timing protocol](benchmark-protocol.md) ·
[Historical validation evidence](archive/validation-pre-unified8.md) ·
[Historical Full4-reference measurements](../benchmarks/archive/pre-unified8-full4-reference.json)
