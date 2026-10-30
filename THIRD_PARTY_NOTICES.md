# Third-party code

Original integration and Ascend adaptations are distributed under Apache-2.0.
Vendored components retain their original licenses; the root license does not
replace the licenses below. Model weights are not included.

| Component | Source revision | License / location |
| --- | --- | --- |
| Spark-H3 algorithm helpers | `zechengtang/Spark-H3` `2eb2dc1be261490f86a57c93904551a5bf456af7` | Apache-2.0, `LICENSES/Spark-H3-Apache-2.0.txt` |
| Miowtion / Veda | `veda-sparse/Miowtion` `ffbb4c29d9b1f19f026dec0c0485840fffe4639f` | MIT, `LICENSES/Miowtion.txt` |
| VDN algorithm reference | `OpenVDN/vdn-minimax-h3` `e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b` | Apache-2.0; runtime learned layers are loaded from vLLM-Omni |
| FastVideo VSA reference | `hao-ai-lab/FastVideo` `3d8ac9d14bd697a89ede8f170cbfbca012a9edcc` | Apache-2.0, `LICENSES/FastVideo-Apache-2.0.txt` |
| New PRO6000 VSA reference benchmark | `hao-ai-lab/FastVideo` `d4f144581d6882d5bceb288ca13d767c054ca041` | Separate unchanged official CUDA implementation; source hashes retained with results |
| Ascend block sparse attention and derived CANN operators | source snapshots recorded in `provenance/imported-sources.json` | CANN Open Software License Agreement 2.0, `LICENSES/CANN-Open-Software-2.0.txt` |

The CANN-derived source is for Huawei processor/software systems under its
original license. The NVIDIA reference results come from separate upstream
CUDA implementations. This repository does not port CANN source to NVIDIA.

`provenance/imported-sources.json` records imported file hashes before this
integration. Adaptations include private Python namespaces, configurable source
and cache locations, per-request geometry, global head indices for rank-local
Veda predictors, native attention registration, and explicit VSA routing budgets.
