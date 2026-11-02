<p align="center">
  <img src="../docs/assets/ascend-h3-logo.png" alt="Ascend-H3 logo" width="190">
</p>

<h1 align="center">Ascend-H3</h1>

<div align="center">

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](../LICENSE)
[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue)](../docs/environment.md)
[![OS: Linux](https://img.shields.io/badge/OS-Linux-lightgrey)](../docs/environment.md)
[![Platform: Ascend 910B3](https://img.shields.io/badge/platform-Ascend%20910B3-0077C8)](../docs/validation.md)
[![Backend: vLLM-Omni](https://img.shields.io/badge/backend-vLLM--Omni-orange)](../docs/environment.md)
[![Email: liusonghua@sjtu.edu.cn](https://img.shields.io/badge/%E2%9C%89-Email-0077C8)](mailto:liusonghua@sjtu.edu.cn)

</div>

<p align="center"><a href="#install">Installation</a> | <a href="#models">Models</a> | <a href="#serve">Serve</a> | <a href="#performance">Performance</a></p>

<div align="center">

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | **Tiếng Việt** | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> README hoàn chỉnh này là bản dịch có sự hỗ trợ của AI của README tiếng Anh tại cam kết be5c42e03f9a. Nó có thể trở nên lỗi thời; tham khảo phiên bản tiếng Anh để biết thông tin mới nhất. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

Suy luận MiniMax H3 trên Ascend với sự chú ý đầy đủ, VSA, Veda, VDN và Spark-H3 trong một kho lưu trữ. Việc tích hợp tái sử dụng tải mô hình vLLM-Omni, lượng tử hóa biến áp INT8 và các đầu chú ý song song tensor gốc.

**Phạm vi đã xác thực:** 40/40 ô INT8 E2E ở 1344×768, 124/345 khung và 1/2/4/8 NPU (37 phép đo mới và 3 ô Giảm tải lớp lịch sử được đánh dấu rõ ràng). So sánh BF16 thường trú trên Ascend 910B3 và PRO 6000 chứa 20 phép đo lõi chú ý và 20 phép đo khối hoàn chỉnh. Các toán tử chú ý giữ lại các phiên bản đã được xác thực của chúng.

## Khối lượng công việc 768p mặc định

| trường hợp | Chiều rộng × chiều cao | Khung đầu ra gốc | Lưới video tiềm ẩn |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

FPS đầu ra là 24. Yêu cầu sử dụng số lượng khung hình rõ ràng, với tính năng nội suy khung bị tắt. Trường hợp dài là khối lượng công việc 345 khung hình gốc đã được thỏa thuận; nó không bị cắt thành 341 khung hình. BF16 và INT8 sử dụng hai khối lượng công việc 1344×768 này. Bỏ qua `--case` sẽ chọn `1344-124`.

<a id="install"></a>

## Cài đặt

Sử dụng Linux, Python 3.12 và môi trường Ascend Torch/CANN tương thích. Cài đặt ngăn xếp vLLM, vLLM-Ascend và vLLM-Omni được ghim được mô tả trong [ghi chú môi trường](../docs/environment.md). Không kết hợp các bản phát hành khung không liên quan. Sau đó cài đặt kho lưu trữ này từ nguồn:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

Các bản dựng không chứa trọng lượng mô hình và chạy mà không chiếm NPU. Thư viện gốc được biên dịch cục bộ và được lưu giữ bên ngoài thanh toán. Đối với một chương trình phụ trợ duy nhất, hãy thay thế `all` bằng `vsa`, `veda` hoặc `spark`; Đầy đủ và VDN không yêu cầu các tiện ích mở rộng CANN tùy chỉnh này. Giữ sẵn thanh toán có thể chỉnh sửa.

<a id="models"></a>

## Điểm kiểm tra và lựa chọn thuật toán

| phương pháp | Nội dung mô hình bắt buộc | tính toán |
| --- | --- | --- |
| `full` | Đế H3; LightX2V 8 bước theo mặc định hoặc cộng đồng LoRA | Sự chú ý dày đặc |
| `vsa` | Hoàn thành học viên FastH3 V2 chính thức, bao gồm tất cả 50 cổng nén được đào tạo | 8 chuyển tiếp, độ thưa thớt video 80%; logic KV64, các khối KV được chọn ghép thành KV128 trên chip |
| `veda` | Cơ sở H3 + LightX2V 8 bước + công cụ dự đoán Veda được đào tạo chính thức | Các tuyến khối Veda động trên mỗi đầu, độ thưa thớt danh nghĩa 90% |
| `vdn` | Cơ sở H3 + thư mục điểm kiểm tra DMD giai đoạn VDN chính thức | Softmax cục bộ cộng với nhánh DeltaNet đã học 128 chiều |
| `spark` | Đế H3 + LightX2V 8 bước | Phân vùng phân cấp Spark-H3 và softmax chính xác/proxy chung, độ thưa nhánh chính xác danh nghĩa 90% |

Tải xuống các mô hình theo giấy phép ban đầu của họ. Trọng lượng không được phân phối lại. Những học sinh giỏi khác nhau có lịch trình và quy tắc học tập khác nhau; trao đổi sự chú ý trên các trọng số tùy ý không đảm bảo chất lượng phù hợp. VSA yêu cầu các cổng được huấn luyện khác 0. VDN yêu cầu trọng số nhánh đã học của nó. Các lệnh tải xuống và chuẩn bị có trong [nội dung mô hình](../docs/weights.md).

Theo mặc định, Veda và Spark đầy đủ sẽ bật **LightX2V tám bước LoRA** được ghim. Sử dụng `--no-few-step-lora` để chạy mô hình nhiều bước ban đầu. Cộng đồng `--lora /models/style` đã chuẩn bị sẵn sẽ được hợp nhất cùng với LightX2V trước khi tải xong; vô hiệu hóa sinh viên mặc định khi chọn một sinh viên cộng đồng khác một cách rõ ràng. Veda luôn cần công cụ dự đoán được đào tạo chính thức. VSA chỉ chấp nhận điểm kiểm tra đầy đủ chính thức hoàn chỉnh **FastH3 V2**: tám chuyển tiếp, độ thưa thớt video 80%, dịch chuyển video/âm thanh 10/3, ô64, hướng dẫn 1. Nó tải trực tiếp các trọng số đã phát hành và không lấy `--lora`. VDN chỉ chấp nhận điểm kiểm tra DMD giai đoạn chính thức hoàn chỉnh. Không chấp nhận cờ cộng đồng bổ sung LoRA hoặc cờ cấm học sinh.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

Nguồn PEFT chính thức được chuẩn hóa sau khi sử dụng alpha/cấp bậc đã khai báo của nó; các tập tin đã chuẩn bị đã được kiểm tra băm. Tất cả các bộ điều hợp đã chọn sẽ hợp nhất trước khi cắt TP và lượng tử hóa INT8. Không có phép nhân ma trận LoRA theo thời gian yêu cầu. Xem [chuẩn bị mô hình](../docs/weights.md) để biết các định dạng nguồn và lịch trình cố định.

<a id="serve"></a>

## Khởi động một máy chủ

Chỉ sử dụng các thiết bị không có quy trình tính toán khác. Đối với một thẻ:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Các chương trình phụ trợ khác sử dụng cùng một API máy chủ:

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

Chạy một máy chủ tại một thời điểm trên các thiết bị đã chọn. Thêm `--dry-run` để kiểm tra lệnh vLLM-Omni gốc được tạo mà không cần tải mô hình.

Các ví dụ về một thẻ truyền phát bộ mã hóa văn bản trong khi vẫn giữ INT8 DiT cư trú. Công thức gốc định lượng DiT; bộ mã hóa văn bản vẫn giữ nguyên BF16, do đó, dung lượng bộ nhớ của nó vẫn cần được đưa vào chính sách khởi chạy. Đối với **BF16**, hãy sử dụng `--precision bf16`; nếu cần, hãy chọn `--offload layer --no-precompute-adaln` một cách rõ ràng. INT8 sử dụng công thức gốc:

```json
{"transformer":{"method":"int8"}}
```

Các phép chiếu INT8 đưa BF16 Q/K/V vào sự chú ý. Bộ tích lũy chú ý, dự đoán Veda và trạng thái VDN duy trì độ chính xác như quy định. Đây là suy luận mô hình INT8, không phải khẳng định rằng mọi hoạt động chú ý đều là INT8. Công thức gốc giữ lại các lớp không được hỗ trợ trong BF16. Thẻ đơn H3 có hình chiếu rộng 96.768 vượt quá giới hạn đầu ra matmul lượng tử hóa của NPU là 65.535; TP có thể giảm chiều rộng mỗi cấp bậc của nó xuống phạm vi.

Đối với **suy luận bốn thẻ**, hãy sử dụng phân đoạn đầu song song tensor gốc:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` hỗ trợ 1, 2, 4, 7 và 8, chia 56 đầu. Veda giữ lại ID đầu dự đoán toàn cầu sau khi phân chia. DiT TP bảy thẻ sử dụng một thứ hạng bộ mã hóa văn bản vì 64 đầu truy vấn / 8 KV của nó không thể chia cho bảy. Khởi chạy mặc định thành `--offload none` (trọng số thường trú). Nếu bộ nhớ không đủ, `--offload text` chỉ truyền bộ mã hóa văn bản trong khi vẫn giữ DiT trên thiết bị; `--offload module` hoán đổi các thành phần hoàn chỉnh (trình độ thử nghiệm); `--offload layer` truyền phát cả hai thành phần, giữ một khối hoạt động và một khối được tìm nạp trước. Tham chiếu giảm tải khớp hoàn chỉnh bên dưới sử dụng rõ ràng `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) trên cả hai nền tảng phần cứng. Các phép đo từ đầu đến cuối mới chỉ sử dụng **INT8**, bắt đầu bằng việc thực thi thường trú và đánh dấu mọi dự phòng Giảm tải cần thiết. Sự chú ý biệt lập và so sánh khối DiT sử dụng BF16 trên cả hai thiết bị: lớp 0/24/49, mỗi lớp ba lần chạy, tổng hợp chín thời gian và chia cho chín. `--parallel ulysses` tùy chọn được hiển thị cho các đường dẫn không phải VDN; TP vẫn là mặc định. Giải mã VAE được xếp theo ô gốc hiện được đặt mặc định cho tất cả các thẻ đã chọn (`--vae-parallel-size 0`); sử dụng `--vae-parallel-size 1` cho một thứ hạng bộ giải mã duy nhất. Đường ống INT8 Full / 124 khung hình đã vượt qua với một, hai, bốn và tám cấp VAE. Các kết hợp phương pháp/số khung khác vẫn còn trong ma trận E2E.

NPU thường trú ra mắt chuẩn bị kết quả AdaLN trước khi máy chủ sẵn sàng. Quá trình chuẩn bị giải quyết mô hình đã tải và các vị trí sigma thực tế của LoRA, số lần cập nhật DiT và các thay đổi video/âm thanh; nó không được mã hóa cứng thành tám bước. Sử dụng `serve --steps N` để chỉ định lần khởi chạy của giáo viên/học sinh tùy chỉnh và sử dụng cùng số lượng trong `generate --steps N`. Lịch trình điểm kiểm tra chưng cất xung đột bị từ chối. Dung lượng bộ nhớ đệm tăng lên theo các kế hoạch được yêu cầu. Phép chiếu ban đầu được giữ lại; Trọng số chính của AdaLN nằm trên CPU, trong khi kết quả đã chuẩn bị vẫn nằm trên NPU. TP4/TP8 cũng ghi nhớ các phần nhúng thời gian gốc từ đầu vào dấu thời gian chính xác để tránh biến thể giảm FP32 bit thấp làm mất hiệu lực các bộ nhớ đệm xuôi dòng. Trọng lượng, cài đặt số và thỏa thuận truy cập tất cả các cấp vẫn đảm bảo việc sử dụng lại. Khởi động lại với cấu hình mới để tính toán trước một lịch trình khác; đầu vào không được chuẩn bị trước sẽ sử dụng tính toán ban đầu và có thể chuyển trọng số. `--no-precompute-adaln` vô hiệu hóa việc chuẩn bị; Truyền phát mô-đun/lớp DiT giữ cho nó bị tắt theo mặc định.

Đối với Spark/VSA/VDN 345 khung thẻ đơn, hãy sử dụng `--offload layer --no-precompute-adaln` với các toán tử được xác thực ban đầu. Bảng E2E đánh dấu các ô Giảm tải lớp gốc này bằng †L. Tất cả các tế bào khác vẫn giữ nguyên chính sách cư trú được ghi lại riêng của chúng.

Trình khởi chạy đặt thời hạn yêu cầu của nhân viên gốc là 7200 giây để video dài không bị hủy theo mặc định 600 giây ngược dòng. Đặt `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` trước khi khởi chạy để ghi đè giới hạn này; nó thay đổi hành vi chờ đợi, không phải tính toán mô hình hoặc thời lượng yêu cầu được đo.

## Tạo hai trường hợp 768p

Tạo tệp nhắc UTF-8, khởi động máy chủ, sau đó sử dụng số bước phù hợp của nó:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Truyền `--method` của máy chủ cho máy khách: tất cả năm phương thức (Full/VSA/Veda/VDN/Spark) mặc định là tám chuyển tiếp thực tế. Đối với mô hình nhiều bước gốc tùy chọn, hãy sử dụng `generate --method full --no-few-step-lora` (50 bước), khớp với máy chủ. `--steps` rõ ràng vẫn có sẵn cho sinh viên cộng đồng được chuẩn bị riêng. `--guidance-scale` là tùy chọn; thiếu sót duy trì chính sách gốc. Không có giai đoạn Tinh chỉnh nào được thêm vào. Máy khách lưu MP4 và biên nhận thời gian JSON.

<!-- BEGIN BF16 LAYER RESULTS -->

## So sánh lớp BF16: cư trú trên cả hai thiết bị

Chỉ 1344×768, 124/345 khung hình. Trọng số thực của học sinh và trạng thái ẩn tổng hợp đông lạnh giống hệt byte; đây không phải là các kích hoạt lớp được ghi lại từ yêu cầu video. Mỗi lớp 0, 24 và 49 nhận được hai lần khởi động không tính thời gian và ba lần chạy chính thức: **tổng chín lần và chia cho chín**. Trọng lượng vẫn ở trên cả hai thiết bị. Kích hoạt/trọng lượng chính là BF16; các yếu tố dự đoán, tích lũy FP32 cần thiết và trạng thái VDN được giữ lại. Tăng tốc sử dụng cùng một phần cứng, cùng hình dạng BF16 Đường cơ sở đầy đủ. Tốc độ tăng tốc tương đối cao hơn là đậm; độ trễ tuyệt đối cũng được hiển thị.

### Chỉ cốt lõi chú ý

Bắt đầu với Q/K/V đã chuẩn bị sẵn và kết thúc ở đầu ra chú ý. Không bao gồm phép chiếu QKV, định mức Q/K, RoPE, phép chiếu đầu ra, AdaLN và FFN. Bao gồm các cổng thưa thớt, dự đoán, lựa chọn, sắp xếp lại và phục hồi cần thiết; VDN bao gồm cả lõi tuyến tính và softmax hoàn chỉnh. Mọi phương pháp đều sử dụng ranh giới này; thời gian của toàn bộ mô-đun không được thay thế. Thời gian cốt lõi đã hoàn thành 10/10 cặp phần cứng phù hợp; các ô đang chờ xử lý không sử dụng lại các phép đo mô-đun trước đó.

| Khung | phương pháp | Độ trễ Ascend | Độ trễ PRO 6000 | Tăng tốc Ascend | Tăng tốc PRO 6000 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 343.31 ms | 111.64 ms | 1.00× | 1.00× |
| 124 | VSA V2 · 80% | 216.24 ms | 93.56 ms | **1.59×** | 1.19× |
| 124 | Veda · 90% | 75.50 ms | 25.01 ms | **4.55×** | 4.46× |
| 124 | VDN · official | 368.89 ms | 142.74 ms | **0.93×** | 0.78× |
| 124 | Spark-H3 · 90% | 128.54 ms | 39.35 ms | 2.67× | **2.84×** |
| 345 | Full | 2578.60 ms | 838.10 ms | 1.00× | 1.00× |
| 345 | VSA V2 · 80% | 1196.30 ms | 498.34 ms | **2.16×** | 1.68× |
| 345 | Veda · 90% | 380.26 ms | 123.45 ms | 6.78× | **6.79×** |
| 345 | VDN · official | 1093.46 ms | 423.12 ms | **2.36×** | 1.98× |
| 345 | Spark-H3 · 90% | 636.98 ms | 184.38 ms | 4.05× | **4.55×** |

### Hoàn thành khối DiT

Chạy từ đầu vào khối đến đầu ra khối, bao gồm các phép chiếu, chuẩn hóa, RoPE, lõi chú ý, AdaLN, FFN và các hoạt động dư. Đo độc lập với lõi; sự khác biệt của chúng không phải là phép đo FFN và cũng không được ngoại suy thành thời gian E2E.

| Khung | phương pháp | Độ trễ Ascend | Độ trễ PRO 6000 | Tăng tốc Ascend | Tăng tốc PRO 6000 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 491.51 ms | 185.93 ms | 1.00× | 1.00× |
| 124 | VSA V2 · 80% | 346.40 ms | 166.56 ms | **1.42×** | 1.12× |
| 124 | Veda · 90% | 213.56 ms | 97.76 ms | **2.30×** | 1.90× |
| 124 | VDN · official | 527.75 ms | 222.89 ms | **0.93×** | 0.83× |
| 124 | Spark-H3 · 90% | 267.42 ms | 112.65 ms | **1.84×** | 1.65× |
| 345 | Full | 3000.60 ms | 1054.58 ms | 1.00× | 1.00× |
| 345 | VSA V2 · 80% | 1550.07 ms | 703.92 ms | **1.94×** | 1.50× |
| 345 | Veda · 90% | 787.91 ms | 330.60 ms | **3.81×** | 3.19× |
| 345 | VDN · official | 1534.19 ms | 647.19 ms | **1.96×** | 1.63× |
| 345 | Spark-H3 · 90% | 1028.43 ms | 393.06 ms | **2.92×** | 2.68× |

![So sánh lõi chú ý BF16 và khối DiT](../docs/figures/bf16-three-layer.png)[Tất cả các bản ghi chín lần chạy và kiểm tra danh tính đầu vào](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1/2/4/8 thẻ

1344×768, tám bản cập nhật DiT thực tế, bao gồm mã hóa văn bản, giải mã video/âm thanh và ghi MP4. Đang tải, việc hợp nhất LoRA trước đó, quá trình chuẩn bị và khởi động AdaLN đều bị loại trừ. Ưu tiên cư dân DiT; các hàng thường trú xác minh không có lỗi AdaLN nào và chuyển trọng số ở mọi cấp độ xung quanh yêu cầu chính thức. Sau khi ghi lại OOM, các ô 345 khung hình thẻ đơn được đánh dấu sử dụng đối sánh Giảm tải và tái sử dụng lớp gốc, xác thực đầy đủ các phép đo trước đó theo yêu cầu; những ô này được đánh dấu rõ ràng thay vì được trình bày dưới dạng số đo mới.

| Khung | phương pháp | 1 thẻ | 2 thẻ | 4 thẻ | 8 thẻ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 197.66 s †T | 116.03 s | 64.78 s | 35.87 s |
| 124 | VSA V2 · 80% | 146.81 s †T | 93.57 s | 53.47 s | 30.40 s |
| 124 | Veda · 90% | 97.40 s †T | 69.66 s | 43.38 s | 25.33 s |
| 124 | VDN · official | 216.91 s †T | 136.02 s | 79.26 s | 44.36 s |
| 124 | Spark-H3 · 90% | 117.11 s †T | 79.41 s | 48.87 s | 29.05 s |
| 345 | Full | 1211.17 s †T | 649.72 s †T | 328.63 s | 178.81 s |
| 345 | VSA V2 · 80% | 617.55 s †L | 363.99 s †T | 193.33 s | 104.71 s |
| 345 | Veda · 90% | 337.95 s †T | 225.31 s †T | 130.11 s | 75.93 s |
| 345 | VDN · official | 620.67 s †L | 388.58 s †T | 209.84 s | 117.27 s |
| 345 | Spark-H3 · 90% | 419.44 s †L | 267.61 s †T | 150.50 s | 87.02 s |

Bao gồm **40/40**: 37 phép đo mới và 3 phép đo trước đó được sử dụng lại. †T: Chỉ giảm tải lớp mã hóa văn bản; †L: DiT/lớp mã hóa văn bản gốc Giảm tải với tìm nạp trước một lớp và VAE cố định. Các ô không được đánh dấu không có thời gian giảm tải. Chỉ các hàng thường trú mới tính toán trước AdaLN; các hàng giảm tải lớp không tạo ra yêu cầu chuyển giao DiT hoặc zero-AdaLN. VAE xếp hạng số lượng thẻ trận đấu; Đầu DiT chia 56/28/14/7. Vùng phủ sóng INT8 gốc được giữ lại.[Bản ghi INT8 E2E](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

So sánh giảm tải lớp phù hợp trước đó được giữ lại trong [tham chiếu thẻ đơn lịch sử](../docs/archive/matched-offload-single-card.md). Các ô được sử dụng lại trong bảng hiện tại liên kết đến các bản ghi gốc không thay đổi.

## Kết quả định tính và xác nhận

Chỉ Ascend 910B3: **1344×768, 345 khung hình gốc, 24 FPS (14,375 giây, khoảng 14,4 giây)**, có âm thanh.

Tất cả năm phương pháp đều sử dụng tám chuyển tiếp. Bên trái: BF16. Phải: W8A8 INT8 gốc. Tất cả các mẫu đều sử dụng một thẻ, cùng một lời nhắc, hạt giống chung **7** và Giảm tải phù hợp.

> Một người đi xe đạp đạp xe dọc theo con đường công viên xanh yên tĩnh, cảnh quay theo dõi mượt mà như điện ảnh, ánh sáng ban ngày tự nhiên.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Đầy đủ + LightX2V8 · 8 bước**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Đầy đủ + LightX2V8 · 8 bước**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 bước**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 bước**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 bước**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 bước**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · chính thức · 8 bước**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · chính thức · 8 bước**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 bước**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 bước**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 sử dụng độ thưa thớt video 80%; Veda sử dụng 90%. Văn bản và âm thanh vẫn giữ nguyên các quy tắc gốc của chúng.[Thông số, SHA256 và xuất xứ của tất cả mười video](../examples/videos/qualitative-1344-345.json). Giải mã đầy đủ xác minh 345 khung hình và âm thanh; 16 khung hình thống nhất trên mỗi clip đã được xem xét trực quan. Những học sinh xuất sắc không được khẳng định là tương đương về điểm ảnh hoặc cảm nhận. VSA giữ lại các thay đổi về lượt bắn; không có đảm bảo không nhấp nháy được thực hiện.

Mục tiêu tích hợp ban đầu là T2VA đợt một mà không có video tham chiếu. Xem [kiến trúc](../docs/architecture.md),[trạng thái xác thực](../docs/validation.md) và[thông báo của bên thứ ba](../THIRD_PARTY_NOTICES.md).
