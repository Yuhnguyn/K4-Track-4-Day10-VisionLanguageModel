# Báo cáo Lab Day 10 — Qwen3-VL-4B QLoRA

Họ tên: Nguyễn Huy
`dataset_version`: `d10-21e4070301`

Trạng thái: Đã có bằng chứng E0–E4.
Mọi số trong bảng lấy từ scorer của đề. Dấu — nghĩa là chưa đo, không phải 0.
E0 là lượt Colab T4 đã có trong lịch sử repo; E1–E4 chạy qua Kaggle CLI.


## Bảng kết quả

| Mã | Thiết lập | Tham số huấn luyện | VRAM GB | Phút train | Micro F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot | 0 | 4.1448 (suy luận) | 0.0000 | 0.0650 | 0.0441–0.0877 | 1.1897 | 0.0994 | 0.0040 | 0.1407 |
| E1 | LoRA LLM | 33,030,144 | 7.1228 | 45.1248 | 0.2738 | 0.2395–0.3137 | 1.4502 | 0.2066 | 0.1906 | 0.5319 |
| E2 | LoRA LLM + projector | 142,135,296 | 8.9433 | 53.7009 | 0.3158 | 0.2796–0.3546 | 1.1093 | 0.1680 | 0.2432 | 0.5714 |
| E3 | LoRA LLM + projector + LoRA ViT | 148,426,752 | 9.1324 | 64.8521 | 0.4078 | 0.3691–0.4491 | 1.1511 | 0.2477 | 0.3189 | 0.6981 |
| E4_DATA_SIZE_0.1 | Ablation data_size | 148,426,752 | 9.1336 | 55.0207 | 0.2453 | 0.2108–0.2806 | 1.4148 | 0.1836 | 0.1720 | 0.4776 |
| E4_DATA_SIZE_1.0 | Ablation data_size | 148,426,752 | 9.1324 | 64.8521 | 0.4078 | 0.3691–0.4491 | 1.1511 | 0.2477 | 0.3189 | 0.6981 |

E0 không huấn luyện; VRAM là suy luận. Log E0 ghi 1937,60 giây suy luận trong session, không gồm tải/nạp model.
Các số VRAM dùng GB thập phân theo `torch.cuda.max_memory_allocated()/1e9`.

### Câu 1 — Kiến trúc và các khối được cập nhật

ViT mã hóa ảnh thành đặc trưng thị giác; merger/projector đổi đặc trưng sang không gian embedding
của LLM; LLM sinh chuỗi JSON tự hồi quy. Theo GUIDE, ViT khoảng 415M tham số, LLM cùng
embedding khoảng 4B; projector nằm trong ViT, không được cộng hai lần vào tổng.
Kiểm tra cây module trên Kaggle xác nhận projector có 109.105.152 tham số.
E1 chỉ cập nhật LoRA LLM: 252 module, 33.030.144 tham số trainable được log thực tế.
E2 thêm toàn bộ projector; E3 thêm LoRA 96 module ViT, tổng 348 module LoRA.
Số phần tử lưu trữ của tensor NF4 không đại diện đúng số tham số model trước lượng tử hóa.

### Câu 2 — E0 và hai lỗi phổ biến

`parse_valid_rate`=0.536977, tương ứng 167/311 ảnh.
Phân tích output ghi 144 ảnh JSON bị cắt và 49 output hợp lệ có box lặp; các nhóm lỗi có thể giao nhau.
Ví dụ `vlcsnap-00200`: model lặp `{"label":"crack","bbox_2d":[750,670,843,700]}`
nhiều lần, hết 256 token trước khi đóng list. Đây vừa là lặp box vừa dẫn tới JSON không hợp lệ.
F1 toàn bộ E0=0.0650; không chỉ chấm những output parse được.

### Câu 3 — Thang tọa độ

YOLO `(xc,yc,w,h)` đổi sang `[xc-w/2,yc-h/2,xc+w/2,yc+h/2]*1000`, làm tròn và kẹp miền.
Khi vẽ, x nhân `640/1000`, y nhân `360/1000`. Nhầm norm1000 thành pixel sẽ làm box bị
phóng khác nhau theo hai trục, giảm IoU và thường giảm F1; không thể suy ra một F1 cụ thể khi chưa chấm.
Round-trip thực tế ghi sai số tối đa 0,32 pixel so với box nguồn đã kẹp, phù hợp sai số làm tròn.
Ví dụ `[400,400,600,600]` tương ứng pixel `[256,144,384,216]`.
Kiểm tra bằng dự đoán E1 thực tế: chấm đúng norm1000 cho F1=0.2738; chấm cùng output nhầm là pixel cho F1=0.0000. Không huấn luyện hay sửa output trong kiểm tra này.

### Câu 4 — Mask loss

Prompt và token ảnh được đặt label=-100; chỉ JSON đáp án và token kết thúc chịu loss.
Kiểm tra Kaggle trên `20250216_164325`: 639 token tổng, 389 token bị mask, 250 token được học,
trong đó prompt chứa 220 vision token. Decode token được học khớp JSON target và `<|im_end|>`.
Nếu học cả prompt, loss bị chi phối bởi việc sao chép chỉ dẫn cố định; học token placeholder ảnh
không trực tiếp tối ưu định vị. Loss thấp khi ấy chưa chứng minh đáp án JSON chính xác hơn.

### Câu 5 — Lượng tử hóa và projector

NF4 lưu trọng số nén, không phải tensor float có thể cập nhật đầy đủ bằng AdamW.
LoRA học ma trận phụ, còn E2/E3 yêu cầu cập nhật trực tiếp projector nên visual và lm_head
được loại khỏi lượng tử hóa. Kiểm tra thực tế thấy 252 module NF4, không module visual/lm_head bị NF4.
`llm_int8_skip_modules` gồm cả `model.visual` để khớp tên đầy đủ ở transformers đã ghim.
Projector trainable được đưa lên float32 để gradient, cập nhật và trạng thái optimizer ổn định;
autocast vẫn chạy phép tính phù hợp ở fp16. Đây là lựa chọn ổn định của lab, không phải quy luật
rằng mọi projector đều bắt buộc float32. Lưu riêng `projector.pt` và nạp lại cùng adapter khi suy luận.

### Câu 6 — Chia theo phiên quay

Hai frame gần nhau có nội dung gần trùng; chia ngẫu nhiên khiến train có frame gần giống test,
cho F1 lạc quan và không đo đúng khả năng tổng quát sang phiên quay mới.

| Split | Ảnh | pothole | crack | manhole | Phiên |
|---|---:|---:|---:|---:|---:|
| train | 1422 | 1074 | 1750 | 691 | 13 |
| val | 276 | 66 | 470 | 103 | 2 |
| test | 311 | 121 | 299 | 163 | 3 |

Khám phá dữ liệu xác nhận giao phiên giữa các split=0, duplicate image_id=0.
Train có một box crack diện tích 0 tại `vlcsnap-00058`, object index 1.
Chỉ bỏ annotation này khỏi target train, giữ ảnh và audit; giữ nguyên val/test và công cụ đề.

### Câu 7 — So sánh E1/E2/E3

E1: F1=0.2738, CI95=[0.23945783132530118, 0.3137254901960784], chênh E0=+0.2088. E2: F1=0.3158, CI95=[0.279646017699115, 0.35463546354635467], chênh E0=+0.2508. E3: F1=0.4078, CI95=[0.36910569105691055, 0.44912280701754387], chênh E0=+0.3428.
Nếu ưu tiên mức tăng F1 so với E0 trên mỗi phút train, tôi chọn E3 làm phương án đáng cân nhắc nhất trên T4. Đây là tiêu chí chi phí được nêu rõ, không phải kiểm định ý nghĩa thống kê; VRAM, F1 từng lớp và CI95 trong bảng vẫn là các giới hạn của lựa chọn.
Đánh giá lợi ích phải đặt F1 và F1 từng lớp cạnh thời gian/VRAM.
E2 thêm khoảng 109M tham số projector; E3 cần gradient qua ViT nên có thể tốn VRAM/thời gian hơn.
Không đồng nhất dự đoán lý thuyết về chi phí với số đo. Các số thực tế nằm trong bảng.

### Câu 8 — Khoảng tin cậy

Độ rộng CI95: E0: 0.0436. E1: 0.0743. E2: 0.0750. E3: 0.0800.
E0 có CI95=[0,0441; 0,0877], rộng khoảng 0,0436.
Với 311 ảnh và chỉ 3 phiên test, không nên kết luận từ chênh lệch F1 nhỏ.
CI của scorer bootstrap theo ảnh; các frame trong một phiên có tương quan nên CI này có thể
lạc quan hơn bootstrap theo phiên. CI chồng nhau không tự chứng minh hai setup tương đương;
kiểm định chênh lệch cần bootstrap ghép cặp cùng ảnh hoặc theo phiên, tốt hơn nữa thêm phiên độc lập.

### Câu 9 — Lỗi theo lớp

E0 yếu nhất ở crack: F1=0,0040 (TP=1, FP=197, FN=298), so với pothole=0,0994 và manhole=0,1407.
Crack có nhiều mẫu train nhất nên không thể giải thích đơn giản do ít dữ liệu.
Hình dạng dài/mảnh, box nhỏ, bóng đổ/vạch đường và định nghĩa vùng crack gây khó; đây là các giả thuyết,
cần ảnh lỗi để đối chiếu. Box nhỏ khiến lệch vài pixel có thể giảm IoU mạnh.
Ảnh minh họa dùng GT xanh và dự đoán đỏ, cả hai đều đổi norm1000 sang pixel trước khi vẽ.

Ở E3 (F1 tổng cao nhất trong các lượt đã chấm), lớp yếu nhất là pothole: F1=0.2477, TP=27, FP=70, FN=94. Cần phân biệt lỗi này với lỗi zero-shot của E0; phân bố phiên và box nhỏ có thể ảnh hưởng, nhưng chưa có kiểm định nhân quả cho các giả thuyết đó.

![vlcsnap-2025-02-18-23h08m22s176: correct, best IoU=0.9306](report/figures/e1-01-correct.png)
![vlcsnap-2025-02-18-23h15m04s500: wrong_box, best IoU=0.4996](report/figures/e1-02-wrong_box.png)
![vlcsnap-2025-02-18-23h08m20s084: invalid_json, best IoU=0.0](report/figures/e1-03-invalid_json.png)
![vlcsnap-2025-02-18-23h27m49s136: correct, best IoU=0.9281](report/figures/e1-04-correct.png)
![vlcsnap-2025-02-18-23h30m21s944: wrong_box, best IoU=0.4929](report/figures/e1-05-wrong_box.png)
![vlcsnap-2025-02-18-23h10m39s932: invalid_json, best IoU=0.0](report/figures/e1-06-invalid_json.png)
![vlcsnap-2025-02-18-23h27m05s418: correct, best IoU=0.8755](report/figures/e2-01-correct.png)
![vlcsnap-2025-02-19-17h05m39s299: wrong_box, best IoU=0.4981](report/figures/e2-02-wrong_box.png)
![vlcsnap-2025-02-18-23h16m39s382: invalid_json, best IoU=0.0](report/figures/e2-03-invalid_json.png)
![vlcsnap-2025-02-19-17h14m36s486: correct, best IoU=0.8717](report/figures/e2-04-correct.png)
![vlcsnap-2025-02-18-23h15m06s511: wrong_box, best IoU=0.495](report/figures/e2-05-wrong_box.png)
![vlcsnap-2025-02-19-17h03m29s526: invalid_json, best IoU=0.0](report/figures/e2-06-invalid_json.png)
![vlcsnap-2025-02-18-23h28m02s296: correct, best IoU=0.9472](report/figures/e3-01-correct.png)
![vlcsnap-2025-02-18-23h27m33s135: wrong_box, best IoU=0.4977](report/figures/e3-02-wrong_box.png)
![vlcsnap-2025-02-19-17h03m28s017: correct, best IoU=0.9366](report/figures/e3-03-correct.png)
![vlcsnap-2025-02-18-23h14m34s881: wrong_box, best IoU=0.4875](report/figures/e3-04-wrong_box.png)
![vlcsnap-2025-02-19-17h04m41s282: correct, best IoU=0.9235](report/figures/e3-05-correct.png)
![vlcsnap-2025-02-19-17h09m25s147: wrong_box, best IoU=0.4863](report/figures/e3-06-wrong_box.png)
![vlcsnap-2025-02-19-17h04m44s457: correct, best IoU=0.9089](report/figures/e3_d10-01-correct.png)
![vlcsnap-2025-02-19-17h02m57s019: wrong_box, best IoU=0.4963](report/figures/e3_d10-02-wrong_box.png)
![vlcsnap-00200: correct, best IoU=0.8907](report/figures/e3_d10-03-correct.png)
![vlcsnap-2025-02-18-23h08m18s609: wrong_box, best IoU=0.4958](report/figures/e3_d10-04-wrong_box.png)
![vlcsnap-2025-02-18-23h18m23s887: correct, best IoU=0.882](report/figures/e3_d10-05-correct.png)
![vlcsnap-2025-02-18-23h23m55s225: wrong_box, best IoU=0.4949](report/figures/e3_d10-06-wrong_box.png)

### Câu 10 — E4

Biến chọn là `data_size`: 10% so với 100% ảnh train, các siêu tham số khác giữ nguyên,
kể cả 300 optimizer steps và effective batch=8. Như vậy mức 10% lặp lại dữ liệu nhiều hơn;
đây là hiệu ứng giảm độ đa dạng ở cùng ngân sách bước, không phải cùng số epoch.
Mốc 100% dùng lại đúng dự đoán của setup gốc theo GUIDE; chỉ train thêm mức 10%.
Setup gốc chọn theo GUIDE: E3, có F1 cao nhất trong ba setup cố định.
Mức 10% dùng 142 ảnh đầu theo thứ tự split. Phân bố phiên/lớp có thể khác toàn bộ train,
nên tác động đo được gồm giảm độ đa dạng và thay đổi thành phần mẫu, không chỉ số lượng.
e4_data_size_0.1: F1=0.2453, CI95=[0.2107969151670951, 0.28061224489795916]. e4_data_size_1.0: F1=0.4078, CI95=[0.36910569105691055, 0.44912280701754387].
Lượt E4 data_size=10% hoàn tất lúc 13:06 ngày 14/10/2026 (giờ Việt Nam), sau hạn nộp 12:00; kết quả hoàn tất không được nộp lại lên LMS.

Chênh F1 (100% trừ 10%)=+0.1625. Hai CI95 không chồng nhau; khác biệt rõ trên test này, nhưng vẫn cần kiểm định ghép cặp và thêm phiên quay để kết luận rộng hơn.
Đọc kết quả kèm CI95 và giới hạn test nhỏ; không lựa chọn lại siêu tham số dựa vào test.

## Nhìn lại

Các điểm dễ sai là mask loss, loại visual khỏi NF4, lưu/nạp projector và quy đổi tọa độ khi vẽ.
Đã sửa lỗi ảnh minh họa dùng tọa độ norm1000 trực tiếp như pixel và lỗi tiếp tục cache E0 ghi đè phần đã chạy.
Không sửa `scripts/` và `tests/` của đề. Phần E5 có hàm reward IoU Python thuần trong
`student_reward.py` và 6 test (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng, box khổng lồ).
Chưa chạy GRPO; chưa có kết quả RL hay kết luận reward hacking từ thực nghiệm.
Phần thiếu bằng chứng vẫn phải được hoàn tất trước khi gọi bài là hoàn chỉnh.
