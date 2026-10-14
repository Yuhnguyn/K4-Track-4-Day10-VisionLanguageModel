# Báo cáo Lab Day 10 — Qwen3-VL-4B QLoRA

Họ tên: Nguyễn Huy
`dataset_version`: `d10-21e4070301`

Trạng thái: Chưa hoàn tất; còn thiếu bằng chứng E1, E2, E3, E4 (chưa đủ cả 10% và 100%).
Mọi số trong bảng lấy từ scorer của đề. Dấu — nghĩa là chưa đo, không phải 0.
E0 là lượt Colab T4 đã có trong lịch sử repo; E1–E4 chạy qua Kaggle CLI.

E1 đã hoàn tất train: 45.5127 phút, VRAM=7.1231 GB. E1 | F1: 0.2596 | CI95: [0.2278, 0.2978] | valid: 0.9646 | FP/img: 1.466

E2 đã hoàn tất train: 46.2135 phút, VRAM=8.9413 GB. E2 | F1: 0.3071 | CI95: [0.2697, 0.3485] | valid: 0.9936 | FP/img: 1.103

E3 đã hoàn tất train: 64.8521 phút, VRAM=9.1324 GB. Chưa có điểm test trong log.

Các số trên lấy từ stdout, có làm tròn; đang chờ file scorer JSON đầy đủ. Checkpoint tiến độ không thay thế report/evidence/*.json.


## Bảng kết quả

| Mã | Tham số huấn luyện | VRAM GB | Phút train | F1 | CI95 | FP/ảnh | pothole | crack | manhole |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | 0 | 4.1448 | 0.0000 | 0.0650 | 0.0441–0.0877 | 1.1897 | 0.0994 | 0.0040 | 0.1407 |
| E1 | — | — | — | Chưa có kết quả | — | — | — | — | — |
| E2 | — | — | — | Chưa có kết quả | — | — | — | — | — |
| E3 | — | — | — | Chưa có kết quả | — | — | — | — | — |

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
Chưa có phép chấm lại toàn bộ dự đoán theo thang sai; không ghi một số F1 giả cho kiểm chứng này.

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

Chưa có kết quả test E1–E3; chưa kết luận setup nào tốt nhất.

Đánh giá lợi ích phải đặt F1 và F1 từng lớp cạnh thời gian/VRAM.
E2 thêm khoảng 109M tham số projector; E3 cần gradient qua ViT nên có thể tốn VRAM/thời gian hơn.
Không đồng nhất dự đoán lý thuyết về chi phí với số đo. Các số thực tế nằm trong bảng.

### Câu 8 — Khoảng tin cậy

Độ rộng CI95: E0: 0.0436.
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



Ảnh dự đoán sẽ được tạo từ output model thực tế khi lượt GPU hoàn tất.

### Câu 10 — E4

Biến chọn là `data_size`: 10% so với 100% ảnh train, các siêu tham số khác giữ nguyên,
kể cả 300 optimizer steps và effective batch=8. Như vậy mức 10% lặp lại dữ liệu nhiều hơn;
đây là hiệu ứng giảm độ đa dạng ở cùng ngân sách bước, không phải cùng số epoch.
Mốc 100% dùng lại đúng dự đoán của setup gốc theo GUIDE; chỉ train thêm mức 10%.
Đang chờ đủ E1–E3 để chọn setup gốc theo GUIDE.
Mức 10% dùng 142 ảnh đầu theo thứ tự split. Phân bố phiên/lớp có thể khác toàn bộ train,
nên tác động đo được gồm giảm độ đa dạng và thay đổi thành phần mẫu, không chỉ số lượng.
Chưa có bằng chứng E4; chưa kết luận tác động của kích thước dữ liệu.
Mức 10% chưa hoàn tất; chưa thể kết luận ablation.

Đọc kết quả kèm CI95 và giới hạn test nhỏ; không lựa chọn lại siêu tham số dựa vào test.

## Nhìn lại

Các điểm dễ sai là mask loss, loại visual khỏi NF4, lưu/nạp projector và quy đổi tọa độ khi vẽ.
Đã sửa lỗi ảnh minh họa dùng tọa độ norm1000 trực tiếp như pixel và lỗi tiếp tục cache E0 ghi đè phần đã chạy.
Không sửa `scripts/` và `tests/` của đề. Phần E5 có hàm reward IoU Python thuần trong
`student_reward.py` và 6 test (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng, box khổng lồ).
Chưa chạy GRPO; chưa có kết quả RL hay kết luận reward hacking từ thực nghiệm.
Phần thiếu bằng chứng vẫn phải được hoàn tất trước khi gọi bài là hoàn chỉnh.
