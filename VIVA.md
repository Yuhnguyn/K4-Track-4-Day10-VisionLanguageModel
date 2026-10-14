# Ôn vấn đáp 5 phút

## Tóm tắt bài trong 30 giây

Ảnh dashcam đi qua ViT để lấy đặc trưng, projector đưa đặc trưng về không gian của LLM,
rồi Qwen3-VL sinh JSON gồm lớp và bounding box. Tôi dùng NF4 cho LLM, giữ visual/lm_head
ở float, dùng LoRA để giảm số tham số học. So sánh E1 học LoRA LLM, E2 thêm projector,
E3 thêm LoRA ViT; chấm cùng tập test theo phiên quay bằng công cụ của đề.

## Những câu cần hiểu thật

1. **Vì sao không chia ngẫu nhiên?** Frame cùng video gần giống nhau. Chia ngẫu nhiên có
   thể cho model thấy gần như cùng cảnh ở train và test. Split hiện có 13/2/3 phiên
   train/val/test, giao phiên bằng 0.

2. **Tại sao thang 0–1000?** Tọa độ tương đối tách khỏi kích thước ảnh. `[400,400,600,600]`
   trên ảnh 640×360 là `[256,144,384,216]` pixel. Khi vẽ phải nhân riêng theo chiều rộng/cao.

3. **Mask loss là gì?** Đặt label=-100 cho phần user/prompt và token ảnh, chỉ học đáp án
   assistant cùng token kết thúc. Mẫu đã kiểm tra có 639 token: mask 389, học 250;
   220 token ảnh nằm trong phần bị mask.

4. **QLoRA khác LoRA?** LoRA học ma trận phụ hạng thấp thay vì toàn bộ trọng số gốc;
   QLoRA thêm việc lưu model nền ở dạng lượng tử hóa. Trong lab, các lớp LLM được NF4,
   visual không NF4 để projector vẫn có thể huấn luyện đầy đủ.

5. **E1 học bao nhiêu?** Log thực tế ghi 33.030.144 tham số, 252 module LoRA.
   Projector có 109.105.152 tham số, nên E2 có 142.135.296 tham số trainable.
   E3 thêm LoRA cho 96 module ViT, tổng 348 module LoRA.

6. **Tại sao phải lưu projector riêng?** `save_pretrained` của PEFT lưu adapter LoRA.
   Nếu E2/E3 không lưu/nạp projector đã học, suy luận sẽ dùng projector gốc và không
   phản ánh đúng thí nghiệm. Bài lưu riêng `projector.pt` và kiểm tra đủ tensor khi nạp.

7. **Một bước train dùng bao nhiêu ảnh?** Batch 2, tích lũy gradient 4 lần, effective batch=8.
   300 optimizer steps tương ứng 2400 lượt ảnh, có lặp lại do train chỉ có 1422 ảnh.
   Batch suy luận 12 phục vụ tốc độ; khi hết bộ nhớ tự giảm batch và thử lại đúng ảnh đó.

8. **F1 đo gì?** Ghép box cùng lớp với IoU≥0,5. Precision phạt box bịa; recall phạt bỏ sót;
   F1 cân bằng hai đại lượng. Box đúng vị trí nhưng sai lớp vẫn không được tính TP.

9. **CI95 có nghĩa gì ở đây?** Khoảng bootstrap của F1 trên mẫu ảnh test. Test chỉ có
   311 ảnh thuộc 3 phiên, nên các ảnh có tương quan; bootstrap theo ảnh có thể đánh giá
   thấp bất định so với bootstrap theo phiên. CI chồng nhau không phải chứng minh hai setup ngang nhau.

10. **Có thể nói crack kém chỉ vì ít dữ liệu không?** Không. Train có 1750 box crack,
    nhiều hơn hai lớp còn lại. Crack nhỏ/mảnh, hình dạng phức tạp và dễ lẫn với bóng/vạch đường
    là các giả thuyết hợp lý, nhưng cần đối chiếu ảnh lỗi thực tế.

## Số liệu để đối chiếu

E0 đã có file scorer: F1=0,06497, JSON hợp lệ=53,70%, FP/ảnh=1,1897.
Lượt E1 Kaggle version 4 đã log F1=0,2596, JSON hợp lệ=96,46%, train=45,51 phút,
VRAM train=7,1231 GB. Đây là số làm tròn từ log; đối chiếu REPORT.md và file scorer
mới nhất trước khi dùng số của lượt chạy lại.

Không nói “đã chạy xong” cho setup chưa có bằng chứng. Không dùng kết quả test để chọn
điều chỉnh prompt/siêu tham số rồi vẫn gọi đó là đánh giá độc lập trên test.
