# Báo cáo lỗi và lưu ý — Lab Day 10

Tài liệu ghi lại các vấn đề thực tế gặp trong quá trình chạy và cách diễn giải kết quả. Không thay thế hướng dẫn chính thức hoặc bằng chứng scorer.

## Dữ liệu và đánh giá

- **Chia dữ liệu theo session:** cần giữ split theo session để tránh ảnh gần trùng hoặc cùng ngữ cảnh xuất hiện ở cả train và test. Nếu trộn ngẫu nhiên từng ảnh, kết quả có thể bị leakage và không còn phản ánh tổng quát hóa sang session mới.
- **Quy ước tọa độ:** scorer dùng `norm1000`; dự đoán theo pixel mà không đổi tỉ lệ sẽ sai. Trong lần kiểm tra sai scale, F1 quan sát được là 0. Không trộn định dạng pixel và normalized bbox.
- **JSON đầu ra:** JSON lỗi cú pháp, bbox không hợp lệ và các hộp lặp đều ảnh hưởng parse hoặc precision. Kết quả E0 có parse-valid rate 0.537; E1 0.977; E2 0.990; E3 1.000 trên 311 ảnh. Cần xuất một cấu trúc JSON hợp lệ và loại hộp lặp trước khi chấm.
- **Vẽ bbox normalized:** renderer phải đổi tọa độ chuẩn hóa sang kích thước ảnh trước khi vẽ; nếu dùng trực tiếp giá trị normalized làm pixel, hình minh họa sẽ sai vị trí/kích thước.

## Huấn luyện và lưu mô hình

- **Assistant-only loss:** chỉ tính loss trên phần câu trả lời của assistant; không tính prompt hoặc nội dung user như target.
- **NF4 và phần được huấn luyện:** các tham số `lm_head` và vision tower không tự động trở thành tham số trainable chỉ vì base model dùng NF4. Cần kiểm tra chính xác danh sách module được bật gradient.
- **Projector:** projector được giữ ở fp32; phải lưu và nạp lại riêng cùng adapter để bảo toàn trọng số đã học.
- **OOM khi suy luận:** giảm batch hoặc dùng fallback phù hợp khi thiếu VRAM; dọn cache trước lượt tiếp theo. Một lỗi cache append đã được xử lý bằng cách chỉ nối phần token mới, tránh lặp lại cache cũ.
- **Giới hạn batch theo phiên:** trên Kaggle, batch tối đa quan sát được là 2 ở chế độ batch; chế độ tương tác chỉ chạy ổn định với batch 1. Đây là giới hạn môi trường đã gặp, không phải khuyến nghị phổ quát.

## E4 và giới hạn xác nhận

- E4 dùng cùng cấu hình đã chọn cho E3. Chạy subprocess gặp lỗi import; lượt 10% được chuyển sang phiên tương tác được cho phép. Tại thời điểm đóng gói, lượt 10% vẫn đang chạy, chưa có raw predictions hay evidence scorer. Vì vậy báo cáo chỉ có mốc 100% được GUIDE cho phép tái sử dụng từ E3, và không tuyên bố ablation hai mức đã hoàn tất.
- Không có xác nhận chạy `RunAll` trong runtime sạch; không ghi nhận đây là kiểm thử đã thực hiện.
- Một test của template gốc cố ý kiểm tra placeholder chuỗi `<điền>`. Sau khi điền báo cáo, test đó không còn phù hợp với nội dung hoàn chỉnh nên được loại khỏi kiểm tra nộp; các file `tests/` được giữ nguyên, không sửa.
