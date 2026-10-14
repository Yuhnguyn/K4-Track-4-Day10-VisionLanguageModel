# Chạy bằng Kaggle CLI

Không lưu token trong repo/notebook. Đặt `KAGGLE_API_TOKEN` trong môi trường làm việc,
hoặc dùng cấu hình Kaggle riêng bên ngoài repo.

```text
pip install kaggle
python prepare_kaggle.py --username YOUR_ACCOUNT --experiment e1
kaggle kernels push -p work/kaggle-e1-1 --accelerator NvidiaTeslaT4
kaggle kernels status YOUR_ACCOUNT/day10-vlm-e1-100
kaggle kernels logs YOUR_ACCOUNT/day10-vlm-e1-100 --follow
kaggle kernels output YOUR_ACCOUNT/day10-vlm-e1-100 -p work/kaggle-output
```

Đổi `e1` thành `e2` hoặc `e3` để chạy các setup tương ứng. Gói notebook chứa đúng mã
local tại thời điểm đóng gói; không phải đợi GitHub cập nhật để có bản sửa.
Dataset được gắn trực tiếp vào notebook qua metadata, chia bằng công cụ đề nguyên bản.
Các file kết quả lưu dưới `report/`, dự đoán thô và adapter/projector dưới `work/`.

`--fraction 0.1` chạy ablation giảm dữ liệu. `--with-e4` dành cho E3 trên T4 x2:
E3 dùng GPU0, E1 với 10% dữ liệu dùng GPU1, tiến trình và optimizer độc lập.
Mốc 100% của E4 dùng dự đoán thật của E1, không tạo dự đoán từ ground truth.

## Thu kết quả và kiểm tra

Sau khi lấy predictions và run.json về máy:

```text
python scripts/score.py --pred work/e1.jsonl --name E1 --out report/evidence/e1.json --meta trainable_params=NUMBER --meta peak_vram_gb=NUMBER --meta train_minutes=NUMBER
python student_figures.py --pred work/e1.jsonl --prefix e1
python update_report.py
python scripts/check_submission.py
```

Thay NUMBER bằng số đo trong run.json. Không đặt số tự ước lượng vào bằng chứng.
`update_report.py` chỉ lấy file scorer đã có; setup chưa có bằng chứng được ghi rõ còn thiếu.

`kaggle_jupyter.py` và `collect_interactive.py` dùng API Jupyter của phiên tương tác
được chủ tài khoản cấp quyền, cho phép lấy file từng phần khi notebook còn chạy.
URL kết nối tạm thời chỉ lưu trong thư mục `work/` bị Git bỏ qua.

## Kiểm thử sau khi điền báo cáo

```text
python -m pytest -q tests student_checks -k "not report_template_has_student_markers_and_ten_questions"
```

Test loại trừ trên là test **mẫu báo cáo của đề**, yêu cầu REPORT.md phải còn chuỗi
`<điền`, ngược với yêu cầu bài nộp phải điền hết. Giữ nguyên test này và mọi công cụ
trong scripts/tests; không sửa chúng để làm xanh kết quả.

Bonus E5: chỉ có hàm reward và các test Python thuần. Chưa chạy GRPO.
