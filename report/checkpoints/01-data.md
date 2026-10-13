# CP1 — Hiểu dữ liệu và kiểm tra nhãn

## Mục tiêu

Đọc dataset thật, thống kê ảnh/box, xác nhận split không chung phiên quay và vẽ sáu ảnh train với nhãn gốc. Tất cả thao tác dữ liệu và kiểm thử của checkpoint này chạy trên Colab Tesla T4; chưa nạp model.

## Số liệu thực chạy

`dataset_version`: `d10-21e4070301`.

| Split | Ảnh | Phiên quay | Box pothole | Box crack | Box manhole |
|---|---:|---:|---:|---:|---:|
| train | 1422 | 13 | 1074 | 1750 | 691 |
| val | 276 | 2 | 66 | 470 | 103 |
| test | 311 | 3 | 121 | 299 | 163 |

- Đã kiểm tra file ảnh, kích thước khớp metadata và khả năng đọc của 2009 ảnh.
- Không trùng `image_id`; không có `session` nằm ở hai split. Điều này xác nhận hợp đồng chia theo phiên, không chứng minh mọi cảnh tương tự giữa các phiên đã được loại bỏ.
- Counts, mapping phiên và dataset_version tính lại đều khớp `data/splits/meta.json`.
- Có 45 nhãn có góc tính ra vượt biên ảnh (tolerance 1e-6 trên tọa độ chuẩn hóa). Đây là phát hiện hình học, chưa kết luận tất cả là nhãn sai; chưa thay đổi dữ liệu.
- Có một nhãn diện tích bằng 0: split train, ảnh `vlcsnap-00058`, object index 1, lớp `crack`, YOLO `[0.639062, 0.786111, 0.003125, 0.0]`. Giữ nguyên và báo rõ; chưa quyết định lọc ví dụ huấn luyện hay sửa đáp án. Tập val/test không có nhãn diện tích bằng 0 theo kiểm tra này.

## Nhìn ảnh và hiểu nhãn

Ảnh `01-train-labels.png` có sáu ảnh **train**, hiển thị **ground truth**, không phải dự đoán model. Pothole màu đỏ, crack màu vàng, manhole màu xanh. Các ảnh được chọn để minh họa ba lớp, không phải mẫu ngẫu nhiên để suy luận phân bố dataset.

YOLO lưu `[x_center, y_center, width, height]`, chuẩn hóa theo kích thước ảnh. Khi vẽ, đổi về hai góc pixel bằng tâm ± nửa kích thước. Bước chuyển sang JSON 0–1000 cho model thuộc CP2, chưa triển khai ở đây.

Train dùng để học; val dùng để gỡ lỗi/chọn thiết lập; test dùng để đánh giá cuối. Hai frame gần như giống nhau ở train/test có thể khiến điểm cao giả do rò rỉ dữ liệu, tức model gặp lại "câu tủ".

## Code và kiểm tra

- `student_data.py`: đọc split, kiểm tra metadata/session/nhãn, xác minh file ảnh và vẽ minh họa.
- `student_checks/test_data_exploration.py`: kiểm thử riêng, không sửa `tests/` của đề.
- `lab_day10.ipynb`: điền đúng cell khám phá dữ liệu bước 1, các phần model vẫn giữ nguyên khung.
- Colab: `D10_RAW=<dataset thật đã giải nén> python -m pytest -q tests student_checks` → **78 passed**, không skip. Trong đó có kiểm thử dữ liệu thật của đề.
- Tất cả code cell notebook vượt qua kiểm tra cú pháp bằng IPython transformer + AST. Chưa chạy toàn bộ notebook vì các checkpoint sau chưa triển khai.
- Không sửa `scripts/` và `tests/`; không commit dataset thô, ZIP tải về, secrets hay trọng số.

Output gốc ở `01-prepare-data.txt`, `01-explore-data.txt`, `01-tests.txt`; kết quả có cấu trúc ở `01-data-summary.json`.

Dataset nguồn: https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes (MIT theo README của đề). Minh họa được tạo từ ảnh và nhãn gốc, không tính là sáu ảnh dự đoán đúng/sai mà rubric yêu cầu ở bước cuối.

## Ranh giới checkpoint

CP1 đã kiểm tra cấu trúc dữ liệu và minh họa nhãn, không xác nhận chất lượng của mọi annotation bằng mắt. Bài nộp vẫn thiếu E0–E4 và báo cáo. CP2 sẽ xử lý chuyển thang tọa độ, kiểm chứng quy ngược pixel, và thống nhất xử lý nhãn xẹp mà không thay đổi trọng tài/split của đề.
