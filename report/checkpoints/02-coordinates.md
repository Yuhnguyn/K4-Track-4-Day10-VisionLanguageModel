# CP2 — Nhãn JSON 0–1000 và prompt grounding

## Đã triển khai

- `student_labels.py`: `yolo_to_box`, `objects_to_answer`, `box_to_pixels`, `PROMPT` và kiểm chứng trên dữ liệu thật.
- Cell bước 2 trong `lab_day10.ipynb` gọi cùng các hàm; không sửa `scripts/`/`tests/` của đề.
- `student_checks/test_labels.py`: kiểm thử riêng cho chuyển thang, kẹp biên, thứ tự, JSON, dữ liệu hỏng và sai thang trong trọng tài.

YOLO là `[xc,yc,w,h]` chuẩn hóa. JSON dùng hai góc: `[(xc-w/2)*1000,(yc-h/2)*1000,(xc+w/2)*1000,(yc+h/2)*1000]`, làm tròn bằng Python `round` và kẹp vào [0,1000]. Các object được sắp theo `(y1,x1)`, trả JSON compact; không có object thì trả `[]`.

Prompt yêu cầu JSON list duy nhất, ba lớp chính xác, `bbox_2d` nguyên trên thang 0–1000, không dùng pixel và không thêm markdown. Chưa đo hiệu quả của prompt bằng model.

## Kiểm chứng thực chạy trên Colab Tesla T4

`dataset_version`: `d10-21e4070301`.

| Split | Ảnh được duyệt | Object chuyển được | Ảnh cần quyết định xử lý trước train |
|---|---:|---:|---:|
| train | 1422 | 3514 | 1 |
| val | 276 | 639 | 0 |
| test | 311 | 583 | 0 |

- Đáp án của mọi ảnh không có object bị từ chối đều được `scripts.score.parse_output` chấp nhận.
- Kiểm tra từng góc của tất cả object chuyển được: sai số quy ngược pixel không vượt nửa đơn vị làm tròn quy đổi sang kích thước ảnh.
- Sai số lớn nhất so với **box gốc đã kẹp biên**: khoảng **0,32 pixel**. So sánh với box đã kẹp biên để không đánh đồng phần tọa độ nguồn ngoài ảnh với lỗi công thức đổi thang.
- `02-coordinate-roundtrip.png`: bên trái box YOLO gốc màu xanh; bên phải JSON 0–1000 quy ngược về pixel màu vàng. Đây là kiểm chứng nhãn, không phải dự đoán model.
- Chạy thực các source cell CP1 và CP2 của notebook trong một subprocess trên Colab; lưu output ở `02-notebook-checks.txt`. Đồng thời kiểm tra cú pháp tất cả code cell bằng IPython transformer + AST. Không chạy các cell model chưa triển khai.
- Pytest Colab với `D10_RAW` trỏ vào dataset thật: **93 passed**, không skip (`02-tests.txt`).
- Kiểm thử CPU bản workspace local: **92 passed, 1 skipped**; chỉ test dữ liệu thật bị skip vì dataset thô nằm trên Colab. Không nạp model trên Mac.

Ví dụ kiểm thử hình học: YOLO `[0.5,0.5,0.2,0.2]` → JSON corners `[400,400,600,600]` → pixel `[256,144,384,216]` ở ảnh 640×360. Kiểm thử trọng tài trên **fixture tổng hợp này** cho F1 = 1 khi chấm đúng `norm1000`, F1 = 0 khi nhầm `pixel` dù JSON vẫn parse hợp lệ. Đây không phải F1 của E0 hay thí nghiệm model.

## Nhãn lỗi không được sửa ngầm

Nhãn `crack` ở train `vlcsnap-00058`, object index 1 có YOLO `[0.639062,0.786111,0.003125,0.0]`. Hàm strict từ chối box diện tích bằng 0, hoặc box bị xẹp sau làm tròn/kẹp biên, thay vì bịa chiều cao hay phát ra đáp án không hợp lệ.

Chỉ audit và ghi lỗi; **chưa xoá object/ảnh khỏi dataset, chưa tạo training set đã lọc**. Trước CP4 dựng dữ liệu huấn luyện phải quyết định công khai cách xử lý ảnh train này. Dữ liệu và split val/test, trọng tài, counts chính thức giữ nguyên. Không nhầm số object chuyển được với số box nguồn trong CP1.

## Ranh giới checkpoint

CP2 đạt phần chuyển tọa độ, prompt và kiểm chứng. Chưa E0, chưa train. Trước E0 cần hoàn tất các thư viện model theo `requirements-colab.txt`, kiểm tra import/quantization và chuẩn bị lưu bền vững cho các lượt huấn luyện. Notebook setup vẫn dùng Secrets/`--kaggle`, còn đường tải đã chạy là public ZIP; chưa xác minh Restart & Run All từ runtime sạch.
