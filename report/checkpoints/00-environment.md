# CP0 — Môi trường và baseline trước khi làm bài

## Phân chia công việc

- Google Colab T4: nạp model, huấn luyện, suy luận và chấm kết quả.
- Máy local: sửa mã, lưu bản sao bằng chứng và commit Git. Không huấn luyện model trên Mac.
- Giữ nguyên `scripts/` và `tests/` của đề.

## Kết quả thực chạy trên Colab

| Kiểm tra | Kết quả |
|---|---|
| GPU | Tesla T4 |
| CUDA | `torch.cuda.is_available() == True` |
| VRAM thiết bị | 15,637,086,208 bytes; không phải VRAM đỉnh khi train |
| PyTorch | `2.11.0+cu130` |
| Python | `3.13.15` |
| `python -m pytest -q` | 68 passed, 1 skipped |
| `python scripts/check_submission.py` | 7 FAIL, 1 WARN trước khi triển khai |

Output gốc: `00-pytest-BASELINE.txt` và `00-submission-BASELINE.txt` trong cùng thư mục.

Test bị skip cần dữ liệu thật và biến `D10_RAW`; dữ liệu chưa được tải ở checkpoint này. Các FAIL của submission là thiếu E0–E4, báo cáo chưa điền và lịch sử mới có một commit. WARN là chưa có `data/splits/meta.json`. PASS về dataset_version khi chưa có bằng chứng không chứng minh dữ liệu đã được xác minh.

Chưa tải Qwen3-VL, chưa chạy E0, chưa huấn luyện và chưa có số đo F1/VRAM đỉnh/thời gian train. Chưa kiểm chứng các thư viện model trong `requirements-colab.txt`; đó là bước chuẩn bị cho E0.

## Nhịp học và commit dự kiến

Mỗi checkpoint: hiểu mục tiêu → viết phần nhỏ → chạy kiểm tra → đọc kết quả → commit. Không tạo commit rỗng để đủ số lượng.

1. CP1: tải dữ liệu, thống kê split, kiểm tra rò rỉ theo session, vẽ nhãn.
2. CP2: chuyển YOLO sang JSON 0–1000, kiểm chứng box và viết prompt.
3. CP3: E0 zero-shot, lưu output thô và chấm bằng công cụ của đề.
4. CP4: chat template, mask loss, collator và kiểm tra token được học.
5. CP5: E1 LoRA LLM, huấn luyện và chấm kết quả thật.
6. CP6: E2 thêm projector; lưu/nạp cả projector đã cập nhật.
7. CP7: E3 thêm LoRA ViT; so sánh chất lượng và tài nguyên.
8. CP8: E4 đổi đúng một biến, chạy hai giá trị và đọc CI95.
9. CP9: báo cáo, ảnh minh hoạ và kiểm tra bài nộp; E5 chỉ làm nếu chọn bonus.

Colab có thể ngắt phiên hoặc hết quota. Trước mỗi lượt train phải chuẩn bị nơi lưu adapter và projector bền vững; không coi `/content` là bản sao an toàn. Thời gian 20–30 phút mỗi lượt trong README là ước lượng của đề, chưa phải số đo của bài làm.
