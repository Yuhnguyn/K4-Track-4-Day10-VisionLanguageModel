# CP3a — Môi trường model, chính sách nhãn và smoke E0

## Chính sách đã được người học chọn

Chỉ bỏ box nguồn có diện tích bằng 0 khỏi đáp án **train**, giữ nguyên ảnh và các object còn lại. Không sửa file nguồn, split hoặc tập val/test. `training_target` trả cả đáp án và danh sách annotation bỏ để tầng huấn luyện lưu audit, không lọc im lặng. Lỗi khác, gồm box dương bị xẹp sau làm tròn, vẫn gây lỗi thay vì được bỏ ngầm.

Kiểm chứng trên toàn bộ train: giữ 1422 ảnh; đúng một object `crack` ở `vlcsnap-00058`, index 1 bị loại khỏi target train. Các đáp án train sau áp dụng chính sách đều được parser của đề chấp nhận. Bằng chứng: `03-training-label-policy.json`.

## Môi trường thực chạy trên Colab Tesla T4

| Thư viện | Phiên bản được cài và kiểm tra |
|---|---|
| transformers | 5.19.0 |
| peft | 0.21.2 |
| trl | 1.9.2 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |

Import Qwen3VL, AutoProcessor, PEFT và TRL đạt; đã chạy forward NF4 trên CUDA với tensor thật. Output/log và exit code ở `03-model-environment.txt`. Không cài bộ thư viện model hoặc tải trọng số lên Mac.

## Lỗi cấu hình đã phát hiện và khắc phục

Trong `transformers==5.19.0`, `should_convert_module` khớp prefix/regex hoặc suffix, không khớp tên module cha ở giữa đường dẫn. `visual` không loại được các lớp con `model.visual.*`. Đã dùng `llm_int8_skip_modules=["model.visual", "visual", "lm_head"]` và kiểm tra cây model sau nạp: phải có NF4 LLM, không có NF4 trong visual/lm_head. Không sửa pseudo code hay công cụ của đề.

Lần smoke thành công kiểm chứng **252 module NF4**, phần visual/lm_head không có Linear4bit. Mỗi ảnh có **220 vision token**. Model giữ nguyên `Qwen/Qwen3-VL-4B-Instruct`; không dùng adapter, không cập nhật trọng số.

## Suy luận thử thật trên bốn ảnh val

- Batch 4, greedy (`do_sample=False`), `max_new_tokens=256`, prompt CP2.
- Chạy đủ 4 ảnh val trong 30,2294 giây suy luận (không gồm tải/nạp model).
- VRAM đỉnh đo bằng torch CUDA allocation: 4,144761344 GB (đơn vị thập phân). Đây là suy luận, không phải VRAM train.
- Parser đọc được 2/4 output. Hai output lặp cùng một box crack nhiều lần rồi bị cắt trước khi đóng JSON ở giới hạn token. Giữ nguyên output, không sửa tay hoặc dùng nhãn để tạo dự đoán.
- Raw output ở `03-e0-val-smoke-outputs.jsonl`, tài nguyên ở `03-e0-val-smoke-run.json`, log ở `03-e0-val-smoke.log`.
- Smoke val không phải bằng chứng `e0.json` trên toàn bộ test; chưa kết luận F1 model từ bốn ảnh này.

`student_inference.py` lưu raw JSONL theo batch, kiểm tra signature của cache, ngăn duplicate/unexpected image_id và không cho truncate test bằng `--limit`. Không nạp model lúc import module. Cell E0 notebook gọi cùng CLI trong subprocess để tránh import cũ trong kernel.

## Kiểm thử và ranh giới

- Colab có dataset thật: `python -m pytest -q tests student_checks` → **102 passed**, không skip (`03-tests.txt`).
- Bản local CPU cũng được kiểm thử; test dataset thật có thể skip vì dataset ở Colab, không tải toàn bộ data về Mac.
- Bộ test bổ sung ở `student_checks/`; `scripts/` và `tests/` của đề giữ nguyên.
- Checkpoint này xác nhận môi trường và smoke E0, không xác nhận hoàn thành E0 test hay E1–E4. Kết quả E0 test sẽ được ghi ở checkpoint sau khi trọng tài chạy và xác minh đủ 311 ảnh.
- Drive chưa mount; trước các lượt **huấn luyện** phải có lưu adapter/projector bền vững. E0 không tạo adapter; raw predictions được sao lưu về drafts local bằng đọc Contents API, không xếp hàng kernel.
- Chưa xác minh Restart & Run All từ runtime sạch; notebook setup/data vẫn cần đúng repo và đường Kaggle Secrets hoặc ZIP.
