# CP3b — E0 zero-shot trên toàn bộ test

Mục tiêu của E0: đo model gốc trước khi huấn luyện, làm mốc so sánh cho E1–E3. Đây **không phải** kết quả finetune.

## Kết quả chính thức

Chạy trên Colab Tesla T4, `Qwen/Qwen3-VL-4B-Instruct` NF4 4-bit, không adapter, không cập nhật trọng số. Trọng tài: `scripts/score.py` **không sửa**.

| Chỉ số | Giá trị |
|---|---:|
| Ảnh test | 311 |
| `missing_predictions` | 0 |
| `parse_valid_rate` | 0,537 |
| F1 (micro) | **0,065** |
| CI95 | 0,044–0,088 |
| mean IoU | 0,625 |
| FP/ảnh | 1,19 |
| Precision / Recall | 0,080 / 0,055 |
| TP / FP / FN | 32 / 370 / 551 |

| Lớp | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| pothole | 0,200 | 0,066 | 0,099 | 8 | 32 | 113 |
| crack | 0,005 | 0,003 | 0,004 | 1 | 197 | 298 |
| manhole | 0,140 | 0,141 | 0,141 | 23 | 141 | 140 |

Thời gian suy luận: 1937,6 giây cho 311 ảnh. VRAM đỉnh đo bằng torch CUDA allocation: 4,1448 GB — **suy luận, không phải VRAM huấn luyện**. Tham số huấn luyện: 0.

## Hai lỗi phổ biến nhất của model gốc

Phân loại tự động trên 311 output thật (`03-e0-errors.json`):

| Loại lỗi | Số ảnh | Tỉ lệ |
|---|---:|---:|
| JSON hỏng do bị cắt giữa chừng | 144 | 0,463 |
| Model trả `[]` (không tìm thấy gì) | 44 | 0,142 |
| Box lặp lại y hệt trong cùng một output | 49 | 0,158 |
| Box phủ gần hết khung | 45 | 0,145 |
| Output hợp lệ và hợp lý | 45 | 0,145 |

1. **JSON bị cắt / không hợp lệ (46,3%).** Model trả một box crack rồi lặp lại box đó nhiều lần và hết `max_new_tokens=256` trước khi đóng `]`. Output bị hỏng cấu trúc chứ không phải chỉ sai vị trí.
2. **Không dự đoán được gì (14,2% trả `[]`)**, cộng với **box bịa phủ gần hết khung (14,5%)**. Hai kiểu này đối lập nhau nhưng cùng cho F1 thấp: bỏ sót hoặc đoán quá rộng.

Ba lớp đều yếu: `crack` gần như bằng 0 (TP = 1). Model gốc gọi tên "crack" nhiều nhưng IoU không đạt ngưỡng 0,5. `mean_iou = 0,625` chỉ tính trên các cặp ghép được, nên **không** mâu thuẫn với F1 thấp.

## Ý nghĩa và ranh giới

- E0 rất yếu, nên mốc so sánh rất thấp. Rubric yêu cầu F1 của E1 cao hơn E0 **ít nhất 0,10** — mục tiêu này khả thi vì E0 chỉ 0,065.
- F1 0,065 với CI95 0,044–0,088: khoảng tin cậy hẹp vì micro-F1 gộp 311 ảnh. Các khoảng này **không** thay thế CI95 từng lớp.
- Chưa kết luận finetune có giúp hay không. E0 mới là mốc.
- `parse_valid_rate` 0,537 thấp hơn nhiều so với yêu cầu 0,95 của E1; nếu E1 không cải thiện phần này thì chưa đạt.
- Chỉ số này **không** dùng cho `meta` (E0 không huấn luyện nên không có `trainable_params` hợp lệ theo nghĩa rubric).

## Bằng chứng

| File | Nội dung |
|---|---|
| `report/evidence/e0.json` | Output chính thức của trọng tài |
| `03-e0-score.txt` | Lệnh + output nguyên văn |
| `03-e0-errors.json` | Phân loại lỗi trên 311 ảnh |
| `03-e0-raw-examples.json` | 5 output thô thật của model |
| `03-e0-run.json` | Thời gian và VRAM đỉnh |
| `03-e0-test.log` | Log chạy đủ 311 ảnh |

`work/e0.jsonl` (311 dòng dự đoán thô) bị `.gitignore` của đề loại trừ nên không commit; bản sao nằm ở `~/.hermes/drafts/day10/e0-final/work/e0.jsonl`. `report/evidence/e0.json` không nằm trong `.gitignore`.

## Điểm cần chú ý khi nộp

`check_submission.py` trên máy local sẽ báo WARN `khớp data/splits/meta.json` vì `data/` bị gitignore — không có `meta.json` để đối chiếu. Trên Colab (có dataset) mục này PASS. Đây là giới hạn của môi trường, không phải lỗi bằng chứng. `dataset_version` của E0 là `d10-21e4070301`.

## Ranh giới checkpoint

E0 đã xong và có bằng chứng thật. Chưa làm E1–E4, chưa có `REPORT.md` hoàn chỉnh, chưa đủ 5 commit, chưa push. Tiếp theo là CP4: dựng dữ liệu huấn luyện, mask loss và kiểm tra token được học — bước này quyết định E1 có học được hay không.
