# Checklist nộp bài — Lab Day 10

Họ tên: Nguyễn Huy  
Dataset: `d10-21e4070301` · split đánh giá: test (311 ảnh)

## Hồ sơ và kiểm tra

- [x] Báo cáo `REPORT.md` có kết quả đo, diễn giải, hình minh họa và trạng thái thí nghiệm.
- [x] Có ít nhất 6 hình trong `report/figures/`.
- [x] Các bằng chứng E0–E3 và E4 mức 100% được lưu trong `report/evidence/`.
- [x] Kết quả E2 được đối chiếu lại từ raw predictions; cả 311 ảnh khớp trong sai số số thực.
- [x] `scripts/check_submission.py`: 0 FAIL, 0 WARN (kết quả lưu tại `report/checkpoints/12-final-submission-check.txt`).
- [x] `git diff --check` sạch; đã kiểm tra danh sách file bị cấm và số commit.
- [x] Các file gốc trong `configs/`, `scripts/`, `tests/` được giữ nguyên.

## Trạng thái thí nghiệm

| Thí nghiệm | Trạng thái | Micro F1 | Ghi chú |
|---|---|---:|---|
| E0 | Hoàn tất | 0.0650 | Zero-shot; 311 ảnh |
| E1 | Hoàn tất | 0.2738 | LoRA LLM; 311 ảnh |
| E2 | Hoàn tất | 0.3158 | LoRA LLM + projector; 311 ảnh |
| E3 | Hoàn tất | 0.4078 | LoRA LLM + projector + LoRA ViT; 311 ảnh |
| E4 data_size=100% | Hoàn tất, tái sử dụng raw predictions E3 theo GUIDE | 0.4078 | Đây là mốc đối chứng 100% |
| E4 data_size=10% | Đang chạy, chưa có bằng chứng scorer | — | Chưa thể kết luận so sánh hai mức |

**Kết luận tình trạng:** Hồ sơ nộp và các thí nghiệm E0–E3 cùng mốc E4 100% đã sẵn sàng. Ablation E4 hai mức chưa hoàn tất vì kết quả 10% còn đang chạy; không suy diễn hoặc ghi số cho mức này.
