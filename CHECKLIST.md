# Checklist nộp bài — Lab Day 10

Họ tên: Nguyễn Huy  
Dataset: `d10-21e4070301` · split test: 311 ảnh

## Hồ sơ và kiểm tra

- [x] `REPORT.md` có kết quả scorer, diễn giải, hình minh họa và tình trạng hoàn tất.
- [x] Có ít nhất 6 hình trong `report/figures/`.
- [x] Có evidence E0–E3 và hai mức E4 trong `report/evidence/`.
- [x] Raw predictions E4 10% (`work/e3_d10.jsonl`) có 311 dòng và đã chấm lại bằng `scripts/score.py` gốc.
- [x] `scripts/check_submission.py`: 0 FAIL, 0 WARN; kết quả mới lưu tại `report/checkpoints/13-post-deadline-e4-final-check.txt`.
- [x] `git diff --check` sạch; kiểm tra file cấm và giữ nguyên `configs/`, `scripts/`, `tests/`.

## Thí nghiệm

| Thí nghiệm | Trạng thái | Micro F1 | Ghi chú |
|---|---|---:|---|
| E0 | Hoàn tất | 0.0650 | Zero-shot; 311 ảnh |
| E1 | Hoàn tất | 0.2738 | LoRA LLM; 311 ảnh |
| E2 | Hoàn tất | 0.3158 | LoRA LLM + projector; 311 ảnh |
| E3 | Hoàn tất | 0.4078 | LoRA LLM + projector + LoRA ViT; 311 ảnh |
| E4 data_size=10% | Hoàn tất sau hạn nộp | 0.2453 | 142 train examples, 300 steps; evidence scorer xác minh |
| E4 data_size=100% | Hoàn tất theo GUIDE | 0.4078 | Dùng lại raw predictions E3 làm mốc 100% |

**Trạng thái nộp:** Hai mức ablation đã có kết quả sau khi lượt 10% hoàn tất lúc 13:06 ngày 14/10/2026 (giờ Việt Nam), sau hạn nộp 12:00. Bài đã được nộp LMS trước đó; không ghi nhận lần nộp lại.
