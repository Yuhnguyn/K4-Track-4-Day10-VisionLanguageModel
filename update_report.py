"""Write the report from real scorer evidence; missing runs remain explicitly pending."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def load(relative):
    path = ROOT / relative
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def main():
    evidence = {p.stem: json.loads(p.read_text(encoding='utf-8'))
                for p in (ROOT / 'report/evidence').glob('*.json')}
    e0 = evidence['e0']
    pending = [key.upper() for key in ('e1', 'e2', 'e3') if key not in evidence]
    if not all(k in evidence for k in ('e4_data_size_0.1', 'e4_data_size_1.0')):
        pending.append('E4 (chưa đủ cả 10% và 100%)')
    table = ['| Mã | Tham số huấn luyện | VRAM GB | Phút train | F1 | CI95 | FP/ảnh | pothole | crack | manhole |',
             '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for key in ['e0', 'e1', 'e2', 'e3'] + sorted(k for k in evidence if k.startswith('e4')):
        item = evidence.get(key)
        if not item:
            table.append(f'| {key.upper()} | — | — | — | Chưa có kết quả | — | — | — | — | — |')
            continue
        meta = item.get('meta', {})
        vram = meta.get('peak_vram_gb')
        if key == 'e0':
            vram = load('report/checkpoints/03-e0-run.json').get('peak_vram_gb')
        f = lambda value: f'{value:.4f}' if isinstance(value, (int, float)) else '—'
        ci = '–'.join(f(v) for v in item['f1_ci95'])
        cells = [key.upper(), str(meta.get('trainable_params', 0 if key == 'e0' else '—')),
                 f(vram), f(meta.get('train_minutes', 0 if key == 'e0' else None)),
                 f(item['micro']['f1']), ci, f(item['fp_per_image'])]
        cells += [f(item['per_class'][c]['f1']) for c in ('pothole', 'crack', 'manhole')]
        table.append('| ' + ' | '.join(cells) + ' |')
    trained = [k for k in ('e1', 'e2', 'e3') if k in evidence]
    comparison = []
    for key in trained:
        d = evidence[key]
        comparison.append(f"{key.upper()}: F1={d['micro']['f1']:.4f}, CI95={d['f1_ci95']}, "
                          f"chênh E0={d['micro']['f1'] - e0['micro']['f1']:+.4f}.")
    pairwise = ' '.join(comparison) or 'Chưa có kết quả test E1–E3; chưa kết luận setup nào tốt nhất.'
    value_note = ''
    if len(trained) == 3:
        value_best = max(trained, key=lambda key:
                         (evidence[key]['micro']['f1'] - e0['micro']['f1']) /
                         evidence[key]['meta']['train_minutes'])
        value_note = (f"Nếu ưu tiên mức tăng F1 so với E0 trên mỗi phút train, tôi chọn "
                      f"{value_best.upper()} làm phương án đáng cân nhắc nhất trên T4. "
                      'Đây là tiêu chí chi phí được nêu rõ, không phải kiểm định ý nghĩa thống kê; '
                      'VRAM, F1 từng lớp và CI95 trong bảng vẫn là các giới hạn của lựa chọn.')
    trained_weakness = ''
    if trained:
        best_measured = max(trained, key=lambda key: evidence[key]['micro']['f1'])
        weak = min(('pothole', 'crack', 'manhole'),
                   key=lambda label: evidence[best_measured]['per_class'][label]['f1'])
        values = evidence[best_measured]['per_class'][weak]
        trained_weakness = (f"Ở {best_measured.upper()} (F1 tổng cao nhất trong các lượt đã chấm), "
                            f"lớp yếu nhất là {weak}: F1={values['f1']:.4f}, "
                            f"TP={values['tp']}, FP={values['fp']}, FN={values['fn']}. "
                            'Cần phân biệt lỗi này với lỗi zero-shot của E0; phân bố phiên và box nhỏ '
                            'có thể ảnh hưởng, nhưng chưa có kiểm định nhân quả cho các giả thuyết đó.')
    widths = ' '.join(f"{k.upper()}: {evidence[k]['f1_ci95'][1] - evidence[k]['f1_ci95'][0]:.4f}."
                      for k in ['e0'] + trained)
    e4text = []
    for key in sorted(k for k in evidence if k.startswith('e4')):
        d = evidence[key]
        e4text.append(f"{key}: F1={d['micro']['f1']:.4f}, CI95={d['f1_ci95']}.")
    baseline = load('report/checkpoints/08-e4-choice.json').get('baseline')
    baseline_note = (f"Setup gốc chọn theo GUIDE: {baseline.upper()}, có F1 cao nhất trong ba setup cố định."
                     if baseline else 'Đang chờ đủ E1–E3 để chọn setup gốc theo GUIDE.')
    ablation_conclusion = ''
    if all(k in evidence for k in ('e4_data_size_0.1', 'e4_data_size_1.0')):
        small, full = evidence['e4_data_size_0.1'], evidence['e4_data_size_1.0']
        delta = full['micro']['f1'] - small['micro']['f1']
        overlap = max(small['f1_ci95'][0], full['f1_ci95'][0]) <= min(small['f1_ci95'][1], full['f1_ci95'][1])
        ablation_conclusion = (f"Chênh F1 (100% trừ 10%)={delta:+.4f}. "
                               + ('Hai CI95 chồng nhau; không kết luận chắc về chênh lệch từ hai CI riêng lẻ.'
                                  if overlap else 'Hai CI95 không chồng nhau; khác biệt rõ trên test này, '
                                  'nhưng vẫn cần kiểm định ghép cặp và thêm phiên quay để kết luận rộng hơn.'))
    scale_check = load('report/checkpoints/10-e1-wrong-scale.json')
    scale_note = (f"Kiểm tra bằng dự đoán E1 thực tế: chấm đúng norm1000 cho F1="
                  f"{evidence['e1']['micro']['f1']:.4f}; chấm cùng output nhầm là pixel cho F1="
                  f"{scale_check['micro']['f1']:.4f}. Không huấn luyện hay sửa output trong kiểm tra này."
                  if scale_check and 'e1' in evidence else
                  'Chưa có phép chấm lại toàn bộ dự đoán theo thang sai; không ghi một số F1 giả cho kiểm chứng này.')
    live = load('report/checkpoints/04-e1-kaggle-progress.json') if 'e1' not in evidence else {}
    live_note = ''
    if live:
        live_note = ('\nKết quả E1 đã hiện trong stdout Kaggle version 4: F1=0,2596, '
                     'CI95=[0,2278; 0,2978], JSON hợp lệ=0,9646, FP/ảnh=1,466; '
                     'train=45,5127 phút, VRAM train=7,1231 GB. '
                     'Đây là số được log, có làm tròn; đang chờ lấy file scorer JSON đầy đủ. '
                     'Checkpoint này không thay thế report/evidence/e1.json.\n')
    figures = []
    for summary in (ROOT / 'report/figures').glob('*-summary.json') if (ROOT / 'report/figures').exists() else []:
        for tile in json.loads(summary.read_text(encoding='utf-8'))['tiles']:
            path = Path(tile['file']).as_posix()
            figures.append(f"![{tile['image_id']}: {tile['tag']}, best IoU={tile['best_iou']}]({path})")
    text = f'''# Báo cáo Lab Day 10 — Qwen3-VL-4B QLoRA

Họ tên: Nguyễn Huy
`dataset_version`: `{e0['dataset_version']}`

Trạng thái: {'Đã có bằng chứng E0–E4.' if not pending else 'Chưa hoàn tất; còn thiếu bằng chứng ' + ', '.join(pending) + '.'}
Mọi số trong bảng lấy từ scorer của đề. Dấu — nghĩa là chưa đo, không phải 0.
E0 là lượt Colab T4 đã có trong lịch sử repo; E1–E4 chạy qua Kaggle CLI.
{live_note}

## Bảng kết quả

{chr(10).join(table)}

E0 không huấn luyện; VRAM là suy luận. Log E0 ghi 1937,60 giây suy luận trong session, không gồm tải/nạp model.
Các số VRAM dùng GB thập phân theo `torch.cuda.max_memory_allocated()/1e9`.

### Câu 1 — Kiến trúc và các khối được cập nhật

ViT mã hóa ảnh thành đặc trưng thị giác; merger/projector đổi đặc trưng sang không gian embedding
của LLM; LLM sinh chuỗi JSON tự hồi quy. Theo GUIDE, ViT khoảng 415M tham số, LLM cùng
embedding khoảng 4B; projector nằm trong ViT, không được cộng hai lần vào tổng.
Kiểm tra cây module trên Kaggle xác nhận projector có 109.105.152 tham số.
E1 chỉ cập nhật LoRA LLM: 252 module, 33.030.144 tham số trainable được log thực tế.
E2 thêm toàn bộ projector; E3 thêm LoRA 96 module ViT, tổng 348 module LoRA.
Số phần tử lưu trữ của tensor NF4 không đại diện đúng số tham số model trước lượng tử hóa.

### Câu 2 — E0 và hai lỗi phổ biến

`parse_valid_rate`={e0['parse_valid_rate']:.6f}, tương ứng 167/311 ảnh.
Phân tích output ghi 144 ảnh JSON bị cắt và 49 output hợp lệ có box lặp; các nhóm lỗi có thể giao nhau.
Ví dụ `vlcsnap-00200`: model lặp `{{"label":"crack","bbox_2d":[750,670,843,700]}}`
nhiều lần, hết 256 token trước khi đóng list. Đây vừa là lặp box vừa dẫn tới JSON không hợp lệ.
F1 toàn bộ E0={e0['micro']['f1']:.4f}; không chỉ chấm những output parse được.

### Câu 3 — Thang tọa độ

YOLO `(xc,yc,w,h)` đổi sang `[xc-w/2,yc-h/2,xc+w/2,yc+h/2]*1000`, làm tròn và kẹp miền.
Khi vẽ, x nhân `640/1000`, y nhân `360/1000`. Nhầm norm1000 thành pixel sẽ làm box bị
phóng khác nhau theo hai trục, giảm IoU và thường giảm F1; không thể suy ra một F1 cụ thể khi chưa chấm.
Round-trip thực tế ghi sai số tối đa 0,32 pixel so với box nguồn đã kẹp, phù hợp sai số làm tròn.
Ví dụ `[400,400,600,600]` tương ứng pixel `[256,144,384,216]`.
{scale_note}

### Câu 4 — Mask loss

Prompt và token ảnh được đặt label=-100; chỉ JSON đáp án và token kết thúc chịu loss.
Kiểm tra Kaggle trên `20250216_164325`: 639 token tổng, 389 token bị mask, 250 token được học,
trong đó prompt chứa 220 vision token. Decode token được học khớp JSON target và `<|im_end|>`.
Nếu học cả prompt, loss bị chi phối bởi việc sao chép chỉ dẫn cố định; học token placeholder ảnh
không trực tiếp tối ưu định vị. Loss thấp khi ấy chưa chứng minh đáp án JSON chính xác hơn.

### Câu 5 — Lượng tử hóa và projector

NF4 lưu trọng số nén, không phải tensor float có thể cập nhật đầy đủ bằng AdamW.
LoRA học ma trận phụ, còn E2/E3 yêu cầu cập nhật trực tiếp projector nên visual và lm_head
được loại khỏi lượng tử hóa. Kiểm tra thực tế thấy 252 module NF4, không module visual/lm_head bị NF4.
`llm_int8_skip_modules` gồm cả `model.visual` để khớp tên đầy đủ ở transformers đã ghim.
Projector trainable được đưa lên float32 để gradient, cập nhật và trạng thái optimizer ổn định;
autocast vẫn chạy phép tính phù hợp ở fp16. Đây là lựa chọn ổn định của lab, không phải quy luật
rằng mọi projector đều bắt buộc float32. Lưu riêng `projector.pt` và nạp lại cùng adapter khi suy luận.

### Câu 6 — Chia theo phiên quay

Hai frame gần nhau có nội dung gần trùng; chia ngẫu nhiên khiến train có frame gần giống test,
cho F1 lạc quan và không đo đúng khả năng tổng quát sang phiên quay mới.

| Split | Ảnh | pothole | crack | manhole | Phiên |
|---|---:|---:|---:|---:|---:|
| train | 1422 | 1074 | 1750 | 691 | 13 |
| val | 276 | 66 | 470 | 103 | 2 |
| test | 311 | 121 | 299 | 163 | 3 |

Khám phá dữ liệu xác nhận giao phiên giữa các split=0, duplicate image_id=0.
Train có một box crack diện tích 0 tại `vlcsnap-00058`, object index 1.
Chỉ bỏ annotation này khỏi target train, giữ ảnh và audit; giữ nguyên val/test và công cụ đề.

### Câu 7 — So sánh E1/E2/E3

{pairwise}
{value_note}
Đánh giá lợi ích phải đặt F1 và F1 từng lớp cạnh thời gian/VRAM.
E2 thêm khoảng 109M tham số projector; E3 cần gradient qua ViT nên có thể tốn VRAM/thời gian hơn.
Không đồng nhất dự đoán lý thuyết về chi phí với số đo. Các số thực tế nằm trong bảng.

### Câu 8 — Khoảng tin cậy

Độ rộng CI95: {widths}
E0 có CI95=[0,0441; 0,0877], rộng khoảng 0,0436.
Với 311 ảnh và chỉ 3 phiên test, không nên kết luận từ chênh lệch F1 nhỏ.
CI của scorer bootstrap theo ảnh; các frame trong một phiên có tương quan nên CI này có thể
lạc quan hơn bootstrap theo phiên. CI chồng nhau không tự chứng minh hai setup tương đương;
kiểm định chênh lệch cần bootstrap ghép cặp cùng ảnh hoặc theo phiên, tốt hơn nữa thêm phiên độc lập.

### Câu 9 — Lỗi theo lớp

E0 yếu nhất ở crack: F1=0,0040 (TP=1, FP=197, FN=298), so với pothole=0,0994 và manhole=0,1407.
Crack có nhiều mẫu train nhất nên không thể giải thích đơn giản do ít dữ liệu.
Hình dạng dài/mảnh, box nhỏ, bóng đổ/vạch đường và định nghĩa vùng crack gây khó; đây là các giả thuyết,
cần ảnh lỗi để đối chiếu. Box nhỏ khiến lệch vài pixel có thể giảm IoU mạnh.
Ảnh minh họa dùng GT xanh và dự đoán đỏ, cả hai đều đổi norm1000 sang pixel trước khi vẽ.

{trained_weakness}

{chr(10).join(figures) if figures else 'Ảnh dự đoán sẽ được tạo từ output model thực tế khi lượt GPU hoàn tất.'}

### Câu 10 — E4

Biến chọn là `data_size`: 10% so với 100% ảnh train, các siêu tham số khác giữ nguyên,
kể cả 300 optimizer steps và effective batch=8. Như vậy mức 10% lặp lại dữ liệu nhiều hơn;
đây là hiệu ứng giảm độ đa dạng ở cùng ngân sách bước, không phải cùng số epoch.
Mốc 100% dùng lại đúng dự đoán của setup gốc theo GUIDE; chỉ train thêm mức 10%.
{baseline_note}
Mức 10% dùng 142 ảnh đầu theo thứ tự split. Phân bố phiên/lớp có thể khác toàn bộ train,
nên tác động đo được gồm giảm độ đa dạng và thay đổi thành phần mẫu, không chỉ số lượng.
{' '.join(e4text) if e4text else 'Chưa có bằng chứng E4; chưa kết luận tác động của kích thước dữ liệu.'}
{'Mức 10% chưa hoàn tất; chưa thể kết luận ablation.' if 'e4_data_size_0.1' not in evidence else ''}
{ablation_conclusion}
Đọc kết quả kèm CI95 và giới hạn test nhỏ; không lựa chọn lại siêu tham số dựa vào test.

## Nhìn lại

Các điểm dễ sai là mask loss, loại visual khỏi NF4, lưu/nạp projector và quy đổi tọa độ khi vẽ.
Đã sửa lỗi ảnh minh họa dùng tọa độ norm1000 trực tiếp như pixel và lỗi tiếp tục cache E0 ghi đè phần đã chạy.
Không sửa `scripts/` và `tests/` của đề. Phần E5 có hàm reward IoU Python thuần trong
`student_reward.py` và 6 test (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng, box khổng lồ).
Chưa chạy GRPO; chưa có kết quả RL hay kết luận reward hacking từ thực nghiệm.
Phần thiếu bằng chứng vẫn phải được hoàn tất trước khi gọi bài là hoàn chỉnh.
'''
    (ROOT / 'REPORT.md').write_text(text, encoding='utf-8')
    print('REPORT.md updated; pending:', ', '.join(pending) or 'none')


if __name__ == '__main__':
    main()
