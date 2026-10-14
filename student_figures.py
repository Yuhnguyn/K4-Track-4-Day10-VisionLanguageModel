"""CP9: rubric figure — six predictions with both predicted and ground-truth boxes."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.labcommon import read_jsonl
from scripts.score import IOU_THR, _gt_boxes, iou_xyxy, parse_output

GT_COLOR = "#41d97b"
PRED_COLOR = "#ff6b6b"


def match_one(entry, raw):
    """Return tag + best IoU for one image. Predicted boxes never come from ground truth."""
    parsed = parse_output(raw)
    if parsed is None:
        return "invalid_json", 0.0
    gt = _gt_boxes(entry)
    if not parsed and not gt:
        return "both_empty", 1.0
    if not parsed:
        return "missed_all", 0.0
    if not gt:
        return "hallucinated", 0.0
    best, matched = 0.0, False
    for prediction in parsed:
        for truth in gt:
            if prediction["label"] != truth["label"]:
                continue
            value = iou_xyxy(prediction["bbox"], truth["bbox"])
            best = max(best, value)
            matched = matched or value >= IOU_THR
    return ("correct" if matched else "wrong_box"), best


def render(entry, raw, data_dir, out_path, font):
    parsed = parse_output(raw) or []
    with Image.open(Path(data_dir) / entry["image_path"]) as handle:
        image = handle.convert("RGB")
    draw = ImageDraw.Draw(image)
    def pixels(box):
        return tuple(value * (image.width if index % 2 == 0 else image.height) / 1000
                     for index, value in enumerate(box))
    for truth in _gt_boxes(entry):
        box = pixels(truth["bbox"])
        draw.rectangle(box, outline=GT_COLOR, width=3)
        draw.text((box[0] + 3, box[1] + 3), f"GT {truth['label']}",
                  font=font, fill=GT_COLOR)
    for prediction in parsed:
        box = pixels(prediction["bbox"])
        draw.rectangle(box, outline=PRED_COLOR, width=2)
        draw.text((box[0] + 3, box[3] - 18),
                  f"PRED {prediction['label']}", font=font, fill=PRED_COLOR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return image.size


def build(pred, data_dir="data", split="test", out_dir="report/figures", want=6, prefix="predictions"):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    entries = read_jsonl(data_dir / "splits" / f"{split}.jsonl")
    preds = {row["image_id"]: row["raw"] for row in read_jsonl(pred)}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    buckets = {}
    for entry in entries:
        raw = preds.get(entry["image_id"])
        if raw is None:
            continue
        tag, score = match_one(entry, raw)
        buckets.setdefault(tag, []).append((score, entry, raw))

    # Prefer a mix: correct hits, wrong boxes, and an invalid one for the error analysis.
    order = ["correct", "wrong_box", "hallucinated", "missed_all", "invalid_json", "both_empty"]
    chosen = []
    ranked = {tag: sorted(buckets.get(tag, []), key=lambda item: -item[0]) for tag in order}
    while len(chosen) < want and any(ranked.values()):
        for tag in order:
            if ranked[tag] and len(chosen) < want:
                score, entry, raw = ranked[tag].pop(0)
                chosen.append((tag, score, entry, raw))
    if not chosen:
        raise ValueError("No predictions to illustrate")

    tiles = []
    for index, (tag, score, entry, raw) in enumerate(chosen):
        path = out_dir / f"{prefix}-{index + 1:02d}-{tag}.png"
        render(entry, raw, data_dir, path, font)
        tiles.append({"tag": tag, "best_iou": round(score, 4), "image_id": entry["image_id"],
                      "n_gt": len(entry["objects"]), "n_pred": len(parse_output(raw) or []),
                      "file": str(path)})

    summary = {"split": split, "prediction_file": str(pred), "n_selected": len(tiles),
               "legend": {"GT_COLOR": GT_COLOR, "PRED_COLOR": PRED_COLOR},
               "note": "Ground truth (green) and model prediction (red) drawn on the same image.",
               "tiles": tiles}
    (out_dir / f"{prefix}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out-dir", default="report/figures")
    parser.add_argument("--want", type=int, default=6)
    parser.add_argument("--prefix", default="predictions")
    args = parser.parse_args()
    build(args.pred, args.data_dir, args.split, args.out_dir, args.want, args.prefix)
