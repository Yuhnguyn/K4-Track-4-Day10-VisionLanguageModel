"""E0 error taxonomy: classify raw model output. Never rewrites predictions."""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import evaluate, parse_output


def _looks_truncated(raw):
    text = (raw or "").strip()
    if not text:
        return True
    return text.count("[") != text.count("]") or text.count("{") != text.count("}")


def _duplicate_count(parsed):
    seen = Counter((item["label"], tuple(item["bbox"])) for item in parsed)
    return sum(count - 1 for count in seen.values() if count > 1)


def classify_output(raw):
    """Return human-readable defect tags for one raw model output."""
    parsed = parse_output(raw)
    if parsed is None:
        return ["json_invalid_truncated"] if _looks_truncated(raw) else ["json_invalid_other"]
    tags = []
    if not parsed:
        return ["empty_answer"]
    if _duplicate_count(parsed):
        tags.append("duplicate_boxes")
    unknown = sorted({item["label"] for item in parsed} - set(CLASSES))
    if unknown:
        tags.append("unknown_label:" + ",".join(unknown))
    if len(parsed) > 20:
        tags.append("very_many_boxes")
    widths = [item["bbox"][2] - item["bbox"][0] for item in parsed]
    heights = [item["bbox"][3] - item["bbox"][1] for item in parsed]
    spans = [max(widths) >= 600, max(heights) >= 600]
    if any(spans):
        tags.append("full_frame_sized_box")
    if any(min(item["bbox"][0], item["bbox"][1]) <= 1 for item in parsed) and \
       any(max(item["bbox"][2], item["bbox"][3]) >= 999 for item in parsed):
        tags.append("touches_0_and_1000")
    coords = [value for item in parsed for value in item["bbox"]]
    if max(coords) > 1000:
        tags.append("coordinate_above_1000_wrong_scale_or_range")
    return tags or ["plausible"]


def analyse(pred="work/e0.jsonl", data_dir="data", split="test", out="report/checkpoints/03-e0-errors.json"):
    data_dir, out = Path(data_dir), Path(out)
    entries = read_jsonl(data_dir / "splits" / f"{split}.jsonl")
    preds = {row["image_id"]: row["raw"] for row in read_jsonl(pred)}
    missing = [e["image_id"] for e in entries if e["image_id"] not in preds]
    results = evaluate(entries, preds, coord="norm1000")
    taxonomy, examples = Counter(), {}
    for entry in entries:
        raw = preds.get(entry["image_id"])
        if raw is None:
            taxonomy["missing_prediction"] += 1
            continue
        for tag in classify_output(raw):
            taxonomy[tag] += 1
            examples.setdefault(tag, []).append({"image_id": entry["image_id"], "raw": raw[:400]})
    by_class = {c: results["per_class"][c] for c in (*CLASSES, "other")}
    summary = {
        "name": "E0-error-taxonomy",
        "purpose": "Describes raw zero-shot output; does not replace the official scorer.",
        "limitation": ("On 640x360 images, pixel-scale coordinates stay inside 0-1000, so a range "
                       "check cannot detect a wrong coordinate scale. Detecting that error needs "
                       "geometric comparison by the official scorer, not this taxonomy."),
        "split": split, "n_images": len(entries), "scored_predictions": len(preds),
        "missing_predictions": len(missing),
        "parse_valid_rate": results["parse_valid_rate"],
        "f1": results["micro"]["f1"],
        "f1_ci95": results["f1_ci95"],
        "per_class": by_class,
        "error_taxonomy_counts": dict(taxonomy.most_common()),
        "example_outputs": {tag: rows[:3] for tag, rows in examples.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "example_outputs"}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", default="work/e0.jsonl")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out", default="report/checkpoints/03-e0-errors.json")
    args = parser.parse_args()
    analyse(args.pred, args.data_dir, args.split, args.out)
