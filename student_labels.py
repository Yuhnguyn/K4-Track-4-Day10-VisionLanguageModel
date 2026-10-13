"""CP2: deterministic grounding targets; never repair invalid boxes silently."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import parse_output
from scripts.splitting import SPLITS
from student_data import yolo_to_pixels

PROMPT = (
    'Locate all potholes, cracks, and manholes in this image. '
    'Return only a JSON list, without markdown or explanations. '
    'Each object must have "label" (one of "pothole", "crack", "manhole") '
    'and "bbox_2d": [x1,y1,x2,y2], the top-left and bottom-right corners. '
    'Use integer coordinates on the normalized 0-1000 scale: '
    'x is relative to image width and y is relative to image height, not pixels. '
    'Require 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000. '
    'Order objects from top to bottom, then left to right. '
    'If there are no such objects, return [].'
)


class ZeroAreaSourceBox(ValueError):
    """Source annotation has zero extent; train-only filtering is explicit."""


def yolo_to_box(label, yolo):
    if label not in CLASSES:
        raise ValueError(f"Unknown label: {label}")
    if len(yolo) != 4 or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and math.isfinite(v) and 0 <= v <= 1 for v in yolo
    ):
        raise ValueError("YOLO must contain four finite numbers in [0,1]")
    xc, yc, w, h = yolo
    if w <= 0 or h <= 0:
        raise ZeroAreaSourceBox("Source box has zero area")
    corners = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
    box = [max(0, min(1000, round(v * 1000))) for v in corners]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Box collapses after rounding/clamping")
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects):
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda obj: (obj["bbox_2d"][1], obj["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixels(box, width, height):
    return tuple(v * size / 1000 for v, size in zip(box, (width, height, width, height)))


def training_target(row, split="train"):
    """Approved policy: keep image and other boxes; return an explicit removal log."""
    if split != "train":
        raise ValueError("Training-only filtering must not change val/test targets")
    kept, removed = [], []
    for index, obj in enumerate(row["objects"]):
        try:
            yolo_to_box(obj["label"], obj["yolo"])
        except ZeroAreaSourceBox:
            removed.append({"image_id": row["image_id"], "object_index": index,
                            "reason": "Source box has zero area", "object": obj})
        else:
            kept.append(obj)
    return {"answer": objects_to_answer(kept), "removed_annotations": removed}


def verify_coordinates(data_dir="data", out_dir="report/checkpoints"):
    """Audit conversions, not model predictions; official splits stay unchanged."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"dataset_version": json.loads((data_dir / 'splits/meta.json').read_text())["dataset_version"],
               "scale": "norm1000", "splits": {}, "rejected_objects": [],
               "max_roundtrip_error_px_vs_clipped_source": 0.0}
    example = None
    for split in SPLITS:
        rows = read_jsonl(data_dir / "splits" / f"{split}.jsonl")
        valid_objects, rejected_images = 0, set()
        for row in rows:
            for index, obj in enumerate(row["objects"]):
                try:
                    converted = yolo_to_box(obj["label"], obj["yolo"])
                except ValueError as error:
                    rejected_images.add(row["image_id"])
                    summary["rejected_objects"].append({"split": split, "image_id": row["image_id"],
                        "object_index": index, "reason": str(error), "object": obj})
                    continue
                valid_objects += 1
                original = yolo_to_pixels(obj["yolo"], row["width"], row["height"])
                restored = box_to_pixels(converted["bbox_2d"], row["width"], row["height"])
                for before, after, size in zip(original, restored,
                                              (row["width"], row["height"], row["width"], row["height"])):
                    error = abs(max(0, min(size, before)) - after)
                    if error > size / 2000 + 1e-8:
                        raise AssertionError(f"Roundtrip exceeds rounding bound: {row['image_id']}")
                    summary["max_roundtrip_error_px_vs_clipped_source"] = max(
                        summary["max_roundtrip_error_px_vs_clipped_source"], error)
            if row["image_id"] not in rejected_images:
                answer = objects_to_answer(row["objects"])
                if parse_output(answer) is None:
                    raise AssertionError(f"Official parser rejects target: {row['image_id']}")
                if example is None and split == "train" and row["objects"]:
                    example = row
        summary["splits"][split] = {"images": len(rows), "converted_objects": valid_objects,
                                    "images_requiring_explicit_training_policy": len(rejected_images)}
    if example is None:
        raise ValueError("No valid training example to visualize")
    answer = objects_to_answer(example["objects"])
    summary["example"] = {"image_id": example["image_id"], "answer": answer}
    with Image.open(data_dir / example["image_path"]) as image:
        original, restored = image.convert("RGB"), image.convert("RGB")
    left, right = ImageDraw.Draw(original), ImageDraw.Draw(restored)
    for obj in example["objects"]:
        left.rectangle(yolo_to_pixels(obj["yolo"], *original.size), outline="#55ddff", width=2)
        box = yolo_to_box(obj["label"], obj["yolo"])["bbox_2d"]
        right.rectangle(box_to_pixels(box, *restored.size), outline="#ffd84d", width=2)
    canvas = Image.new("RGB", (original.width * 2, original.height + 40), "#141a24")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Original YOLO -> pixels (cyan)", fill="white")
    draw.text((original.width + 8, 8), "YOLO -> integer 0-1000 -> pixels (yellow)", fill="white")
    canvas.paste(original, (0, 40))
    canvas.paste(restored, (original.width, 40))
    canvas.save(out_dir / "02-coordinate-roundtrip.png")
    (out_dir / "02-coordinate-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="report/checkpoints")
    args = parser.parse_args()
    verify_coordinates(args.data_dir, args.out_dir)
