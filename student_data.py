"""CP1: inspect prepared data without changing the instructor's tools."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.labcommon import CLASSES, read_jsonl
from scripts.splitting import SPLITS, dataset_version

COLORS = {"pothole": "#ff6868", "crack": "#ffd84d", "manhole": "#55ddff"}


def yolo_to_pixels(yolo, width, height):
    """Normalized center/size -> pixel corners, only for displaying labels."""
    xc, yc, w, h = yolo
    return ((xc - w / 2) * width, (yc - h / 2) * height,
            (xc + w / 2) * width, (yc + h / 2) * height)


def inspect_splits(rows, meta):
    """Reject leakage/invalid schema; report source-label defects without changing data."""
    seen_ids, session_split, counts, outside = set(), {}, {}, 0
    zero_area = []
    assignments = []
    for split in SPLITS:
        boxes = {c: 0 for c in CLASSES}
        sessions = set()
        for row in rows[split]:
            image_id, session = row["image_id"], row["session"]
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image_id: {image_id}")
            seen_ids.add(image_id)
            if session in session_split and session_split[session] != split:
                raise ValueError(f"Session leakage: {session}")
            session_split[session] = split
            sessions.add(session)
            assignments.append((image_id, split))
            if row["width"] <= 0 or row["height"] <= 0:
                raise ValueError(f"Invalid image dimensions: {image_id}")
            for index, obj in enumerate(row["objects"]):
                lab, yolo = obj["label"], obj["yolo"]
                if lab not in CLASSES or len(yolo) != 4:
                    raise ValueError(f"Invalid label: {image_id}")
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) and 0 <= v <= 1 for v in yolo):
                    raise ValueError(f"Invalid YOLO coordinates: {image_id}")
                if yolo[2] <= 0 or yolo[3] <= 0:
                    zero_area.append({"split": split, "image_id": image_id,
                                      "object_index": index, "label": lab, "yolo": yolo})
                boxes[lab] += 1
                x1, y1, x2, y2 = yolo_to_pixels(yolo, 1, 1)
                outside += int(x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6)
        counts[split] = {"images": len(rows[split]), "boxes": boxes}
        if counts[split] != meta["counts"][split]:
            raise ValueError(f"Counts disagree with meta.json: {split}")
    version = dataset_version(assignments)
    if version != meta["dataset_version"] or session_split != meta["groups"]:
        raise ValueError("Split assignment disagrees with meta.json")
    return {"dataset_version": version, "counts": counts,
            "sessions": {s: len({r['session'] for r in rows[s]}) for s in SPLITS},
            "session_overlap": 0, "duplicate_image_ids": 0,
            "zero_area_annotations": zero_area,
            "boxes_extending_outside_image": outside}


def render_train_examples(train, data_dir, output):
    """Show six ground-truth examples; do not use test images for inspection."""
    chosen = []
    for label in CLASSES:
        candidates = [r for r in train if 1 <= len(r["objects"]) <= 5
                      and all(o["yolo"][2] > 0 and o["yolo"][3] > 0 for o in r["objects"])
                      and any(o["label"] == label for o in r["objects"])]
        for row in candidates:
            if row["image_id"] not in {r["image_id"] for r in chosen}:
                chosen.append(row)
                if len(chosen) % 2 == 0:
                    break
    if len(chosen) != 6:
        raise ValueError("Need six distinct training examples covering all three classes")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (1280, 1280), "#141a24")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "TRAIN ground truth only | pothole: red | crack: yellow | manhole: cyan", font=font, fill="white")
    for index, row in enumerate(chosen):
        with Image.open(Path(data_dir) / row["image_path"]) as image:
            image = image.convert("RGB")
        painter = ImageDraw.Draw(image)
        for obj in row["objects"]:
            box = yolo_to_pixels(obj["yolo"], *image.size)
            color = COLORS[obj["label"]]
            painter.rectangle(box, outline=color, width=3)
            x, y = max(0, box[0]), max(0, box[1] - 20)
            text = obj["label"]
            painter.rectangle(painter.textbbox((x, y), text, font=font), fill="#141a24")
            painter.text((x, y), text, font=font, fill=color)
        x, y = (index % 2) * 640, 40 + (index // 2) * 410
        draw.text((x + 8, y), row["image_id"], font=font, fill="white")
        canvas.paste(image, (x, y + 24))
        draw.text((x + 8, y + 386), f"session: {row['session']} | boxes: {len(row['objects'])}", font=font, fill="white")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return [{"image_id": r["image_id"], "session": r["session"], "objects": r["objects"]} for r in chosen]


def explore(data_dir="data", out_dir="report/checkpoints"):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    rows = {s: read_jsonl(data_dir / "splits" / f"{s}.jsonl") for s in SPLITS}
    meta = json.loads((data_dir / "splits/meta.json").read_text(encoding="utf-8"))
    summary = inspect_splits(rows, meta)
    for split in SPLITS:
        for row in rows[split]:
            with Image.open(data_dir / row["image_path"]) as image:
                if image.size != (row["width"], row["height"]):
                    raise ValueError(f"Image size mismatch: {row['image_id']}")
                image.verify()
    summary["verified_image_files"] = sum(len(v) for v in rows.values())
    summary["examples"] = render_train_examples(rows["train"], data_dir, out_dir / "01-train-labels.png")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "01-data-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dataset_version:", summary["dataset_version"])
    print("split   images  sessions  pothole  crack  manhole")
    for s, c in summary["counts"].items():
        print(s, c["images"], summary["sessions"][s], *(c["boxes"][lab] for lab in CLASSES))
    print("Session overlap:", summary["session_overlap"])
    print("Verified image files:", summary["verified_image_files"])
    print("Boxes extending outside image:", summary["boxes_extending_outside_image"])
    print("Zero-area annotations (preserved, not silently removed):", summary["zero_area_annotations"])
    print("First displayed training label:", json.dumps(summary["examples"][0], ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="report/checkpoints")
    args = parser.parse_args()
    explore(args.data_dir, args.out_dir)
