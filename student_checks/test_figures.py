"""Checks for the rubric-figure matcher. Synthetic boxes only, never model evidence."""
from PIL import Image

from student_figures import build, match_one, render, GT_COLOR


def _entry(image_id="a", objects=None, width=640, height=360):
    return {"image_id": image_id, "width": width, "height": height,
            "image_path": "img.jpg", "objects": objects or []}


def _obj(label, xc, yc, w, h):
    return {"label": label, "yolo": [xc, yc, w, h]}


POTHOLE_PERFECT = '[{"label":"pothole","bbox_2d":[400,400,600,600]}]'


def test_correct_match_is_labelled_correct():
    tag, score = match_one(_entry(objects=[_obj("pothole", 0.5, 0.5, 0.2, 0.2)]), POTHOLE_PERFECT)
    assert tag == "correct" and score == 1.0


def test_wrong_class_is_not_counted_as_correct():
    raw = '[{"label":"crack","bbox_2d":[400,400,600,600]}]'
    tag, _ = match_one(_entry(objects=[_obj("pothole", 0.5, 0.5, 0.2, 0.2)]), raw)
    assert tag == "wrong_box"


def test_low_overlap_is_wrong_box():
    raw = '[{"label":"pothole","bbox_2d":[10,10,60,60]}]'
    tag, score = match_one(_entry(objects=[_obj("pothole", 0.5, 0.5, 0.2, 0.2)]), raw)
    assert tag == "wrong_box" and score < 0.5


def test_boxes_without_ground_truth_are_hallucinated():
    tag, _ = match_one(_entry(), POTHOLE_PERFECT)
    assert tag == "hallucinated"


def test_missing_boxes_when_ground_truth_exists():
    tag, _ = match_one(_entry(objects=[_obj("pothole", 0.5, 0.5, 0.2, 0.2)]), "[]")
    assert tag == "missed_all"


def test_invalid_json_is_its_own_tag():
    tag, _ = match_one(_entry(objects=[_obj("pothole", 0.5, 0.5, 0.2, 0.2)]), "not json")
    assert tag == "invalid_json"


def test_empty_on_both_sides_is_correct_by_definition():
    assert match_one(_entry(), "[]")[0] == "both_empty"


def test_render_converts_normalized_boxes_to_image_pixels(tmp_path):
    from PIL import ImageColor, ImageFont
    Image.new('RGB', (640, 360), (20, 20, 20)).save(tmp_path / 'img.jpg')
    entry = _entry(objects=[_obj('pothole', 0.5, 0.5, 0.2, 0.2)])
    output = tmp_path / 'result.png'
    render(entry, '[]', tmp_path, output, ImageFont.load_default())
    with Image.open(output) as image:
        assert image.getpixel((256, 180)) == ImageColor.getrgb(GT_COLOR)
        assert image.getpixel((400, 180)) != ImageColor.getrgb(GT_COLOR)


def test_build_writes_images_and_summary(tmp_path):
    (tmp_path / "splits").mkdir(parents=True)
    entry = _entry(objects=[_obj("pothole", 0.5, 0.5, 0.2, 0.2)])
    (tmp_path / "splits" / "test.jsonl").write_text(
        __import__("json").dumps(entry) + "\n", encoding="utf-8")
    Image.new("RGB", (640, 360), (20, 20, 20)).save(tmp_path / "img.jpg")
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text(
        __import__("json").dumps({"image_id": "a", "raw": POTHOLE_PERFECT}) + "\n", encoding="utf-8")
    summary = build(str(predictions), str(tmp_path), "test", str(tmp_path / "figs"), want=1)
    assert summary["n_selected"] == 1
    assert summary["tiles"][0]["tag"] == "correct"
    assert (tmp_path / "figs" / "predictions-summary.json").exists()


def test_build_rejects_empty_prediction_file(tmp_path):
    (tmp_path / "splits").mkdir(parents=True)
    (tmp_path / "splits" / "test.jsonl").write_text(
        __import__("json").dumps(_entry()) + "\n", encoding="utf-8")
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text("", encoding="utf-8")
    try:
        build(str(predictions), str(tmp_path), "test", str(tmp_path / "figs"))
    except ValueError as error:
        assert "No predictions" in str(error)
    else:
        raise AssertionError("expected ValueError for empty predictions")
