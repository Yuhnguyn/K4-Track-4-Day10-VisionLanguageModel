"""Checks for the E0 error classifier. Synthetic strings, not model evidence."""
import json

import pytest

from student_analysis import classify_output, _duplicate_count


def test_plausible_output():
    raw = json.dumps([{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}])
    assert classify_output(raw) == ["plausible"]


def test_truncated_json_is_named_truncation():
    raw = '[{"label": "crack", "bbox_2d": [0, 370, 1000, 700]}, {"label": "crack"'
    assert classify_output(raw) == ["json_invalid_truncated"]


def test_broken_but_balanced_json_is_not_called_truncated():
    assert classify_output("[1,2]") == ["json_invalid_other"]


def test_empty_answer_is_its_own_tag():
    assert classify_output("[]") == ["empty_answer"]


def test_duplicate_boxes_are_detected():
    box = {"label": "crack", "bbox_2d": [0, 370, 1000, 700]}
    assert _duplicate_count([{**box, "bbox": (0.0, 370.0, 1000.0, 700.0)}] * 3) == 2
    tags = classify_output(json.dumps([box] * 3))
    assert "duplicate_boxes" in tags and "full_frame_sized_box" in tags


def test_unknown_label_is_reported_verbatim():
    raw = json.dumps([{"label": "puddle", "bbox_2d": [1, 1, 50, 50]}])
    assert classify_output(raw) == ["unknown_label:puddle"]


def test_coordinates_above_1000_violate_the_norm1000_contract():
    raw = json.dumps([{"label": "crack", "bbox_2d": [10, 20, 1280, 720]}])
    tags = classify_output(raw)
    assert "coordinate_above_1000_wrong_scale_or_range" in tags


def test_pixel_scale_on_640x360_is_not_detectable_by_range_alone():
    """Pixel values fit inside 0-1000, so range checks cannot flag this error."""
    raw = json.dumps([{"label": "pothole", "bbox_2d": [256, 144, 384, 216]}])
    assert "coordinate_above_1000_wrong_scale_or_range" not in classify_output(raw)


def test_degenerate_box_is_invalid_not_plausible():
    raw = json.dumps([{"label": "crack", "bbox_2d": [5, 5, 5, 9]}])
    assert classify_output(raw) == ["json_invalid_other"]


@pytest.mark.parametrize("bad", ["", "not json", None])
def test_garbage_is_never_called_plausible(bad):
    assert classify_output(bad) != ["plausible"]
