import json

import pytest

from scripts.score import evaluate, parse_output
from student_labels import PROMPT, box_to_pixels, objects_to_answer, yolo_to_box


def test_center_box_and_pixel_roundtrip():
    obj = yolo_to_box('pothole', [0.5, 0.5, 0.2, 0.2])
    assert obj == {'label': 'pothole', 'bbox_2d': [400, 400, 600, 600]}
    assert box_to_pixels(obj['bbox_2d'], 640, 360) == (256, 144, 384, 216)


def test_clamps_to_image_bounds():
    assert yolo_to_box('crack', [0.05, 0.95, 0.2, 0.2])['bbox_2d'] == [0, 850, 150, 1000]


def test_stable_order_and_compact_json():
    objects = [{'label': 'crack', 'yolo': [0.7, 0.7, 0.1, 0.1]},
               {'label': 'pothole', 'yolo': [0.3, 0.3, 0.1, 0.1]}]
    answer = objects_to_answer(objects)
    assert ' ' not in answer
    assert json.loads(answer)[0]['label'] == 'pothole'
    assert parse_output(answer) is not None
    assert objects_to_answer([]) == '[]'


@pytest.mark.parametrize('label,yolo', [
    ('unknown', [0.5, 0.5, 0.2, 0.2]),
    ('crack', [0.5, 0.5, 0.2, 0]),
    ('crack', [0.5, 0.5, 0, 0.2]),
    ('crack', [0.5, 0.5, 0.0001, 0.2]),
    ('crack', [float('nan'), 0.5, 0.2, 0.2]),
    ('crack', [True, 0.5, 0.2, 0.2]),
    ('crack', [1.1, 0.5, 0.2, 0.2]),
    ('crack', [0.5, 0.5, -0.2, 0.2]),
    ('crack', [0.5, 0.5, 0.2]),
])
def test_rejects_invalid_or_collapsed_targets(label, yolo):
    with pytest.raises(ValueError):
        yolo_to_box(label, yolo)


def test_rounding_error_bound():
    box = yolo_to_box('manhole', [0.333333, 0.777777, 0.123456, 0.234567])['bbox_2d']
    xc, yc, w, h = 0.333333, 0.777777, 0.123456, 0.234567
    expected = ((xc - w/2) * 640, (yc - h/2) * 360, (xc + w/2) * 640, (yc + h/2) * 360)
    for before, after, size in zip(expected, box_to_pixels(box, 640, 360), (640, 360, 640, 360)):
        assert abs(before - after) <= size / 2000 + 1e-8


def test_scale_mismatch_with_official_scorer_on_synthetic_fixture():
    # Geometry contract test, not an E0 model result.
    entries = [{'image_id': 'synthetic', 'width': 640, 'height': 360,
                'objects': [{'label': 'pothole', 'yolo': [0.5, 0.5, 0.2, 0.2]}]}]
    raw = objects_to_answer(entries[0]['objects'])
    normal = evaluate(entries, {'synthetic': raw}, coord='norm1000', n_boot=20)
    wrong = evaluate(entries, {'synthetic': raw}, coord='pixel', n_boot=20)
    assert normal['micro']['f1'] == 1.0
    assert wrong['micro']['f1'] == 0.0
    assert wrong['parse_valid_rate'] == 1.0


def test_prompt_matches_target_contract():
    assert all(label in PROMPT for label in ('pothole', 'crack', 'manhole'))
    assert '0-1000' in PROMPT and 'bbox_2d' in PROMPT and 'return []' in PROMPT
