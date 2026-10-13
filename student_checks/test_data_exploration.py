"""Student checks live outside the protected starter tests/ directory."""
from copy import deepcopy

import pytest
from PIL import Image

from scripts.labcommon import CLASSES
from scripts.splitting import SPLITS, dataset_version
from student_data import inspect_splits, render_train_examples, yolo_to_pixels


def fixture_data():
    rows = {s: [{"image_id": s, "session": s, "width": 640, "height": 360,
                 "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]}]
            for s in SPLITS}
    meta = {"counts": {s: {"images": 1, "boxes": {c: int(c == 'pothole') for c in CLASSES}}
                       for s in SPLITS}, "groups": {s: s for s in SPLITS},
            "dataset_version": dataset_version([(s, s) for s in SPLITS])}
    return rows, meta


def test_yolo_to_pixels():
    assert yolo_to_pixels([0.5, 0.5, 0.2, 0.2], 640, 360) == pytest.approx((256, 144, 384, 216))


def test_clean_splits():
    rows, meta = fixture_data()
    assert inspect_splits(rows, meta)["session_overlap"] == 0


@pytest.mark.parametrize('problem,match', [
    ('session', 'Session leakage'), ('id', 'Duplicate image_id'),
    ('nan', 'Invalid YOLO'), ('count', 'Counts disagree'),
    ('version', 'Split assignment disagrees'),
])
def test_reject_bad_data(problem, match):
    rows, meta = fixture_data()
    if problem == 'session':
        rows['val'][0]['session'] = 'train'
    elif problem == 'id':
        rows['val'][0]['image_id'] = 'train'

    elif problem == 'nan':
        rows['train'][0]['objects'][0]['yolo'][0] = float('nan')
    elif problem == 'count':
        meta['counts']['train']['images'] = 2
    else:
        meta['dataset_version'] = 'wrong'
    with pytest.raises(ValueError, match=match):
        inspect_splits(rows, meta)


def test_zero_area_annotation_is_reported_without_changing_source():
    rows, meta = fixture_data()
    rows['train'][0]['objects'][0]['yolo'][3] = 0
    before = deepcopy(rows)
    summary = inspect_splits(rows, meta)
    assert len(summary['zero_area_annotations']) == 1
    assert summary['counts']['train']['boxes']['pothole'] == 1
    assert rows == before


def test_render_six_distinct_examples(tmp_path):
    rows, _ = fixture_data()
    train = []
    for index, label in enumerate(CLASSES * 2):
        row = deepcopy(rows['train'][0])
        row.update(image_id=str(index), image_path=f'{index}.jpg')
        row['objects'][0]['label'] = label
        Image.new('RGB', (640, 360)).save(tmp_path / row['image_path'])
        train.append(row)
    output = tmp_path / 'labels.png'
    examples = render_train_examples(train, tmp_path, output)
    assert len({r['image_id'] for r in examples}) == 6
    with Image.open(output) as image:
        assert image.size == (1280, 1280)
