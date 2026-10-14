import json
import pytest
from student_reward import iou_reward

BOX = {'label': 'pothole', 'bbox_2d': [100, 100, 200, 200]}


def test_perfect():
    assert iou_reward(json.dumps([BOX]), [BOX]) == 1.0


def test_invalid_json():
    assert iou_reward('invalid JSON', [BOX]) == 0.0


def test_wrong_class():
    assert iou_reward(json.dumps([{**BOX, 'label': 'crack'}]), [BOX]) == 0.2


def test_extra_and_duplicate_boxes_are_penalized():
    assert iou_reward(json.dumps([BOX, BOX]), [BOX]) == pytest.approx(0.6)


def test_empty_cases():
    assert iou_reward('[]', []) == 1.0
    assert iou_reward('[]', [BOX]) == 0.2
    assert iou_reward(json.dumps([BOX]), []) == 0.2


def test_giant_box_cannot_collect_small_object_rewards():
    giant = {'label': 'pothole', 'bbox_2d': [0, 0, 1000, 1000]}
    assert iou_reward(json.dumps([giant]), [BOX]) == 0.2
