import pytest

from student_inference import predict_split, validate_cached_predictions


def test_prediction_cache_keeps_raw_output_unchanged():
    raw = 'Here is JSON: []'
    rows = [{'image_id': 'a', 'raw': raw}]
    assert validate_cached_predictions(rows, {'a'}) == {'a'}
    assert rows[0]['raw'] == raw


@pytest.mark.parametrize('rows', [
    [{'image_id': 'a', 'raw': '[]'}, {'image_id': 'a', 'raw': '[]'}],
    [{'image_id': 'unexpected', 'raw': '[]'}],
    [{'image_id': 'a', 'raw': None}],
])
def test_bad_prediction_cache_is_rejected(rows):
    with pytest.raises(ValueError, match='Prediction cache'):
        validate_cached_predictions(rows, {'a'})


def test_test_split_cannot_be_truncated():
    with pytest.raises(ValueError, match='full official test'):
        predict_split(split='test', limit=4)


@pytest.mark.parametrize('batch,limit', [(0, None), (4, 0)])
def test_positive_batch_and_limit_required(batch, limit):
    with pytest.raises(ValueError, match='positive'):
        predict_split(split='val', batch_size=batch, limit=limit)
