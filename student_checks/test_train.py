"""CPU checks for the training-data contract. Uses fake tensors: no torch needed."""
import pytest

from student_train import PAD_KEYS, encode_example, lr_at


class _Row:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _Mask:
    def __init__(self, flags):
        self._flags = list(flags)

    def sum(self):
        return sum(self._flags)


class FakeTensor:
    """Minimal 2-D tensor with just the operations encode_example relies on."""

    def __init__(self, rows):
        self._rows = [list(r) for r in rows]

    @property
    def shape(self):
        return (len(self._rows), len(self._rows[0]))

    def clone(self):
        return FakeTensor(self._rows)

    def tolist(self):
        return [list(r) for r in self._rows]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, cols = key
            return _Row(self._rows[row][cols])
        return _Row(self._rows[key])

    def __setitem__(self, key, value):
        rows, cols = key
        if rows != slice(None, None, None):
            raise NotImplementedError("test helper only supports labels[:, :n] = v")
        for row in self._rows:
            indexes = range(len(row))[cols]
            for index in indexes:
                row[index] = value

    def __ne__(self, other):
        return _Mask([value != other for row in self._rows for value in row])


class FakeProcessor:
    def __init__(self, prompt_ids=(10, 11, 12), answer_ids=(20, 21, 22, 23), corrupt_prefix=False):
        self.prompt_ids = list(prompt_ids)
        self.answer_ids = list(answer_ids)
        self.corrupt_prefix = corrupt_prefix

    def apply_chat_template(self, messages, **kwargs):
        has_assistant = any(message["role"] == "assistant" for message in messages)
        if has_assistant:
            ids = list(self.prompt_ids) + list(self.answer_ids)
            if self.corrupt_prefix:
                ids[0] = 99
            return {"input_ids": FakeTensor([ids])}
        return {"input_ids": FakeTensor([list(self.prompt_ids)])}


def test_only_answer_tokens_are_supervised():
    processor = FakeProcessor()
    encoded = encode_example(processor, image=None, answer="ANSWER")
    labels = encoded["labels"].tolist()[0]
    assert labels[:3] == [-100, -100, -100]
    assert labels[3:] == [20, 21, 22, 23]
    assert encoded["input_ids"].tolist()[0] == [10, 11, 12, 20, 21, 22, 23]


def test_prompt_is_not_modified_by_masking():
    processor = FakeProcessor()
    encoded = encode_example(processor, image=None, answer="ANSWER")
    assert encoded["input_ids"].tolist()[0][:3] == [10, 11, 12]


def test_detects_a_prompt_that_is_not_a_prefix():
    with pytest.raises(ValueError, match="prefix"):
        encode_example(FakeProcessor(corrupt_prefix=True), image=None, answer="ANSWER")


def test_rejects_when_there_is_nothing_to_learn():
    processor = FakeProcessor(answer_ids=())
    with pytest.raises(ValueError, match="not longer than the prompt"):
        encode_example(processor, image=None, answer="ANSWER")


def test_pad_keys_include_labels_and_input_ids():
    assert "labels" in PAD_KEYS and "input_ids" in PAD_KEYS


@pytest.mark.parametrize("step,total,expected_lo,expected_hi", [
    (0, 300, 0.0, 2e-4),
    (14, 300, 2e-4 * 0.99, 2e-4),
    (149, 300, 0.0, 2e-4 * 0.6),
    (299, 300, 0.0, 2e-4 * 0.05),
])
def test_lr_schedule_warmup_then_cosine(step, total, expected_lo, expected_hi):
    value = lr_at(step, total, base_lr=2e-4)
    assert expected_lo <= value <= expected_hi


def test_lr_is_non_increasing_after_warmup():
    values = [lr_at(step, 300, base_lr=2e-4) for step in range(15, 300)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def test_lr_peaks_at_warmup_boundary_and_ends_near_zero():
    warmup_end = lr_at(14, 300, base_lr=2e-4)
    assert warmup_end == pytest.approx(2e-4, rel=0.02)
    assert lr_at(299, 300, base_lr=2e-4) < 2e-4 * 0.01
