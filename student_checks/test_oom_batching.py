"""Checks for the OOM-resilient batch loop: no image may ever be skipped.

Fakes live in this file only; the real torch path is exercised on the GPU host.
"""
import contextlib
import json
import sys
import types

import pytest
from PIL import Image

import student_inference

IMAGE_TOKEN_ID = 7
VISION_TOKENS = 220
PROMPT_IDS = [IMAGE_TOKEN_ID] * VISION_TOKENS + [1, 2, 3]


class SimulatedOOM(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError on a host without CUDA."""


class MiniBatch(dict):
    """What apply_chat_template returns: keyed tensors that move to a device."""

    def to(self, *args, **kwargs):
        return self


class MiniTensor:
    """Just the operations generate_predictions performs on input tensors."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        _, columns = key
        return MiniTensor([row[columns] for row in self.rows])

    def __eq__(self, other):
        return MiniMask([[value == other for value in row] for row in self.rows])


class MiniMask:
    def __init__(self, flags):
        self.flags = flags

    def sum(self, dim=None):
        if dim == 1:
            return MiniColumn([sum(row) for row in self.flags])
        return sum(sum(row) for row in self.flags)


class MiniColumn:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeProcessor:
    def __init__(self):
        self.seen_batches = []

    def apply_chat_template(self, conversations, **kwargs):
        self.seen_batches.append(len(conversations))
        return MiniBatch(input_ids=MiniTensor([PROMPT_IDS for _ in conversations]))

    def batch_decode(self, generated, skip_special_tokens=True):
        return [json.dumps({"n": len(row)}) for row in generated.rows]


class FakeModel:
    """Raises OOM for batches larger than `oom_above`, like a small GPU would."""

    def __init__(self, oom_above):
        self.oom_above = oom_above
        self.device = "cpu"
        self.generated_batch_sizes = []
        self.config = types.SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)

    def eval(self):
        return self

    def generate(self, **inputs):
        size = inputs["input_ids"].shape[0]
        if size > self.oom_above:
            raise SimulatedOOM("simulated")
        self.generated_batch_sizes.append(size)
        return MiniTensor([PROMPT_IDS + [9, 9] for _ in range(size)])


@pytest.fixture
def fake_torch(monkeypatch):
    fake = types.ModuleType("torch")
    fake.inference_mode = contextlib.nullcontext
    fake.cuda = types.SimpleNamespace(
        OutOfMemoryError=SimulatedOOM,
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        get_device_name=lambda index=0: "fake-t4",
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def dataset(tmp_path):
    entries = []
    for index in range(5):
        name = f"img{index}.jpg"
        Image.new("RGB", (640, 360), (30, 30, 30)).save(tmp_path / name)
        entries.append({"image_id": f"id{index}", "image_path": name})
    return entries


def run(entries, tmp_path, model, batch_size):
    output = tmp_path / "pred.jsonl"
    processor = FakeProcessor()
    stats = student_inference.generate_predictions(
        model, processor, entries, tmp_path, output, batch_size=batch_size, log=lambda *a: None)
    lines = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    return stats, lines, processor


def test_all_images_written_once_in_order(fake_torch, dataset, tmp_path):
    stats, lines, _ = run(dataset, tmp_path, FakeModel(oom_above=99), batch_size=2)
    assert stats["n_images"] == 5
    assert [line["image_id"] for line in lines] == ["id0", "id1", "id2", "id3", "id4"]


def test_oom_halves_batch_and_retries_without_losing_images(fake_torch, dataset, tmp_path):
    # batch 4 OOMs, so the loop must fall back to 2 and still cover every image.
    stats, lines, processor = run(dataset, tmp_path, FakeModel(oom_above=2), batch_size=4)
    assert stats["n_images"] == 5
    assert [line["image_id"] for line in lines] == ["id0", "id1", "id2", "id3", "id4"]
    assert processor.seen_batches[0] == 4, "phải thử batch 4 trước"
    assert 2 in processor.seen_batches, "phải thử lại với batch 2"


def test_oom_at_batch_one_is_raised_not_hidden(fake_torch, dataset, tmp_path):
    with pytest.raises(SimulatedOOM):
        run(dataset, tmp_path, FakeModel(oom_above=0), batch_size=2)


def test_rejects_non_positive_batch(fake_torch, dataset, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run(dataset, tmp_path, FakeModel(oom_above=99), batch_size=0)


def test_resume_appends_without_erasing_previous_images(fake_torch, dataset, tmp_path):
    output = tmp_path / 'pred.jsonl'
    old = {'image_id': 'id0', 'raw': 'original raw output'}
    output.write_text(json.dumps(old) + '\n', encoding='utf-8')
    student_inference.generate_predictions(
        FakeModel(99), FakeProcessor(), dataset[1:], tmp_path, output,
        batch_size=2, append=True, log=lambda *a: None)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0] == old
    assert [row['image_id'] for row in rows] == [row['image_id'] for row in dataset]
