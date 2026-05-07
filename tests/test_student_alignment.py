import numpy as np
import pytest

from classes.data_classes import ConvoProcessed, Segment


def _make_convo(cid, segments, content_sha):
    return ConvoProcessed(
        origin_convo_id=cid,
        tokens=np.zeros(10, dtype=np.int32),
        segments=segments,
        content_sha=content_sha,
        formatted_text="x" * 100,
        padding=0,
        cropped=False,
        length=10,
    )


def test_segment_byte_count():
    s = Segment(0, "answer", 10, 50)
    assert s.byte_count == 40


def test_convo_byte_ranges():
    segs = [Segment(0, "reasoning", 0, 10), Segment(0, "answer", 10, 30)]
    cv = _make_convo(0, segs, "sha")
    assert cv.byte_ranges() == [(0, 10), (10, 30)]


def test_convo_empty_segments():
    cv = _make_convo(0, [], "sha")
    assert cv.byte_ranges() == []


def test_manifest_offset_matching():
    """Test the offset-building logic used in _compute_sample_loss."""
    manifest = [
        {"msg_idx": 0, "type": "reasoning", "byte_count": 100},
        {"msg_idx": 0, "type": "answer", "byte_count": 50},
    ]
    teacher_offset = {}
    row = 0
    for entry in manifest:
        bc = entry["byte_count"]
        teacher_offset[(entry["msg_idx"], entry["type"])] = (row, row + bc)
        row += bc

    assert teacher_offset[(0, "reasoning")] == (0, 100)
    assert teacher_offset[(0, "answer")] == (100, 150)


def test_segment_teacher_matching():
    """Test that student segments match teacher manifest entries by (msg_idx, type)."""
    segs = [Segment(0, "answer", 0, 20)]
    manifest = [
        {"msg_idx": 0, "type": "reasoning", "byte_count": 50},
        {"msg_idx": 0, "type": "answer", "byte_count": 30},
    ]
    teacher_offset = {}
    row = 0
    for entry in manifest:
        bc = entry["byte_count"]
        teacher_offset[(entry["msg_idx"], entry["type"])] = (row, row + bc)
        row += bc

    matched = []
    for seg in segs:
        key = (seg.msg_idx, seg.type)
        if key in teacher_offset:
            matched.append((seg, teacher_offset[key]))

    assert len(matched) == 1
    seg, (t_start, t_end) = matched[0]
    assert seg.type == "answer"
    assert t_start == 50
    assert t_end == 80


def test_length_mismatch_takes_min():
    """Student segment shorter than teacher → use student length."""
    seg = Segment(0, "answer", 0, 20)
    t_start, t_end = 0, 5000
    n = min(seg.byte_count, t_end - t_start)
    assert n == 20
