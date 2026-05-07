import numpy as np
import pytest
from classes.data_classes import Segment, ConvoProcessed


def _make_convo(text="Hello world", segments=None):
    return ConvoProcessed(
        origin_convo_id=0,
        tokens=np.array([1, 2, 3]),
        segments=segments or [],
        content_sha="abc",
        formatted_text=text,
        padding=0,
        cropped=False,
        length=3,
    )


def test_segment_byte_count():
    s = Segment(msg_idx=0, type="content", byte_start=5, byte_end=15)
    assert s.byte_count == 10


def test_byte_ranges():
    segs = [Segment(0, "content", 0, 5), Segment(1, "reasoning", 10, 20)]
    convo = _make_convo(segments=segs)
    assert convo.byte_ranges() == [(0, 5), (10, 20)]


def test_byte_ranges_empty():
    convo = _make_convo()
    assert convo.byte_ranges() == []


def test_actual_bytes():
    text = "Hello world"
    segs = [Segment(0, "content", 0, 5), Segment(0, "content", 6, 11)]
    convo = _make_convo(text=text, segments=segs)
    result = convo.actual_bytes()
    assert bytes(result) == b"Helloworld"


def test_actual_bytes_empty():
    convo = _make_convo()
    assert len(convo.actual_bytes()) == 0


def test_filter_by_handling():
    segs = [
        Segment(0, "content", 0, 5),
        Segment(0, "reasoning", 5, 10),
        Segment(1, "content", 10, 15),
    ]
    convo = _make_convo(segments=segs)
    handling = {"content": "active", "reasoning": "context"}
    assert len(convo.filter_by_handling(handling, "active")) == 2
    assert len(convo.filter_by_handling(handling, "context")) == 1
    assert len(convo.filter_by_handling(handling, "disabled")) == 0


def test_segment_has_canonical_hash_and_truncated_defaults():
    s = Segment(0, "content", 0, 5)
    assert s.canonical_target_hash == ""
    assert s.truncated is False


def test_to_manifest_entry_includes_new_fields():
    s = Segment(2, "reasoning", 10, 20, canonical_target_hash="abc123", truncated=True)
    entry = s.to_manifest_entry()
    assert entry == {
        "msg_idx": 2,
        "type": "reasoning",
        "byte_count": 10,
        "canonical_target_hash": "abc123",
        "truncated": True,
    }


def test_manifest_roundtrip_preserves_hash_and_truncated():
    originals = [
        Segment(0, "content", 0, 5, canonical_target_hash="h0", truncated=False),
        Segment(1, "reasoning", 5, 12, canonical_target_hash="h1", truncated=True),
        Segment(1, "content", 12, 20, canonical_target_hash="h2", truncated=False),
    ]
    manifest = [s.to_manifest_entry() for s in originals]
    restored = Segment.from_manifest(manifest)
    assert [r.canonical_target_hash for r in restored] == ["h0", "h1", "h2"]
    assert [r.truncated for r in restored] == [False, True, False]
    assert [r.byte_count for r in restored] == [5, 7, 8]
    assert [r.type for r in restored] == ["content", "reasoning", "content"]


def test_from_manifest_requires_manifest_fields():
    manifest = [{"msg_idx": 0, "type": "content", "byte_count": 5}]
    with pytest.raises(KeyError):
        Segment.from_manifest(manifest)
