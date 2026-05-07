"""Tests for v8 segment + anchor manifest extensions (plan §5, §6, §11)."""
import pytest

from classes.formatter import SentinelFormatter
from utils.dataset_utils import DatasetCanonicalizer


@pytest.fixture(scope="module")
def formatter(tokenizer):
    return SentinelFormatter(tokenizer)


def _canon(messages):
    return DatasetCanonicalizer().canonicalize_messages(messages)


def test_segments_carry_role_and_segment_idx(formatter):
    messages = _canon([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    text, _, segs, _, _, anchors = formatter.render(
        messages, save_roles={"assistant"}, context_len=512,
    )
    assert all(s.role == "assistant" for s in segs)
    # First segment in each message is segment_idx=0.
    by_msg = {}
    for s in segs:
        by_msg.setdefault(s.msg_idx, []).append(s.segment_idx)
    for ids in by_msg.values():
        assert ids == list(range(len(ids)))


def test_anchors_emitted_pre_and_post_per_segment(formatter):
    messages = _canon([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    _, tokens, segs, _, _, anchors = formatter.render(
        messages, save_roles={"assistant"}, context_len=512,
    )
    # Each segment should produce a pre + post anchor.
    sides = sorted(a.anchor_side for a in anchors if a.msg_idx == 1)
    assert sides.count("pre") == len([s for s in segs if s.msg_idx == 1])
    assert sides.count("post") == len([s for s in segs if s.msg_idx == 1])
    for a in anchors:
        if not a.unreachable:
            assert 0 <= a.token_pos < len(tokens)


def test_anchor_manifest_entry_has_keys():
    from classes.data_classes import Anchor
    a = Anchor(msg_idx=1, segment_idx=0, type="content", tool_call_idx=None,
               anchor_side="post", token_pos=12)
    e = a.to_manifest_entry()
    assert e["msg_idx"] == 1
    assert e["segment_idx"] == 0
    assert e["anchor_side"] == "post"


def test_segment_to_manifest_emits_v8_only_when_set():
    from classes.data_classes import Segment
    s = Segment(0, "content", 0, 5)
    e = s.to_manifest_entry()
    # v7 surface preserved verbatim when nothing v8-specific is set.
    assert "segment_idx" not in e
    assert "tool_call_idx" not in e
    assert "pre_event_row" not in e

    s2 = Segment(0, "content", 0, 5, segment_idx=2, tool_call_idx=1,
                 handling="context", role="user", pre_event_row=4,
                 post_event_row=5, byte_low_confidence=True)
    e2 = s2.to_manifest_entry()
    assert e2["segment_idx"] == 2
    assert e2["tool_call_idx"] == 1
    assert e2["handling"] == "context"
    assert e2["role"] == "user"
    assert e2["pre_event_row"] == 4
    assert e2["post_event_row"] == 5
    assert e2["byte_low_confidence"] is True
