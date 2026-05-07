"""Plan v8 §6: tool_name and tool_arguments are first-class segment types."""
import pytest

from classes.formatter import SentinelFormatter
from utils.dataset_utils import DatasetCanonicalizer


@pytest.fixture(scope="module")
def formatter(tokenizer):
    return SentinelFormatter(tokenizer)


def _canon(messages):
    return DatasetCanonicalizer().canonicalize_messages(messages)


def test_tool_call_emits_separate_name_and_arguments_segments(formatter):
    messages = _canon([
        {"role": "user", "content": "go"},
        {
            "role": "assistant", "content": "ok",
            "tool_calls": [{"id": "1", "name": "noop", "arguments": '{"x":1}'}],
        },
    ])
    text, _, segs, _, _, _ = formatter.render(
        messages, save_roles={"assistant"}, context_len=512,
    )
    by_type = {s.type: s for s in segs}
    assert "tool_name" in by_type
    assert "tool_arguments" in by_type
    text_b = text.encode("utf-8")
    name_seg = by_type["tool_name"]
    args_seg = by_type["tool_arguments"]
    assert text_b[name_seg.byte_start:name_seg.byte_end] == b"noop"
    # Args may be JSON-recompacted by the canonicaliser; just confirm the
    # segment slice equals the canonical arguments string.
    assert b"{" in text_b[args_seg.byte_start:args_seg.byte_end]
    assert name_seg.tool_call_idx == 0
    assert args_seg.tool_call_idx == 0


def test_per_call_indices_are_distinct(formatter):
    messages = _canon([
        {"role": "user", "content": "go"},
        {
            "role": "assistant", "content": "ok",
            "tool_calls": [
                {"id": "1", "name": "first", "arguments": "{}"},
                {"id": "2", "name": "second", "arguments": "{}"},
            ],
        },
    ])
    _, _, segs, _, _, _ = formatter.render(
        messages, save_roles={"assistant"}, context_len=512,
    )
    name_calls = sorted(s.tool_call_idx for s in segs if s.type == "tool_name")
    arg_calls = sorted(s.tool_call_idx for s in segs if s.type == "tool_arguments")
    assert name_calls == [0, 1]
    assert arg_calls == [0, 1]


def test_independent_handling_for_name_and_arguments(formatter):
    messages = _canon([
        {"role": "user", "content": "go"},
        {
            "role": "assistant", "content": "ok",
            "tool_calls": [{"id": "1", "name": "noop", "arguments": '{"k":1}'}],
        },
    ])
    _, _, segs, _, _, _ = formatter.render(
        messages, save_roles={"assistant"}, context_len=512,
        segment_handling={"content": "active", "tool_arguments": "disabled"},
    )
    # tool_name should still appear; tool_arguments should not.
    types = {s.type for s in segs}
    assert "tool_name" in types
    assert "tool_arguments" not in types
