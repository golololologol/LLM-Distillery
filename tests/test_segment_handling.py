import pytest

from classes.formatter import SentinelFormatter
from classes.data_classes import ConvoProcessed
from classes.preprocessing import preprocess_samples
from utils.dataset_utils import DatasetCanonicalizer


@pytest.fixture(scope="module")
def formatter(tokenizer):
    return SentinelFormatter(tokenizer)


def _canon(messages):
    return DatasetCanonicalizer().canonicalize_messages(messages)


def _run(formatter, messages, handling) -> ConvoProcessed:
    out = preprocess_samples(
        [{"id": 0, "messages": messages}],
        formatter, context_len=512,
        segment_handling=handling,
    )
    assert out
    return out[0]


def test_content_collect_default(formatter):
    messages = _canon([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer-bytes"},
    ])
    r = _run(formatter, messages, {"content": "active"})
    assert "answer-bytes" in r.formatted_text
    types = {s.type for s in r.segments}
    assert "content" in types


def test_content_context_renders_but_no_segment(formatter):
    messages = _canon([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer-bytes"},
    ])
    text, _, segments, _, _, _ = formatter.render(
        messages, save_roles={"assistant"}, context_len=512,
        segment_handling={"content": "context"},
    )
    assert "answer-bytes" in text
    assert all(s.type != "content" for s in segments)


def test_content_disabled_strips_field(formatter):
    messages = _canon([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "secret-bytes"},
    ])
    text, _, segments, _, _, _ = formatter.render(
        messages, save_roles={"assistant"}, context_len=512,
        segment_handling={"content": "disabled"},
    )
    assert "secret-bytes" not in text
    assert all(s.type != "content" for s in segments)


def test_tool_call_context(formatter):
    messages = _canon([
        {"role": "user", "content": "use tool"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "1", "name": "noop", "arguments": '{"x":1}'}
        ]},
        {"role": "tool", "content": "done", "tool_call_id": "1"},
    ])
    r = _run(formatter, messages, {"content": "active", "tool_call": "context"})
    assert '{"x":1}' in r.formatted_text or '{"x": 1}' in r.formatted_text
    assert all(s.type not in ("tool_call", "tool_name", "tool_arguments") for s in r.segments)
    assert any(s.type == "content" for s in r.segments)


def test_tool_call_disabled(formatter):
    messages = _canon([
        {"role": "user", "content": "use tool"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "1", "name": "noop", "arguments": '{"secret":42}'}
        ]},
        {"role": "tool", "content": "done", "tool_call_id": "1"},
    ])
    r = _run(formatter, messages, {"content": "active", "tool_call": "disabled"})
    assert "secret" not in r.formatted_text
    assert all(s.type not in ("tool_call", "tool_name", "tool_arguments") for s in r.segments)


def test_default_none_collects_content(formatter):
    messages = _canon([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ])
    out = preprocess_samples([{"id": 0, "messages": messages}], formatter, context_len=256)
    assert any(s.type == "content" for s in out[0].segments)
