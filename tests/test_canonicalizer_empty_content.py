"""Plan v8 §6: assistant messages without raw content but with reasoning/tool_calls
must materialise an empty content segment so the byte channel sees the boundary.
"""
from utils.dataset_utils import DatasetCanonicalizer


def test_assistant_with_only_tool_calls_gets_empty_content():
    msgs = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "1", "name": "noop", "arguments": "{}"}],
        },
    ]
    out = DatasetCanonicalizer().canonicalize_messages(msgs)
    assert out[1]["content"] == ""
    assert out[1]["tool_calls"]


def test_assistant_with_only_reasoning_gets_empty_content():
    msgs = [
        {"role": "user", "content": "think"},
        {"role": "assistant", "reasoning": "thinking"},
    ]
    out = DatasetCanonicalizer().canonicalize_messages(msgs)
    assert out[1]["content"] == ""
    assert out[1]["reasoning"] == "thinking"


def test_user_message_with_no_content_keeps_none():
    msgs = [{"role": "user", "content": None}]
    out = DatasetCanonicalizer().canonicalize_messages(msgs)
    assert out[0]["content"] is None
