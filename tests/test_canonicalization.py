import json
import logging

import pytest

from utils.dataset_utils import DatasetCanonicalizer, compute_source_sha


C = DatasetCanonicalizer()
CT = DatasetCanonicalizer(inline_reasoning_mode="think_tags")


def test_reasoning_content_renamed():
    out = C.canonicalize_messages([
        {"role": "assistant", "content": "ans", "reasoning_content": "rsn"},
    ])
    assert out[0]["reasoning"] == "rsn"
    assert out[0]["content"] == "ans"


def test_existing_reasoning_preserved():
    out = C.canonicalize_messages([
        {"role": "assistant", "content": "ans", "reasoning": "rsn"},
    ])
    assert out[0]["reasoning"] == "rsn"
    assert out[0]["content"] == "ans"


def test_inline_think_not_extracted_by_default():
    out = C.canonicalize_messages([
        {"role": "assistant", "content": "<think>rsn</think>ans"},
    ])
    assert out[0]["reasoning"] is None
    assert out[0]["content"] == "<think>rsn</think>ans"


def test_inline_think_extracted():
    out = CT.canonicalize_messages([
        {"role": "assistant", "content": "<think>rsn</think>ans"},
    ])
    assert out[0]["reasoning"] == "rsn"
    assert out[0]["content"] == "ans"


def test_begin_of_thought_extracted():
    out = CT.canonicalize_messages([
        {"role": "assistant", "content": "<|begin_of_thought|>rsn<|end_of_thought|>ans"},
    ])
    assert out[0]["reasoning"] == "rsn"
    assert out[0]["content"] == "ans"


def test_non_assistant_untouched():
    out = C.canonicalize_messages([
        {"role": "user", "content": "<think>x</think>hi"},
    ])
    assert out[0]["content"] == "<think>x</think>hi"
    assert out[0]["reasoning"] is None


def test_multi_think_default_errors():
    msgs = [{"role": "assistant", "content": "<think>a</think>mid<think>b</think>end"}]
    with pytest.raises(ValueError, match="<think>"):
        CT.canonicalize_messages(msgs)


def test_multi_think_first_only():
    c = DatasetCanonicalizer(on_multi_think="first_only", inline_reasoning_mode="think_tags")
    msgs = [{"role": "assistant", "content": "<think>a</think>mid<think>b</think>end"}]
    out = c.canonicalize_messages(msgs)
    assert out[0]["reasoning"] == "a"
    # Second <think> stays as literal text in content
    assert "<think>b</think>" in out[0]["content"]
    assert "mid" in out[0]["content"]


def test_multi_think_warn_first_only_logs(caplog):
    c = DatasetCanonicalizer(on_multi_think="warn_first_only", inline_reasoning_mode="think_tags")
    msgs = [{"role": "assistant", "content": "<think>a</think>mid<think>b</think>end"}]
    with caplog.at_level(logging.WARNING):
        out = c.canonicalize_messages(msgs)
    assert out[0]["reasoning"] == "a"
    assert any("<think>" in r.message for r in caplog.records)


def test_multi_turn_reasoning_no_false_positive():
    # N assistant messages, each with 1 <think> block - must NOT fire multi-think check.
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>r1</think>a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>r2</think>a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "<think>r3</think>a3"},
    ]
    out = CT.canonicalize_messages(msgs)
    reasonings = [m["reasoning"] for m in out if m["role"] == "assistant"]
    assert reasonings == ["r1", "r2", "r3"]
    contents = [m["content"] for m in out if m["role"] == "assistant"]
    assert contents == ["a1", "a2", "a3"]


def test_tool_call_openai_shape_flattens():
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
        }],
    }]
    out = C.canonicalize_messages(msgs)
    tc = out[0]["tool_calls"][0]
    assert tc["name"] == "get_weather"
    assert tc["arguments"] == '{"city":"London"}'
    assert tc["id"] == "call_1"


def test_tool_call_dict_arguments_json_dumped():
    msgs = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"name": "f", "arguments": {"x": 1, "y": 2}, "id": "c"}],
    }]
    out = C.canonicalize_messages(msgs)
    tc = out[0]["tool_calls"][0]
    assert isinstance(tc["arguments"], str)
    assert json.loads(tc["arguments"]) == {"x": 1, "y": 2}


def test_tool_call_string_arguments_trusted():
    # Non-JSON-object strings are preserved verbatim.
    raw_str = "not valid json at all"
    msgs = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"name": "f", "arguments": raw_str, "id": "c"}],
    }]
    out = C.canonicalize_messages(msgs)
    assert out[0]["tool_calls"][0]["arguments"] == raw_str


def test_tool_call_string_json_object_renormalized():
    # JSON-object strings are re-serialized with sort_keys+compact separators,
    # so semantically-equivalent orderings produce identical canonical arguments.
    msgs_a = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"name": "f", "arguments": '{"a": 1, "b": 2}', "id": "c"}],
    }]
    msgs_b = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"name": "f", "arguments": '{"b":2,"a":1}', "id": "c"}],
    }]
    out_a = C.canonicalize_messages(msgs_a)
    out_b = C.canonicalize_messages(msgs_b)
    assert out_a[0]["tool_calls"][0]["arguments"] == '{"a":1,"b":2}'
    assert out_a[0]["tool_calls"][0]["arguments"] == out_b[0]["tool_calls"][0]["arguments"]


def _args(msgs):
    out = C.canonicalize_messages(msgs)
    return out[0]["tool_calls"][0]["arguments"]


def _wrap(arguments):
    return [{
        "role": "assistant", "content": "",
        "tool_calls": [{"name": "f", "arguments": arguments, "id": "c"}],
    }]


def test_tool_call_string_json_list_renormalized():
    assert _args(_wrap("[1, 2, 3]")) == _args(_wrap("[1,2,3]"))
    assert _args(_wrap("[1, 2, 3]")) == "[1,2,3]"


def test_tool_call_string_json_number_renormalized():
    assert _args(_wrap("42")) == _args(_wrap(" 42 "))
    assert _args(_wrap("42")) == "42"


def test_tool_call_string_json_bool_null_renormalized():
    assert _args(_wrap("true")) == "true"
    assert _args(_wrap(" null ")) == "null"


def test_tool_call_string_json_string_literal_idempotent():
    assert _args(_wrap('"hello"')) == '"hello"'


def test_tool_call_non_json_preserved_verbatim():
    raw = "not json at all"
    assert _args(_wrap(raw)) == raw


def test_tool_call_hash_stable_across_key_order():
    from classes.formatter import hash_tool_call
    msgs_a = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"name": "f", "arguments": '{"a": 1, "b": 2}', "id": "c"}],
    }]
    msgs_b = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"name": "f", "arguments": '{"b":2,"a":1}', "id": "c"}],
    }]
    tc_a = C.canonicalize_messages(msgs_a)[0]["tool_calls"][0]
    tc_b = C.canonicalize_messages(msgs_b)[0]["tool_calls"][0]
    assert hash_tool_call(tc_a) == hash_tool_call(tc_b)


def test_tool_call_missing_id_generated():
    msgs = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"name": "f", "arguments": "{}"}],
    }]
    out = C.canonicalize_messages(msgs)
    tc_id = out[0]["tool_calls"][0]["id"]
    assert tc_id.startswith("call_")
    assert len(tc_id) > len("call_")


def test_role_must_be_string():
    with pytest.raises(ValueError, match="role must be a string"):
        C.canonicalize_messages([{"role": 123, "content": "x"}])


def test_role_must_be_in_valid_set():
    with pytest.raises(ValueError, match="role"):
        C.canonicalize_messages([{"role": "wizard", "content": "x"}])


def test_assistant_missing_everything_raises():
    with pytest.raises(ValueError, match="no content"):
        C.canonicalize_messages([{"role": "assistant", "content": ""}])


def test_tool_call_without_name_raises():
    with pytest.raises(ValueError, match="name"):
        C.canonicalize_messages([{
            "role": "assistant", "content": "x",
            "tool_calls": [{"arguments": "{}", "id": "c"}],
        }])


def test_tool_call_non_json_serializable_dict_raises():
    class Weird:
        pass
    with pytest.raises((ValueError, TypeError)):
        C.canonicalize_messages([{
            "role": "assistant", "content": "x",
            "tool_calls": [{"name": "f", "arguments": {"x": Weird()}, "id": "c"}],
        }])


def test_content_must_be_string_or_none():
    with pytest.raises(ValueError, match="content"):
        C.canonicalize_messages([{"role": "assistant", "content": ["blocks"]}])


def test_canonicalize_sample_preserves_other_fields():
    sample = {"id": 5, "extra": "keep", "messages": [{"role": "user", "content": "q"},
                                                       {"role": "assistant", "content": "a"}]}
    out = C.canonicalize_sample(sample)
    assert out["id"] == 5
    assert out["extra"] == "keep"
    assert out["messages"][1]["content"] == "a"


def test_compute_source_sha_same_across_calls():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ans", "reasoning": "rsn"},
    ]
    assert compute_source_sha(messages, {"assistant"}) == compute_source_sha(messages, {"assistant"})


def test_compute_source_sha_changes_with_reasoning():
    m1 = [{"role": "assistant", "content": "ans", "reasoning": "rsn1"}]
    m2 = [{"role": "assistant", "content": "ans", "reasoning": "rsn2"}]
    assert compute_source_sha(m1, {"assistant"}) != compute_source_sha(m2, {"assistant"})
