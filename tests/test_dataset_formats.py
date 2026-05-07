from classes.dataset_formats import REGISTRY, normalize


def test_registry_has_expected_strategies():
    expected = {
        "openai", "sharegpt", "alpaca", "completion", "instruction_response",
        "query_answer", "dolly", "prompt_completion",
        "hermes_inline", "per_block_roles", "nemotron_post_training",
    }
    assert expected <= set(REGISTRY)


def test_openai_passthrough():
    s = {"messages": [{"role": "user", "content": "hi"}]}
    assert normalize(s, "openai") == {"messages": s["messages"]}


def test_sharegpt():
    s = {"init": "sys", "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}
    out = normalize(s, "sharegpt")
    assert out["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_alpaca_with_input():
    s = {"instruction": "Add", "input": "1 + 2", "output": "3"}
    out = normalize(s, "alpaca")
    assert out["messages"][-2]["content"] == "Add\n1 + 2"
    assert out["messages"][-1] == {"role": "assistant", "content": "3"}


def test_completion():
    out = normalize({"text": "hello"}, "completion")
    assert out == {"messages": [{"role": "assistant", "content": "hello"}]}


def test_instruction_response():
    out = normalize({"instruction": "q", "response": "a"}, "instruction_response")
    assert out["messages"] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_query_answer():
    out = normalize({"query": "q", "answer": "a"}, "query_answer")
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]


def test_dolly_with_context():
    out = normalize({"instruction": "Q", "context": "C", "response": "A"}, "dolly")
    assert out["messages"][0]["content"] == "C\nQ"


def test_prompt_completion():
    out = normalize({"prompt": "p", "completion": "c"}, "prompt_completion")
    assert out["messages"][0]["content"] == "p"
    assert out["messages"][1]["content"] == "c"


def test_hermes_inline_extracts_tool_calls():
    s = {
        "id": 1,
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant",
             "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
            {"role": "tool", "content": '<tool_response>\n{"temp": 13}\n</tool_response>'},
            {"role": "assistant", "content": "It is 13C."},
        ],
    }
    out = normalize(s, "hermes_inline")
    asst1 = out["messages"][1]
    assert asst1["role"] == "assistant"
    assert asst1["content"] == ""
    assert len(asst1["tool_calls"]) == 1
    assert asst1["tool_calls"][0]["name"] == "get_weather"
    assert asst1["tool_calls"][0]["arguments"] == '{"city": "Paris"}'
    cid = asst1["tool_calls"][0]["id"]
    tool_msg = out["messages"][2]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == cid
    assert out["messages"][3] == {"role": "assistant", "content": "It is 13C."}


def test_per_block_roles_collapses_blocks():
    s = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do it"},
            {"role": "reasoning", "content": "<think>plan</think>"},
            {"role": "tool_call",
             "content": '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'},
            {"role": "tool_output", "content": "<tool_response>ok</tool_response>"},
            {"role": "answer", "content": "done"},
        ],
    }
    out = normalize(s, "per_block_roles")
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    asst = out["messages"][2]
    assert asst["reasoning"] == "plan"
    assert asst["tool_calls"][0]["name"] == "f"
    assert out["messages"][3]["tool_call_id"] == asst["tool_calls"][0]["id"]
    assert out["messages"][4]["content"] == "done"


def test_nemotron_post_training():
    s = {
        "input": [{"role": "user", "content": "hi"}],
        "output": "hello",
        "reasoning": "off",
        "system_prompt": "be brief",
    }
    out = normalize(s, "nemotron_post_training")
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][-1] == {"role": "assistant", "content": "hello"}


def test_nemotron_post_training_with_reasoning():
    s = {
        "input": [{"role": "user", "content": "hi"}],
        "output": "hello",
        "reasoning": "I greet back",
    }
    out = normalize(s, "nemotron_post_training")
    assert out["messages"][-1]["reasoning"] == "I greet back"


def test_auto_detection():
    assert normalize({"messages": [{"role": "user", "content": "x"}]}, "auto")["messages"]
    assert normalize({"conversations": [{"from": "human", "value": "x"}]}, "auto")["messages"]
    assert normalize({"instruction": "i", "output": "o"}, "auto")["messages"]
    assert normalize({"text": "t"}, "auto")["messages"]


def test_unknown_format_raises():
    import pytest
    with pytest.raises(ValueError):
        normalize({"messages": []}, "nonexistent_format")


def test_id_preserved():
    out = normalize({"id": "abc", "messages": [{"role": "user", "content": "x"}]}, "openai")
    assert out["id"] == "abc"
