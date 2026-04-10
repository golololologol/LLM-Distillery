import pytest
from classes.preprocessing import find_content_byte_ranges, _merge_reasoning


def _full_bytes(messages, tokenizer, **kwargs):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **kwargs
    ).encode("utf-8")


def test_basic_single_turn(tokenizer):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    ranges = find_content_byte_ranges(messages, tokenizer, {"assistant"})
    full = _full_bytes(messages, tokenizer)
    assert len(ranges) == 1
    assert full[ranges[0][0]:ranges[0][1]].decode("utf-8") == "Hi there!"


def test_multi_turn(tokenizer):
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]
    ranges = find_content_byte_ranges(messages, tokenizer, {"assistant"})
    full = _full_bytes(messages, tokenizer)
    assert len(ranges) == 2
    assert full[ranges[0][0]:ranges[0][1]].decode("utf-8") == "First answer"
    assert full[ranges[1][0]:ranges[1][1]].decode("utf-8") == "Second answer"


def test_unicode_content(tokenizer):
    messages = [
        {"role": "user", "content": "Translate"},
        {"role": "assistant", "content": "こんにちは世界 🌍"},
    ]
    ranges = find_content_byte_ranges(messages, tokenizer, {"assistant"})
    full = _full_bytes(messages, tokenizer)
    assert len(ranges) == 1
    assert full[ranges[0][0]:ranges[0][1]].decode("utf-8") == "こんにちは世界 🌍"


def test_special_characters(tokenizer):
    content = '<tags> "quotes" $money\nnewline'
    messages = [
        {"role": "user", "content": "Test"},
        {"role": "assistant", "content": content},
    ]
    ranges = find_content_byte_ranges(messages, tokenizer, {"assistant"})
    full = _full_bytes(messages, tokenizer)
    assert len(ranges) == 1
    assert full[ranges[0][0]:ranges[0][1]].decode("utf-8") == content


def test_long_content(tokenizer):
    content = "hello " * 200
    messages = [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": content},
    ]
    ranges = find_content_byte_ranges(messages, tokenizer, {"assistant"})
    full = _full_bytes(messages, tokenizer)
    assert len(ranges) == 1
    assert full[ranges[0][0]:ranges[0][1]].decode("utf-8") == content


def test_empty_content(tokenizer):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": ""},
    ]
    ranges = find_content_byte_ranges(messages, tokenizer, {"assistant"})
    assert len(ranges) == 0


def test_user_role_in_save_roles(tokenizer):
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    ranges = find_content_byte_ranges(messages, tokenizer, {"user", "assistant"})
    full = _full_bytes(messages, tokenizer)
    assert len(ranges) == 2
    assert full[ranges[0][0]:ranges[0][1]].decode("utf-8") == "Question"
    assert full[ranges[1][0]:ranges[1][1]].decode("utf-8") == "Answer"


def test_no_matching_roles(tokenizer):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]
    ranges = find_content_byte_ranges(messages, tokenizer, {"system"})
    assert len(ranges) == 0


def test_merge_reasoning_inline():
    messages = [
        {"role": "user", "content": "Think about this"},
        {"role": "assistant", "content": "actual", "reasoning": "reasoning"},
    ]
    result = _merge_reasoning(messages, "inline")
    assert result[1]["content"] == "<think>reasoning</think>actual"
    assert "reasoning" not in result[1]


def test_merge_reasoning_field():
    messages = [
        {"role": "user", "content": "Think about this"},
        {"role": "assistant", "content": "actual", "reasoning": "reasoning"},
    ]
    result = _merge_reasoning(messages, "field")
    assert result[1]["content"] == "actual"
    assert result[1]["reasoning_content"] == "reasoning"
    assert "reasoning" not in result[1]


def test_merge_reasoning_no_reasoning():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = _merge_reasoning(messages, "inline")
    assert result[1]["content"] == "Hello"
    assert "reasoning" not in result[1]
    assert "reasoning_content" not in result[1]
