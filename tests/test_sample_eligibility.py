import pytest

from classes.formatter import SentinelFormatter
from classes.preprocessing import (
    ModelParticipant,
    _eligibility_cache,
    compute_sample_eligibility,
)
from utils.dataset_utils import DatasetCanonicalizer


@pytest.fixture(autouse=True)
def clear_cache():
    _eligibility_cache.clear()
    yield
    _eligibility_cache.clear()


@pytest.fixture(scope="module")
def fmt_with_tools(tokenizer):
    return SentinelFormatter(tokenizer, supports_reasoning=True, supports_tool_calls=True)


@pytest.fixture(scope="module")
def fmt_no_tools(tokenizer):
    return SentinelFormatter(tokenizer, supports_reasoning=False, supports_tool_calls=False)


def _canon(messages):
    return DatasetCanonicalizer().canonicalize_messages(messages)


def _sample(messages, id=0):
    return {"id": id, "messages": _canon(messages)}


def test_no_tool_call_fits_all(fmt_with_tools, fmt_no_tools):
    sample = _sample([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    participants = [
        ModelParticipant("T1", fmt_with_tools, 512, "teacher"),
        ModelParticipant("T2", fmt_no_tools, 512, "teacher"),
        ModelParticipant("S", fmt_with_tools, 512, "student"),
    ]
    results = compute_sample_eligibility([sample], participants)
    assert results[0].eligible_teachers == frozenset({"T1", "T2"})
    assert results[0].student_ok is True


def test_tool_call_sample_excludes_non_tool_teachers(fmt_with_tools, fmt_no_tools):
    sample = _sample([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "checking",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"city":"x"}'}}]},
    ])
    participants = [
        ModelParticipant("T1", fmt_with_tools, 1024, "teacher"),
        ModelParticipant("T2", fmt_no_tools, 1024, "teacher"),
        ModelParticipant("S", fmt_with_tools, 1024, "student"),
    ]
    results = compute_sample_eligibility([sample], participants)
    assert "T1" in results[0].eligible_teachers
    assert "T2" not in results[0].eligible_teachers
    assert results[0].student_ok is True
    assert any("T2" in r for r in results[0].reasons)


def test_tool_call_student_without_support_blocks(fmt_with_tools, fmt_no_tools):
    sample = _sample([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
    ])
    participants = [
        ModelParticipant("T1", fmt_with_tools, 1024, "teacher"),
        ModelParticipant("S", fmt_no_tools, 1024, "student"),
    ]
    results = compute_sample_eligibility([sample], participants)
    assert results[0].student_ok is False


def test_context_too_short_drops_saved_message(fmt_with_tools):
    big = "word " * 400
    sample = _sample([
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "final saved answer"},
    ])
    participants = [
        ModelParticipant("Big", fmt_with_tools, 4096, "teacher"),
        ModelParticipant("Small", fmt_with_tools, 256, "teacher"),
        ModelParticipant("S", fmt_with_tools, 4096, "student"),
    ]
    results = compute_sample_eligibility([sample], participants)
    assert "Big" in results[0].eligible_teachers
    assert "Small" not in results[0].eligible_teachers
    assert results[0].student_ok is True


def test_all_teachers_disqualified(fmt_with_tools):
    big = "word " * 400
    sample = _sample([
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "final"},
    ])
    participants = [
        ModelParticipant("T1", fmt_with_tools, 256, "teacher"),
        ModelParticipant("T2", fmt_with_tools, 256, "teacher"),
        ModelParticipant("S", fmt_with_tools, 4096, "student"),
    ]
    results = compute_sample_eligibility([sample], participants)
    assert results[0].eligible_teachers == frozenset()


def test_cache_hit_returns_same_result(fmt_with_tools):
    sample = _sample([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ])
    participants = [
        ModelParticipant("T1", fmt_with_tools, 512, "teacher"),
        ModelParticipant("S", fmt_with_tools, 512, "student"),
    ]
    r1 = compute_sample_eligibility([sample], participants)
    r2 = compute_sample_eligibility([sample], participants)
    assert r1[0].eligible_teachers == r2[0].eligible_teachers
    assert r1[0].student_ok == r2[0].student_ok
    assert r1[0].reasons == r2[0].reasons
    assert len(_eligibility_cache) == 1


def test_options_fingerprint_distinguishes_render_options(tokenizer):
    fmt_a = SentinelFormatter(tokenizer)
    fmt_b = SentinelFormatter(tokenizer, render_options={"enable_thinking": False})
    assert fmt_a.options_fingerprint() != fmt_b.options_fingerprint()


def test_options_fingerprint_distinguishes_chat_template(tokenizer):
    fmt_a = SentinelFormatter(tokenizer, supports_reasoning=False, supports_tool_calls=False)
    fmt_b = SentinelFormatter(
        tokenizer,
        chat_template="{% for m in messages %}{{ m['role'] + ':' + m['content'] }}{% endfor %}",
        supports_reasoning=False,
        supports_tool_calls=False,
    )
    assert fmt_a.options_fingerprint() != fmt_b.options_fingerprint()


def test_disabled_segment_not_counted_for_eligibility(fmt_with_tools):
    huge_args = '{"x":"' + "y" * 1500 + '"}'
    sample = _sample([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ok",
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "f", "arguments": huge_args}}]},
        {"role": "tool", "content": "done", "tool_call_id": "1"},
        {"role": "assistant", "content": "final"},
    ])
    participants = [
        ModelParticipant(
            "ActiveTools",
            fmt_with_tools,
            128,
            "teacher",
            {"content": "active", "tool_call": "active"},
        ),
        ModelParticipant(
            "DisabledTools",
            fmt_with_tools,
            128,
            "teacher",
            {"content": "active", "tool_call": "disabled"},
        ),
        ModelParticipant(
            "S",
            fmt_with_tools,
            512,
            "student",
            {"content": "active", "tool_call": "disabled"},
        ),
    ]
    result = compute_sample_eligibility([sample], participants)[0]
    assert "DisabledTools" in result.eligible_teachers
    assert "ActiveTools" not in result.eligible_teachers
