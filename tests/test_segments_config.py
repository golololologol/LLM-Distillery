import pytest
from typing import Any
from pydantic import ValidationError

from classes.args import SegmentConfig, StudentConfig, TeacherConfig, _parse_teacher_toml


def _teacher(**kwargs):
    base = dict(model_path="/tmp/m", backend_type="exl2", context_len=2048)
    base.update(kwargs)
    return TeacherConfig(**base)


def _student(**kwargs):
    base = dict(
        model_path="/tmp/m", freeze_layers=[], save_final_training_state=False,
        context_len=2048,
    )
    base.update(kwargs)
    return StudentConfig(**base)


def test_segment_config_only_allows_handling_field():
    seg = SegmentConfig(handling="active")
    assert seg.handling == "active"
    with pytest.raises(ValidationError):
        extra_field_config: dict[str, Any] = {"handling": "active", "open_delimiter": "<a>"}
        SegmentConfig(**extra_field_config)


def test_no_supports_reasoning_rejects_reasoning_segment():
    with pytest.raises(ValidationError, match="does not support segment 'reasoning'"):
        _teacher(supports_reasoning=False, segments={"reasoning": {"handling": "active"}})


def test_no_supports_tool_calls_rejects_tool_call_segment():
    with pytest.raises(ValidationError, match="does not support segment 'tool_call'"):
        _teacher(supports_tool_calls=False, segments={"tool_call": {"handling": "active"}})


def test_student_no_reasoning_rejects_reasoning_segment():
    with pytest.raises(ValidationError, match="does not support segment 'reasoning'"):
        _student(supports_reasoning=False, segments={"reasoning": {"handling": "active"}})


def test_teacher_accepts_reasoning_and_tool_call_by_default():
    tc = _teacher(segments={
        "reasoning": {"handling": "active"},
        "tool_call": {"handling": "active"},
    })
    assert "reasoning" in tc.segments
    assert "tool_call" in tc.segments


def test_teacher_rejects_unknown_handling():
    with pytest.raises(ValidationError, match="'active', 'context' or 'disabled'"):
        _teacher(segments={"content": {"handling": "train"}})


def test_student_rejects_unknown_handling():
    with pytest.raises(ValidationError, match="'active', 'context' or 'disabled'"):
        _student(segments={"content": {"handling": "collect"}})


def test_student_rejects_disabling_content():
    with pytest.raises(ValidationError, match="cannot be disabled"):
        _student(segments={"content": {"handling": "disabled"}})


def test_teacher_effective_segments_defaults():
    tc = TeacherConfig()
    assert tc.effective_segments() == {"content": "active"}


def test_student_effective_segments_defaults():
    sc = _student()
    assert sc.effective_segments() == {"content": "active"}


def test_parse_teacher_toml_handles_backend_and_segments():
    raw = {
        "model_path": "/tmp/m",
        "context_len": 2048,
        "exl2": {"cache_bits": 4},
        "segments": {"reasoning": {"handling": "active"}},
    }
    tc = _parse_teacher_toml(raw, "test.toml")
    assert tc.backend_type == "exl2"
    assert tc.segments["reasoning"].handling == "active"

def test_coerce_segments_from_raw_dicts():
    tc = _teacher(segments={"reasoning": {"handling": "active"}})
    assert isinstance(tc.segments["reasoning"], SegmentConfig)


def test_render_options_pass_through():
    tc = _teacher(render_options={"enable_thinking": True})
    assert tc.render_options == {"enable_thinking": True}


def test_chat_template_path_resolved(tmp_path):
    p = tmp_path / "tpl.jinja"
    p.write_text("{% for m in messages %}X{% endfor %}", encoding="utf-8")
    tc = _teacher(chat_template_path=str(p))
    assert tc.resolve_chat_template() == "{% for m in messages %}X{% endfor %}"


def test_chat_template_inline_resolved():
    tc = _teacher(chat_template="INLINE")
    assert tc.resolve_chat_template() == "INLINE"
