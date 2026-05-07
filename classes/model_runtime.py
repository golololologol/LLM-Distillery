from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from transformers import AutoTokenizer

from classes.formatter import SentinelFormatter
from classes.preprocessing import ModelParticipant


def _apply_mistral_regex_fix(tokenizer) -> None:
    """Manually apply the Mistral pre-tokenizer regex correction.

    Replicates the patch from
    ``transformers.PreTrainedTokenizerFast._patch_mistral_regex`` without
    the duplicate-kwarg bug that prevents passing ``fix_mistral_regex=True``
    through ``AutoTokenizer.from_pretrained``.
    """
    try:
        import tokenizers
    except ImportError:
        return
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return
    split_pretokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
            r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
            r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ),
        behavior="isolated",
    )
    current = backend.pre_tokenizer
    if isinstance(current, tokenizers.pre_tokenizers.Sequence):
        backend.pre_tokenizer[0] = split_pretokenizer
    else:
        if isinstance(current, tokenizers.pre_tokenizers.Metaspace):
            current = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [split_pretokenizer, current]
        )
    setattr(tokenizer, "fix_mistral_regex", True)


def load_tokenizer(model_path: str, tokenizer_kwargs: dict | None = None):
    """Load a tokenizer, isolating ``fix_mistral_regex`` from
    ``from_pretrained`` (transformers passes it twice internally, raising
    a TypeError) and applying the corresponding pre-tokenizer patch
    manually instead.
    """
    import logging
    kwargs = dict(tokenizer_kwargs or {})
    apply_mistral_fix = bool(kwargs.pop("fix_mistral_regex", False))
    # transformers prints a redundant "incorrect regex pattern" warning on
    # every Mistral load; we apply the patch manually so the user already
    # opted in. Suppress it for the duration of this call only.
    _tu_logger = logging.getLogger("transformers.tokenization_utils_base")
    _tt_logger = logging.getLogger("transformers.tokenization_utils_tokenizers")
    prev_levels = (_tu_logger.level, _tt_logger.level)
    if apply_mistral_fix:
        _tu_logger.setLevel(logging.ERROR)
        _tt_logger.setLevel(logging.ERROR)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    finally:
        if apply_mistral_fix:
            _tu_logger.setLevel(prev_levels[0])
            _tt_logger.setLevel(prev_levels[1])
    if apply_mistral_fix:
        _apply_mistral_regex_fix(tokenizer)
    return tokenizer


def build_tokenizer_formatter(model_path: str, config):
    tokenizer_kwargs = getattr(config, "tokenizer_kwargs", {}) or {}
    tokenizer = load_tokenizer(model_path, tokenizer_kwargs)
    return tokenizer, SentinelFormatter(
        tokenizer,
        chat_template=config.resolve_chat_template(),
        render_options=config.render_options,
        supports_reasoning=config.supports_reasoning,
        supports_tool_calls=config.supports_tool_calls,
        tokenizer_kwargs=tokenizer_kwargs,
    )


def build_participant(
    *,
    name: str,
    role: Literal["teacher", "student"],
    model_path: str,
    config,
) -> ModelParticipant:
    _, formatter = build_tokenizer_formatter(model_path, config)
    return ModelParticipant(
        name=name,
        role=role,
        formatter=formatter,
        context_len=config.context_len,
        segment_handling=config.effective_segments(),
    )


def build_teacher_participants(
    teachers: Mapping[str, object] | Iterable[tuple[str, object]],
    resolve_model_path,
) -> list[ModelParticipant]:
    items = teachers.items() if isinstance(teachers, Mapping) else teachers
    participants: list[ModelParticipant] = []
    for name_obj, cfg in items:
        model_path_value = getattr(cfg, "model_path", None)
        if cfg is None or model_path_value is None or getattr(cfg, "context_len", None) is None:
            continue
        try:
            model_path = resolve_model_path(model_path_value)
        except FileNotFoundError:
            continue
        participants.append(build_participant(
            name=str(name_obj),
            role="teacher",
            model_path=model_path,
            config=cfg,
        ))
    return participants


