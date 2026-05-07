import multiprocessing
import os
import sys
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Literal, cast

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from classes.data_classes import ConvoProcessed
from classes.formatter import SentinelFormatter, _crop_byte_end
from utils.dataset_utils import compute_source_sha


_worker_formatter: SentinelFormatter | None = None
_worker_spec: tuple | None = None


def _make_formatter(model_path, chat_template, render_options, supports_reasoning, supports_tool_calls, tokenizer_kwargs=None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import logging
    from classes.model_runtime import load_tokenizer
    # Suppress the spurious "model of type X to instantiate model of type Y"
    # warning that fires when loading tokenizers for architectures (e.g.
    # gemma4) that transformers doesn't fully register. It is harmless for
    # tokenizer-only use and would otherwise spam one line per worker process.
    _tf_logger = logging.getLogger("transformers.configuration_utils")
    _prev_level = _tf_logger.level
    _tf_logger.setLevel(logging.ERROR)
    try:
        tok = load_tokenizer(model_path, tokenizer_kwargs)
    finally:
        _tf_logger.setLevel(_prev_level)

    return SentinelFormatter(
        tok,
        chat_template=chat_template,
        render_options=render_options,
        supports_reasoning=supports_reasoning,
        supports_tool_calls=supports_tool_calls,
    )


def _ensure_worker(spec):
    global _worker_formatter, _worker_spec
    if _worker_spec != spec:
        model_path, chat_template, render_options_items, supports_reasoning, supports_tool_calls, tokenizer_kwargs_items = spec
        _worker_formatter = _make_formatter(
            model_path, chat_template, dict(render_options_items),
            supports_reasoning, supports_tool_calls, dict(tokenizer_kwargs_items),
        )
        _worker_spec = spec
    return _worker_formatter


def _process_one_sample(formatter, raw, idx, context_len, save_roles, pad_token_id, segment_handling):
    sample_id = raw.get("id", idx)
    messages = raw["messages"]

    text, tokens, segments, cropped, _crop_end, anchors = formatter.render(
        messages, save_roles=save_roles, context_len=context_len,
        segment_handling=segment_handling,
    )

    content_sha = compute_source_sha(messages, save_roles)
    length = len(tokens)

    padding = context_len - length
    if padding > 0:
        tokens = tokens + [pad_token_id] * padding

    return ConvoProcessed(
        origin_convo_id=sample_id,
        tokens=np.array(tokens, dtype=np.int32),
        segments=segments,
        content_sha=content_sha,
        formatted_text=text,
        padding=padding,
        cropped=cropped,
        length=length,
        anchors=anchors,
    )


def _worker_task(spec, raw, idx, context_len, save_roles, pad_token_id, segment_handling):
    formatter = _ensure_worker(spec)
    return _process_one_sample(formatter, raw, idx, context_len, save_roles, pad_token_id, segment_handling)


_PROCESS_THRESHOLD = 500 if multiprocessing.get_start_method() == "fork" else 3000


def _formatter_spec(formatter: SentinelFormatter) -> tuple:
    model_path = formatter.tokenizer.name_or_path
    chat_template = formatter._chat_template
    render_items = tuple(sorted(formatter.render_options.items()))
    tokenizer_kwargs_items = tuple(sorted((formatter.tokenizer_kwargs or {}).items()))
    return (model_path, chat_template, render_items, formatter.supports_reasoning, formatter.supports_tool_calls, tokenizer_kwargs_items)


def preprocess_samples(
    raw_samples: list[dict],
    formatter: SentinelFormatter,
    context_len: int,
    save_roles: set[str] | None = None,
    segment_handling: dict[str, str] | None = None,
) -> list[ConvoProcessed]:
    if save_roles is None:
        save_roles = {"assistant"}

    tokenizer = formatter.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    spec = _formatter_spec(formatter)

    global _worker_formatter, _worker_spec
    _worker_formatter = formatter
    _worker_spec = spec

    n_samples = len(raw_samples)
    n_workers = min(cpu_count(), n_samples)
    prefer = "processes" if n_samples >= _PROCESS_THRESHOLD else "threads"

    gen = Parallel(
        n_jobs=n_workers, batch_size="auto", prefer=prefer,
        return_as="generator_unordered",
    )(
        delayed(_worker_task)(spec, raw, idx, context_len, save_roles, pad_token_id, segment_handling)
        for idx, raw in enumerate(raw_samples)
    )

    results = cast(
        list[ConvoProcessed],
        list(tqdm(gen, total=n_samples, desc="  Preprocessing", unit="samples", leave=False, smoothing=0.06)),
    )
    filtered = [r for r in results if r.segments]
    if len(filtered) < len(results):
        dropped_ids = [r.origin_convo_id for r in results if not r.segments]
        print(f"  Filtered out {len(results) - len(filtered)} samples (IDs: {dropped_ids}) with no content in "
              f"save_roles={save_roles} at the given context length.\n"
              f"  Consider increasing context_len or adjusting save_roles if this is unexpected.")
    return filtered


@dataclass(frozen=True)
class ModelParticipant:
    name: str
    formatter: SentinelFormatter
    context_len: int
    role: Literal["teacher", "student"]
    segment_handling: dict[str, str] | None = None


@dataclass(frozen=True)
class SampleEligibility:
    sample_idx: int
    eligible_teachers: frozenset[str]
    student_ok: bool
    reasons: tuple[str, ...]


_eligibility_cache: dict = {}


def _has_tool_calls(messages: list[dict]) -> bool:
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return True
    return False


def _quick_token_count(formatter, messages) -> int:
    """Cheap token count for a conversation without sentinel splicing.

    Equivalent in token count to what ``formatter.render`` would produce
    (sentinels are byte-spliced out before final tokenization), but skips
    the expensive sentinel-find / splice / retokenize round-trip used by
    :meth:`SentinelFormatter.render`. Only used as a fast pre-check in
    eligibility computation: when ``count <= context_len`` we know cropping
    cannot drop any message.
    """
    try:
        text = formatter._apply_template(_render_msgs_for_count(messages, formatter), tokenize=False)
    except Exception:
        return -1
    if not isinstance(text, str):
        return -1
    try:
        ids = formatter.tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return -1
    return len(ids) if ids is not None else -1


def _render_msgs_for_count(messages, formatter):
    """Build the same ``render_msgs`` shape that ``_build_probe`` produces, but
    using the original content (no sentinel substitution). Token count of the
    template applied to this is identical to the spliced-back result modulo
    sentinel collisions, which the formatter would have already rejected.
    """
    out = []
    for m in messages:
        role = m.get("role")
        r = {"role": role}
        content = m.get("content")
        if content is not None:
            r["content"] = content
        if role == "assistant":
            if formatter.supports_reasoning and m.get("reasoning") is not None:
                r["reasoning"] = m["reasoning"]
            if formatter.supports_tool_calls and m.get("tool_calls"):
                r["tool_calls"] = list(m["tool_calls"])
        elif role == "tool" and m.get("tool_call_id"):
            r["tool_call_id"] = m["tool_call_id"]
        out.append(r)
    return out


def _has_save_role_content(messages, save_roles) -> bool:
    """Return True if at least one message has role in ``save_roles`` and a
    non-empty content body. Samples that fail this would render to zero
    content bytes and produce empty distributions downstream."""
    for m in messages:
        if m.get("role") not in save_roles:
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            return True
        # tool_calls / reasoning are also collected payloads; treat assistant
        # messages with either as having content.
        if m.get("tool_calls") or m.get("reasoning"):
            return True
    return False


def _lost_msg_ids(participant, messages, save_roles):
    fmt = participant.formatter

    # Fast path: most conversations fit comfortably. A direct
    # apply_chat_template(tokenize=True) is ~5-10x faster than the full
    # sentinel render and tells us the same thing for the common case.
    quick = _quick_token_count(fmt, messages)
    if 0 <= quick <= participant.context_len:
        return []

    text, tokens, full_segs, _, _, _ = fmt.render(
        messages,
        save_roles=save_roles,
        context_len=sys.maxsize,
        segment_handling=participant.segment_handling,
    )
    if len(tokens) <= participant.context_len:
        return []
    crop_byte_end = _crop_byte_end(fmt.tokenizer, text, tokens, participant.context_len)
    full_ids = {s.msg_idx for s in full_segs}
    kept_ids = {s.msg_idx for s in full_segs if s.byte_start < crop_byte_end}
    return sorted(full_ids - kept_ids)


# ---------------------------------------------------------------------------
# Parallel eligibility computation
# ---------------------------------------------------------------------------
#
# Each worker process keeps a small dict of (spec -> SentinelFormatter) so
# that all participant formatters are loaded once on first use and reused
# across thousands of samples in that worker. Specs are cheap to pickle;
# formatters (which hold tokenizers) are not, so they're rebuilt per worker.

_worker_formatters_by_spec: dict = {}


def _ensure_worker_multi(spec, p_meta):
    fmt = _worker_formatters_by_spec.get(spec)
    if fmt is None:
        model_path, chat_template, render_options_items, supports_reasoning, supports_tool_calls, tokenizer_kwargs_items = spec
        fmt = _make_formatter(
            model_path, chat_template, dict(render_options_items),
            supports_reasoning, supports_tool_calls, dict(tokenizer_kwargs_items),
        )
        _worker_formatters_by_spec[spec] = fmt
    # Build a lightweight participant-shaped object the existing helper accepts.
    return _ParticipantView(
        name=p_meta["name"],
        formatter=fmt,
        context_len=p_meta["context_len"],
        role=p_meta["role"],
        segment_handling=p_meta["segment_handling"],
    )


@dataclass(frozen=True)
class _ParticipantView:
    name: str
    formatter: SentinelFormatter
    context_len: int
    role: str
    segment_handling: dict[str, str] | None


def _eligibility_worker(participant_specs, sample, idx, save_roles_t, teacher_names):
    save_roles = set(save_roles_t)
    messages = sample["messages"]

    # Global pre-check: a sample with no save_role content bytes produces an
    # empty distribution downstream (zero content_byte_ranges), which is both
    # useless for training and would crash SharedMemory(size=0). Reject it
    # for every participant.
    if not _has_save_role_content(messages, save_roles):
        return idx, frozenset(), False, ("no save_role content bytes",)

    has_tc = _has_tool_calls(messages)
    eligible = set(teacher_names)
    student_ok = True
    reasons: list[str] = []

    if has_tc:
        for p_meta in participant_specs:
            handling = p_meta["segment_handling"] or {}
            needs_tool_calls = handling.get("tool_call", "active") != "disabled"
            if needs_tool_calls and not p_meta["supports_tool_calls"]:
                reasons.append(f"{p_meta['role']} {p_meta['name']}: no tool_call support")
                if p_meta["role"] == "teacher":
                    eligible.discard(p_meta["name"])
                else:
                    student_ok = False

    for p_meta in participant_specs:
        if p_meta["role"] == "teacher" and p_meta["name"] not in eligible:
            continue
        if p_meta["role"] == "student" and not student_ok:
            continue
        p_view = _ensure_worker_multi(p_meta["spec"], p_meta)
        lost = _lost_msg_ids(p_view, messages, save_roles)
        if lost:
            reasons.append(f"{p_meta['role']} {p_meta['name']}: context_len too short, msg_idx {lost[0]} dropped")
            if p_meta["role"] == "teacher":
                eligible.discard(p_meta["name"])
            else:
                student_ok = False

    return idx, frozenset(eligible), student_ok, tuple(reasons)


def compute_sample_eligibility(
    samples: list[dict],
    participants: list[ModelParticipant],
    *,
    save_roles: set[str] | None = None,
) -> list[SampleEligibility]:
    if save_roles is None:
        save_roles = {"assistant"}
    participants = list(participants)
    teacher_names = frozenset(p.name for p in participants if p.role == "teacher")
    n_samples = len(samples)
    if n_samples == 0:
        return []

    # Build picklable per-participant metadata (no formatter / tokenizer).
    participant_specs = [
        {
            "name": p.name,
            "role": p.role,
            "context_len": p.context_len,
            "segment_handling": p.segment_handling,
            "supports_tool_calls": p.formatter.supports_tool_calls,
            "spec": _formatter_spec(p.formatter),
        }
        for p in participants
    ]

    save_roles_t = tuple(sorted(save_roles))

    n_workers = min(cpu_count(), max(1, n_samples))
    prefer = "processes" if n_samples >= _PROCESS_THRESHOLD else "threads"

    if prefer == "threads":
        # Threads share the parent's formatter objects. Build a participant
        # list that thread workers can use directly. Also reuse the
        # ``_eligibility_cache`` here for repeat-call efficiency (the
        # process-parallel path skips the cache; the per-sample work is
        # already amortised across worker cores).
        def _fp(fmt):
            fn = getattr(fmt, "options_fingerprint", None)
            return fn() if callable(fn) else ()

        def _handling_key(handling):
            if not handling:
                return ()
            return tuple(sorted(handling.items()))

        key_set = frozenset(
            (p.name, p.context_len, _fp(p.formatter), _handling_key(p.segment_handling))
            for p in participants
        )
        save_key = frozenset(save_roles)

        results_raw = []
        for idx, sample in tqdm(enumerate(samples), total=n_samples, desc="  Eligibility", unit="samples", leave=False, smoothing=0.06):
            messages = sample["messages"]
            sha = compute_source_sha(messages, save_roles)
            cache_key = (sha, key_set, save_key)
            cached = _eligibility_cache.get(cache_key)
            if cached is not None:
                results_raw.append((idx, cached.eligible_teachers, cached.student_ok, cached.reasons))
                continue

            # Global pre-check: zero save_role content -> useless sample.
            if not _has_save_role_content(messages, save_roles):
                elig = SampleEligibility(
                    sample_idx=idx,
                    eligible_teachers=frozenset(),
                    student_ok=False,
                    reasons=("no save_role content bytes",),
                )
                _eligibility_cache[cache_key] = elig
                results_raw.append((idx, elig.eligible_teachers, elig.student_ok, elig.reasons))
                continue

            has_tc = _has_tool_calls(messages)
            eligible = set(teacher_names)
            student_ok = True
            reasons: list[str] = []
            if has_tc:
                for p in participants:
                    needs_tool_calls = (p.segment_handling or {}).get("tool_call", "active") != "disabled"
                    if needs_tool_calls and not p.formatter.supports_tool_calls:
                        reasons.append(f"{p.role} {p.name}: no tool_call support")
                        if p.role == "teacher":
                            eligible.discard(p.name)
                        else:
                            student_ok = False
            for p in participants:
                if p.role == "teacher" and p.name not in eligible:
                    continue
                if p.role == "student" and not student_ok:
                    continue
                lost = _lost_msg_ids(p, messages, save_roles)
                if lost:
                    reasons.append(f"{p.role} {p.name}: context_len too short, msg_idx {lost[0]} dropped")
                    if p.role == "teacher":
                        eligible.discard(p.name)
                    else:
                        student_ok = False
            elig = SampleEligibility(
                sample_idx=idx,
                eligible_teachers=frozenset(eligible),
                student_ok=student_ok,
                reasons=tuple(reasons),
            )
            _eligibility_cache[cache_key] = elig
            results_raw.append((idx, elig.eligible_teachers, elig.student_ok, elig.reasons))
    else:
        gen = Parallel(
            n_jobs=n_workers, batch_size="auto", prefer="processes",
            return_as="generator_unordered",
        )(
            delayed(_eligibility_worker)(participant_specs, sample, idx, save_roles_t, teacher_names)
            for idx, sample in enumerate(samples)
        )
        results_raw = cast(
            list,
            list(tqdm(gen, total=n_samples, desc="  Eligibility", unit="samples", leave=False, smoothing=0.06)),
        )

    # Restore index order.
    results_raw.sort(key=lambda r: r[0])
    return [
        SampleEligibility(
            sample_idx=idx,
            eligible_teachers=eligible,
            student_ok=student_ok,
            reasons=reasons,
        )
        for idx, eligible, student_ok, reasons in results_raw
    ]
