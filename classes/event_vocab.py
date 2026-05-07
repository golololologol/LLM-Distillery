"""Event vocabulary, specials map, and helpers (plan v8 §2-§4, §12).

The event vocabulary is a frozen core ontology of 15 model-output control events,
optionally extended with named slots declared per-collection-run.

Public surface:
    - ``CORE_SLOTS``       : tuple of 15 canonical slot names (frozen).
    - ``EventVocab``       : indexable alphabet object (core + extensions).
    - ``SpecialsMap``      : raw ``{slot_name: [native_string,...]}`` mapping.
    - ``ResolvedSpecialsMap`` : ``{slot_idx: [token_id,...]}`` after tokenizer
      resolution, with derived per-slot ``supported_mask``.
    - ``compute_specials_hash`` : sha256 of canonicalised specials map +
      alphabet + remap, used for HDF5 cache invalidation.
    - ``validate_remap``  : checks that every key/value of the student
      ``[event_remap]`` table refers to a real slot or the ``"drop"`` sentinel.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Mapping, Sequence


# Frozen core ontology - DO NOT REORDER.
CORE_SLOTS: tuple[str, ...] = (
    "E_NONE",                    # 0
    "E_END_TURN",                # 1
    "E_END_MESSAGE",             # 2
    "E_BEGIN_REASONING",         # 3
    "E_END_REASONING",           # 4
    "E_BEGIN_ANSWER",            # 5
    "E_END_ANSWER",              # 6
    "E_BEGIN_TOOL_CALL",         # 7
    "E_END_TOOL_CALL",           # 8
    "E_BEGIN_TOOL_CALL_BLOCK",   # 9
    "E_END_TOOL_CALL_BLOCK",     # 10
    "E_TOOL_NAME_BOUNDARY",      # 11
    "E_BEGIN_COMMENTARY",        # 12
    "E_END_COMMENTARY",          # 13
    "E_OTHER_SPECIAL",           # 14
)

E_NONE_IDX = 0
E_OTHER_SPECIAL_IDX = 14


class EventVocab:
    """Ordered alphabet: 15 core slots followed by user-declared extensions."""

    __slots__ = ("_slots", "_index")

    def __init__(self, extensions: Sequence[str] = ()):
        ext_tuple = tuple(extensions or ())
        seen: set[str] = set(CORE_SLOTS)
        for e in ext_tuple:
            if not isinstance(e, str) or not e:
                raise ValueError(f"extension slot name must be a non-empty string, got {e!r}")
            if e in seen:
                raise ValueError(f"extension slot {e!r} duplicates an existing slot")
            seen.add(e)
        self._slots: tuple[str, ...] = CORE_SLOTS + ext_tuple
        self._index: dict[str, int] = {n: i for i, n in enumerate(self._slots)}

    @property
    def slots(self) -> tuple[str, ...]:
        return self._slots

    @property
    def extensions(self) -> tuple[str, ...]:
        return self._slots[len(CORE_SLOTS):]

    @property
    def size(self) -> int:
        return len(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"unknown event slot {name!r}; known: {self._slots}") from None

    def has(self, name: str) -> bool:
        return name in self._index

    def __iter__(self):
        return iter(self._slots)

    def __eq__(self, other) -> bool:
        return isinstance(other, EventVocab) and other._slots == self._slots

    def __repr__(self) -> str:
        return f"EventVocab(size={self.size}, extensions={self.extensions})"


def _canonical_specials_dict(vocab: EventVocab, raw: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    """Validate slot names and freeze ordering. Drops empty entries."""
    out: dict[str, list[str]] = {}
    for slot_name, strings in raw.items():
        if not vocab.has(slot_name):
            raise ValueError(f"specials map references unknown slot {slot_name!r}")
        if isinstance(strings, str):
            strings = [strings]
        items = [s for s in strings if s]
        if not items:
            continue
        out[slot_name] = list(items)
    return out


class SpecialsMap:
    """Per-model specials map, *string-valued* (pre tokenizer resolution).

    Keys are :class:`EventVocab` slot names. Values are lists of native special
    token strings expected to appear verbatim in the tokenizer.
    """

    __slots__ = ("vocab", "_data")

    def __init__(self, vocab: EventVocab, mapping: Mapping[str, Iterable[str]] | None = None):
        self.vocab = vocab
        self._data = _canonical_specials_dict(vocab, mapping or {})

    @property
    def slots(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._data.items()}

    def to_canonical(self) -> dict[str, list[str]]:
        return {k: list(self._data[k]) for k in sorted(self._data)}

    def resolve(self, tokenizer) -> "ResolvedSpecialsMap":
        """Resolve every string to a single token id against ``tokenizer``.

        Multi-token resolutions (a string that the tokenizer encodes as more
        than one piece) are rejected loudly. Strings that the tokenizer does
        not know are also rejected.

        Unmapped specials (ids that are flagged as special by the tokenizer
        but not named anywhere in the map) are absorbed into
        :data:`E_OTHER_SPECIAL_IDX`.
        """
        slot_to_ids: dict[int, list[int]] = {}
        named_ids: set[int] = set()
        problems: list[str] = []
        for slot_name, strings in self._data.items():
            slot_idx = self.vocab.index(slot_name)
            ids: list[int] = []
            for s in strings:
                tid = _resolve_single_token(tokenizer, s)
                if tid is None:
                    problems.append(
                        f"  - slot {slot_name!r}: token {s!r} is not a single piece in this tokenizer"
                    )
                    continue
                ids.append(int(tid))
                named_ids.add(int(tid))
            if ids:
                slot_to_ids[slot_idx] = ids
        if problems:
            raise ValueError(
                "specials map contains entries that do not resolve to single token ids:\n"
                + "\n".join(problems)
            )

        # Absorb any remaining tokenizer-flagged specials/added tokens into
        # E_OTHER_SPECIAL.
        leftover = _collect_special_ids(tokenizer) - named_ids
        if leftover:
            slot_to_ids.setdefault(E_OTHER_SPECIAL_IDX, []).extend(sorted(leftover))

        return ResolvedSpecialsMap(self.vocab, slot_to_ids)


def _resolve_single_token(tokenizer, s: str) -> int | None:
    """Return the id if ``s`` resolves to exactly one token, else ``None``.

    Tries ``convert_tokens_to_ids`` first (works for tokens registered as
    specials/added tokens) and falls back to ``encode(add_special_tokens=False)``.
    """
    cv = getattr(tokenizer, "convert_tokens_to_ids", None)
    if cv is not None:
        try:
            tid = cv(s)
        except Exception:
            tid = None
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
    try:
        ids = tokenizer.encode(s, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(s)
    if isinstance(ids, list) and len(ids) == 1:
        return int(ids[0])
    return None


def _collect_special_ids(tokenizer) -> set[int]:
    """All ids the tokenizer marks as special / added tokens.

    HuggingFace exposes only tokens registered via ``added_tokens_decoder`` /
    ``all_special_ids``. Many SentencePiece tokenizers (notably Gemma) ship
    thousands of reserved vocabulary slots like ``<unusedN>`` that the model
    treats as ordinary vocab entries but that are *never* legitimate content
    predictions — leaving them in the byte channel leaks measurable mass onto
    byte ``0x3C`` (``<``). We therefore augment the HF set with two extra
    sources:

    1. SentencePiece ``IsControl`` / ``IsUnused`` flags when ``sp_model`` is
       reachable on the tokenizer.
    2. A conservative regex sweep over the full vocab that matches well-known
       reserved-slot surface forms (``<unused\\d+>``, ``<reserved\\S*>``,
       ``<extra_id_\\d+>``).
    """
    out: set[int] = set()
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tid in added.keys():
        try:
            out.add(int(tid))
        except (TypeError, ValueError):
            continue
    special_ids = getattr(tokenizer, "all_special_ids", None) or []
    for tid in special_ids:
        try:
            if tid is not None and int(tid) >= 0:
                out.add(int(tid))
        except (TypeError, ValueError):
            continue

    # SentencePiece control / unused token types (avoids per-tokenizer
    # heuristics where the SP model is exposed). ``IsByte`` is *not* masked:
    # byte-fallback tokens are legitimate content predictions.
    sp = getattr(tokenizer, "sp_model", None)
    if sp is not None:
        try:
            for tid in range(sp.get_piece_size()):
                try:
                    if sp.IsControl(tid) or sp.IsUnused(tid):
                        out.add(int(tid))
                except Exception:
                    continue
        except Exception:
            pass

    # Surface-form sweep for fast tokenizers that don't expose ``sp_model``.
    # Patterns are deliberately strict to avoid masking legitimate vocab
    # entries that happen to start with '<'.
    reserved_patterns = (
        re.compile(r"^<unused\d+>$"),
        re.compile(r"^<reserved[_\d]\S*>$"),
        re.compile(r"^<extra_id_\d+>$"),
    )
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)
        except Exception:
            vocab_size = None
    if convert is not None and vocab_size:
        try:
            tokens = convert(list(range(int(vocab_size))))
            for tid, surface in enumerate(tokens):
                if not isinstance(surface, str):
                    continue
                for pat in reserved_patterns:
                    if pat.match(surface):
                        out.add(tid)
                        break
        except Exception:
            pass
    return out


class ResolvedSpecialsMap:
    """Per-model specials map after tokenizer resolution.

    Holds ``{slot_idx: [token_id,...]}`` and exposes ``supported_mask`` (a
    bitmask flagging which slots have at least one resolved token id).
    """

    __slots__ = ("vocab", "_slot_to_ids", "supported_mask")

    def __init__(self, vocab: EventVocab, slot_to_ids: Mapping[int, Iterable[int]]):
        self.vocab = vocab
        cleaned: dict[int, list[int]] = {}
        mask = 0
        for slot_idx, ids in slot_to_ids.items():
            slot_idx = int(slot_idx)
            if slot_idx < 0 or slot_idx >= vocab.size:
                raise ValueError(f"slot index {slot_idx} out of range for vocab of size {vocab.size}")
            id_list = sorted({int(t) for t in ids})
            if not id_list:
                continue
            cleaned[slot_idx] = id_list
            mask |= (1 << slot_idx)
        self._slot_to_ids = cleaned
        self.supported_mask = int(mask)

    def slot_ids(self, slot_idx: int) -> list[int]:
        return list(self._slot_to_ids.get(int(slot_idx), ()))

    def items(self):
        return self._slot_to_ids.items()

    def all_event_token_ids(self) -> list[int]:
        out: list[int] = []
        for ids in self._slot_to_ids.values():
            out.extend(ids)
        return sorted(set(out))

    def __bool__(self) -> bool:
        return bool(self._slot_to_ids)

    def __repr__(self) -> str:
        return f"ResolvedSpecialsMap(slots={sorted(self._slot_to_ids)}, mask={bin(self.supported_mask)})"


def compute_specials_hash(
    vocab: EventVocab,
    specials: SpecialsMap,
    event_remap: Mapping[str, str] | None = None,
) -> str:
    """Stable hash over the canonicalised specials map + alphabet + remap.

    Joined into the existing collection params hash chain so that any change
    invalidates HDF5 caches exactly like a chat-template change.
    """
    payload = {
        "alphabet": list(vocab.slots),
        "specials": specials.to_canonical() if isinstance(specials, SpecialsMap) else dict(specials or {}),
        "remap": dict(sorted((event_remap or {}).items())),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def validate_remap(vocab: EventVocab, remap: Mapping[str, str] | None) -> None:
    """Raise if remap references an unknown slot or an unsupported value.

    Allowed values: ``"drop"`` or any other slot name in ``vocab``.
    """
    if not remap:
        return
    for src, dst in remap.items():
        if not vocab.has(src):
            raise ValueError(f"event_remap source {src!r} is not in the event alphabet")
        if dst == "drop":
            continue
        if not vocab.has(dst):
            raise ValueError(f"event_remap target {dst!r} is not in the event alphabet")
        if src == dst:
            raise ValueError(f"event_remap of {src!r} -> {dst!r} is a no-op; remove it")
