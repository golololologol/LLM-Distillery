from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from classes.event_vocab import EventVocab, ResolvedSpecialsMap, SpecialsMap, compute_specials_hash


@dataclass(frozen=True)
class EventChannelSpec:
    """Resolved event-channel runtime state for one tokenizer/model."""

    vocab: EventVocab
    specials: SpecialsMap
    resolved: ResolvedSpecialsMap
    alphabet: tuple[str, ...]
    supported_mask: int
    specials_hash: str

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        specials_config: Mapping[str, Iterable[str]],
        extensions: Sequence[str] = (),
    ) -> "EventChannelSpec":
        vocab = EventVocab(extensions=tuple(extensions or ()))
        specials = SpecialsMap(vocab, specials_config)
        resolved = specials.resolve(tokenizer)
        return cls(
            vocab=vocab,
            specials=specials,
            resolved=resolved,
            alphabet=tuple(vocab.slots),
            supported_mask=int(resolved.supported_mask),
            specials_hash=compute_specials_hash(vocab, specials),
        )

    @classmethod
    def hash_for_config(
        cls,
        specials_config: Mapping[str, Iterable[str]],
        extensions: Sequence[str] = (),
        event_remap: Mapping[str, str] | None = None,
    ) -> str:
        vocab = EventVocab(extensions=tuple(extensions or ()))
        specials = SpecialsMap(vocab, specials_config)
        return compute_specials_hash(vocab, specials, event_remap=event_remap)

