"""Tests for the frozen event vocabulary + specials map (plan v8 §2-§4, §11)."""
import pytest

from classes.event_vocab import (
    CORE_SLOTS,
    EventVocab,
    SpecialsMap,
    ResolvedSpecialsMap,
    compute_specials_hash,
    validate_remap,
    E_NONE_IDX,
    E_OTHER_SPECIAL_IDX,
)


def test_core_alphabet_is_frozen_in_canonical_order():
    v = EventVocab()
    assert v.size == 15
    assert v.slots[:15] == CORE_SLOTS
    assert v.index("E_NONE") == E_NONE_IDX
    assert v.index("E_OTHER_SPECIAL") == E_OTHER_SPECIAL_IDX


def test_extensions_append_after_core_in_order():
    v = EventVocab(extensions=["E_VENDOR_A", "E_VENDOR_B"])
    assert v.size == 17
    assert v.slots[15:] == ("E_VENDOR_A", "E_VENDOR_B")
    assert v.index("E_VENDOR_A") == 15
    assert v.extensions == ("E_VENDOR_A", "E_VENDOR_B")


def test_duplicate_extension_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        EventVocab(extensions=["E_END_TURN"])


def test_specials_hash_stable_under_key_reorder():
    v = EventVocab()
    a = SpecialsMap(v, {"E_END_TURN": ["<|im_end|>"], "E_BEGIN_REASONING": ["<think>"]})
    b = SpecialsMap(v, {"E_BEGIN_REASONING": ["<think>"], "E_END_TURN": ["<|im_end|>"]})
    assert compute_specials_hash(v, a) == compute_specials_hash(v, b)


def test_specials_hash_changes_on_remap_change():
    v = EventVocab()
    s = SpecialsMap(v, {"E_END_TURN": ["<|im_end|>"]})
    h1 = compute_specials_hash(v, s)
    h2 = compute_specials_hash(v, s, event_remap={"E_BEGIN_REASONING": "drop"})
    assert h1 != h2


def test_specials_hash_changes_on_extension_change():
    s_a = SpecialsMap(EventVocab(), {"E_END_TURN": ["<|im_end|>"]})
    s_b = SpecialsMap(EventVocab(extensions=["E_VENDOR"]), {"E_END_TURN": ["<|im_end|>"]})
    h1 = compute_specials_hash(EventVocab(), s_a)
    h2 = compute_specials_hash(EventVocab(extensions=["E_VENDOR"]), s_b)
    assert h1 != h2


def test_validate_remap_rejects_unknown_slot():
    v = EventVocab()
    with pytest.raises(ValueError, match="not in the event alphabet"):
        validate_remap(v, {"E_DOES_NOT_EXIST": "drop"})
    with pytest.raises(ValueError, match="not in the event alphabet"):
        validate_remap(v, {"E_END_TURN": "E_DOES_NOT_EXIST"})


def test_validate_remap_rejects_self_alias():
    v = EventVocab()
    with pytest.raises(ValueError, match="no-op"):
        validate_remap(v, {"E_END_TURN": "E_END_TURN"})


def test_validate_remap_accepts_drop_and_alias():
    v = EventVocab()
    validate_remap(v, {"E_BEGIN_REASONING": "drop", "E_END_TURN": "E_END_MESSAGE"})


def test_specials_map_rejects_unknown_slot():
    v = EventVocab()
    with pytest.raises(ValueError, match="unknown slot"):
        SpecialsMap(v, {"E_NOT_A_SLOT": ["x"]})


def test_resolved_specials_supported_mask_reflects_slots():
    v = EventVocab()
    r = ResolvedSpecialsMap(v, {1: [101], 3: [102, 103]})  # E_END_TURN, E_BEGIN_REASONING
    assert r.supported_mask == ((1 << 1) | (1 << 3))
    assert r.slot_ids(1) == [101]
    assert r.slot_ids(3) == [102, 103]
    assert r.all_event_token_ids() == [101, 102, 103]
