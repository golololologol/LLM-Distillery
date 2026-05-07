"""Tests for cross-channel features: byte content-mass renormalisation and
cross-teacher event merging."""
import json

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest
import torch
import zstandard
from typing import Any, cast

from utils.merging_utils import _merge_one_convo, merge_teacher_files


def _write_teacher_with_events(path, *, events_arr, event_manifest, supported_mask, alphabet, hash_="hA"):
    """Write a teacher with a single content segment + an events tensor."""
    with h5py.File(path, "w") as f:
        f.attrs["layout_version"] = 5
        f.attrs["event_alphabet"] = json.dumps(list(alphabet))
        g = f.create_group("convo_0")
        # 1 content segment, 4 bytes
        bc = 4
        arr = np.full((bc, 256), 0.5, dtype=np.float32)
        arr /= arr.sum(1, keepdims=True)
        g.create_dataset("dense_distributions", data=arr.astype(np.float16))
        g.attrs["content_sha"] = "sha_0"
        g.attrs["segment_manifest"] = json.dumps([
            {
                "msg_idx": 0, "segment_idx": 0, "type": "content",
                "tool_call_idx": None, "byte_count": bc,
                "canonical_target_hash": hash_, "truncated": False,
                "handling": "active", "role": "assistant",
            }
        ])
        g.create_dataset(
            "event_distributions", data=events_arr.astype(np.float16),
            **cast(Any, getattr(hdf5plugin, "Zstd"))(clevel=1),
        )
        g.attrs["event_manifest"] = json.dumps(event_manifest)
        g.attrs["event_alphabet"] = json.dumps(list(alphabet))
        g.attrs["supported_mask"] = int(supported_mask)
        g.attrs["event_count"] = events_arr.shape[0]


def test_event_merge_weighted_average(tmp_path):
    # Two teachers, same alphabet, same anchor. Different distributions.
    alphabet = ("E_NONE", "E_END_TURN", "E_END_MESSAGE")
    sup_mask = 0b111  # all 3 slots supported
    a_events = np.array([[0.6, 0.3, 0.1]], dtype=np.float32)
    b_events = np.array([[0.2, 0.7, 0.1]], dtype=np.float32)
    manifest = [{"msg_idx": 0, "segment_idx": 0, "anchor_side": "after", "row": 0, "unreachable": False, "truncated": False}]

    _write_teacher_with_events(
        tmp_path / "A.hdf5", events_arr=a_events, event_manifest=manifest,
        supported_mask=sup_mask, alphabet=alphabet,
    )
    _write_teacher_with_events(
        tmp_path / "B.hdf5", events_arr=b_events, event_manifest=manifest,
        supported_mask=sup_mask, alphabet=alphabet,
    )

    result = _merge_one_convo(str(tmp_path), {"A": 0.5, "B": 0.5}, "convo_0")
    assert result is not None
    merged_events = result[7]
    assert merged_events is not None
    assert merged_events.shape == (1, 3)
    # Equal-weight average over A and B.
    expected = 0.5 * a_events + 0.5 * b_events
    np.testing.assert_allclose(merged_events.astype(np.float32), expected, atol=2e-3)


def test_event_merge_disjoint_supported_masks(tmp_path):
    # A supports slots 0,1; B supports slots 0,2. Merged anchor uses union.
    alphabet = ("E_NONE", "E_END_TURN", "E_END_MESSAGE")
    a_events = np.array([[0.7, 0.3, 0.0]], dtype=np.float32)
    b_events = np.array([[0.6, 0.0, 0.4]], dtype=np.float32)
    manifest = [{"msg_idx": 0, "segment_idx": 0, "anchor_side": "after", "row": 0}]

    _write_teacher_with_events(
        tmp_path / "A.hdf5", events_arr=a_events, event_manifest=manifest,
        supported_mask=0b011, alphabet=alphabet,
    )
    _write_teacher_with_events(
        tmp_path / "B.hdf5", events_arr=b_events, event_manifest=manifest,
        supported_mask=0b101, alphabet=alphabet,
    )
    result = _merge_one_convo(str(tmp_path), {"A": 0.5, "B": 0.5}, "convo_0")
    assert result is not None
    union_mask = result[10]
    assert union_mask == 0b111
    merged = result[7]
    assert merged is not None
    # E_NONE: average of both teachers.
    # E_END_TURN: only A → A's value.
    # E_END_MESSAGE: only B → B's value.
    # Then row renormalised.
    raw = np.array([0.65, 0.3, 0.4])
    expected = raw / raw.sum()
    np.testing.assert_allclose(merged[0].astype(np.float32), expected, atol=2e-3)


def test_event_alphabet_mismatch_skips_events(tmp_path):
    alpha_a = ("E_NONE", "E_END_TURN", "E_END_MESSAGE")
    alpha_b = ("E_NONE", "E_END_MESSAGE", "E_END_TURN")  # different ordering
    manifest = [{"msg_idx": 0, "segment_idx": 0, "anchor_side": "after", "row": 0}]
    a = np.array([[0.6, 0.3, 0.1]], dtype=np.float32)
    _write_teacher_with_events(tmp_path / "A.hdf5", events_arr=a, event_manifest=manifest, supported_mask=0b111, alphabet=alpha_a)
    _write_teacher_with_events(tmp_path / "B.hdf5", events_arr=a, event_manifest=manifest, supported_mask=0b111, alphabet=alpha_b)
    result = _merge_one_convo(str(tmp_path), {"A": 0.5, "B": 0.5}, "convo_0")
    assert result is not None
    # Event distributions should be skipped due to alphabet mismatch
    merged_events = result[7]
    assert merged_events is None
    event_alphabet = result[9]
    assert event_alphabet is None
    # Dense distributions should still be merged
    dense_shape = result[4]
    assert dense_shape[0] > 0  # there is at least one byte segment


def test_byte_marginalize_masks_special_tokens(monkeypatch):
    """plan §4: with attach_specials, byte channel must not include special-token mass."""
    from classes.byte_vocab import ByteVocabIndex
    from classes.event_vocab import EventVocab, ResolvedSpecialsMap

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    # Build a tiny synthetic tokenizer-like wrapper over the byte vocab.
    # Use a HF tokenizer for realism:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("test_data/tiny_tokenizer", use_fast=True)
    bv = ByteVocabIndex(tok, device="cuda")

    vocab = EventVocab()
    # Pretend token id 5 is the E_END_TURN slot.
    resolved = ResolvedSpecialsMap(vocab, {1: [5]})  # slot 1 = E_END_TURN
    bv.attach_specials(vocab, resolved)

    # Create logits that put all mass on token 5 (the special).
    V = bv.vocab_size
    T = 3
    logits = torch.full((T, V), -1e9, device="cuda", dtype=torch.float32)
    logits[:, 5] = 0.0  # all-mass on the masked special
    token_ids = torch.zeros(T, dtype=torch.long, device="cuda")

    # With masking active and content_byte_ranges empty, marginalize_content
    # returns the full sequence; the byte distribution at all positions should
    # have ~zero mass since the only "live" token is the special.
    # We test instead via softmax behaviour: after _mask_event_logits, no slot
    # gets non-trivial mass.
    masked = bv._mask_event_logits(logits)
    probs = torch.softmax(masked, dim=-1)
    # The special token at id 5 must be ~0.
    assert probs[:, 5].max().item() < 1e-4
