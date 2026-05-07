"""Tests for event-loss kernels and student-side remap helpers."""
import torch
import pytest

from classes.event_vocab import EventVocab
from classes.event_losses import (
    apply_event_remap,
    build_remap_indices,
    event_jsd,
    event_kl_fwd,
    event_kl_rev,
    event_hellinger,
    get_event_loss,
)


def _norm(t):
    return t / t.sum(-1, keepdim=True).clamp(min=1e-12)


def test_jsd_zero_when_distributions_match():
    p = _norm(torch.rand(4, 8))
    mask = torch.ones(8, dtype=torch.bool)
    assert event_jsd(p, p, mask).item() == pytest.approx(0.0, abs=1e-5)


def test_kl_fwd_zero_when_match():
    p = _norm(torch.rand(4, 8))
    mask = torch.ones(8, dtype=torch.bool)
    assert event_kl_fwd(p, p, mask).item() == pytest.approx(0.0, abs=1e-5)


def test_hellinger_zero_when_match():
    p = _norm(torch.rand(4, 8))
    mask = torch.ones(8, dtype=torch.bool)
    assert event_hellinger(p, p, mask).item() == pytest.approx(0.0, abs=1e-5)


def test_jsd_symmetric():
    p = _norm(torch.rand(4, 8))
    q = _norm(torch.rand(4, 8))
    mask = torch.ones(8, dtype=torch.bool)
    assert event_jsd(p, q, mask).item() == pytest.approx(event_jsd(q, p, mask).item(), abs=1e-5)


def test_mask_zeroes_unsupported_slots():
    # Differ only in a masked slot - divergence should equal divergence on remaining slots.
    p = _norm(torch.tensor([[0.5, 0.5, 0.0, 0.0]]))
    q = _norm(torch.tensor([[0.5, 0.5, 0.5, 0.0]]))
    full_mask = torch.ones(4, dtype=torch.bool)
    sub_mask = torch.tensor([True, True, False, True])
    a = event_jsd(p, q, sub_mask).item()
    # Manually drop slot 2 from both, renorm, compare.
    p2 = _norm(p[:, [0, 1, 3]])
    q2 = _norm(q[:, [0, 1, 3]])
    b = event_jsd(p2, q2, torch.ones(3, dtype=torch.bool)).item()
    assert a == pytest.approx(b, abs=1e-5)


def test_apply_event_remap_alias_folds_mass():
    # Use core slots: alias E_END_TURN -> E_END_MESSAGE.
    vocab = EventVocab()
    teacher = torch.zeros(1, vocab.size)
    et = vocab.index("E_END_TURN")
    em = vocab.index("E_END_MESSAGE")
    teacher[0, et] = 0.3
    teacher[0, em] = 0.2
    teacher[0, 0] = 0.5  # E_NONE
    idx_map, keep = build_remap_indices(vocab, {"E_END_TURN": "E_END_MESSAGE"})
    out, mask = apply_event_remap(teacher, idx_map, keep)
    assert out[0, et].item() == pytest.approx(0.0, abs=1e-6)
    assert out[0, em].item() == pytest.approx(0.5, abs=1e-5)
    # No drops in this remap, so all slots remain in mask.
    assert mask.all().item()


def test_apply_event_remap_drop_zeroes_and_renorms():
    vocab = EventVocab()
    teacher = torch.zeros(1, vocab.size)
    et = vocab.index("E_END_TURN")
    em = vocab.index("E_END_MESSAGE")
    teacher[0, et] = 0.3
    teacher[0, em] = 0.2
    teacher[0, 0] = 0.5
    idx_map, keep = build_remap_indices(vocab, {"E_END_TURN": "drop"})
    out, mask = apply_event_remap(teacher, idx_map, keep)
    assert out[0, et].item() == pytest.approx(0.0, abs=1e-6)
    assert mask[et].item() is False
    surviving = out[0] * mask
    assert surviving.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_get_event_loss_dispatch():
    assert get_event_loss("jsd") is event_jsd
    assert get_event_loss("kl_fwd") is event_kl_fwd
    assert get_event_loss("kl_rev") is event_kl_rev
    assert get_event_loss("hellinger") is event_hellinger
    with pytest.raises(ValueError):
        get_event_loss("not_a_loss")
