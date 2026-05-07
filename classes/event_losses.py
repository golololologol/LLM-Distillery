"""Event-channel divergence stack.

Pure PyTorch tensor operations over `[N, |E|]` event distributions, with a
per-row supported-slot mask. These are *separate* from the byte-channel loss
kernels; they take an `[N, |E|]` student dist, an `[N, |E|]` teacher dist, and
a slot mask shaped `[|E|]` (or `[N, |E|]`), and return a scalar loss.

First-pass divergences: JSD, KL forward, KL reverse, Hellinger. CE-against-
actual is deferred (plan §8).

The mask combines:
  - teacher.supported_mask (slots the teacher's specials map covers),
  - student.supported_mask (slots the student's specials map covers),
  - student.event_remap drop list (slots the student dropped).

Aliasing remaps (slot → slot) are applied to the teacher target *before* the
divergence call by `apply_event_remap`, which folds source-slot mass into the
target slot and renormalises over the resulting alphabet. Drop-remaps zero the
mask bit; the divergence then ignores those slots.
"""
from __future__ import annotations

import torch


_EPS = 1e-12


def _normalize_masked(p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Renormalise `p` over the slots where `mask` is true. Slots outside the
    mask are zeroed. Rows whose total surviving mass is zero return all zeros.
    """
    p = p * mask
    s = p.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    return p / s


def apply_event_remap(
    teacher: torch.Tensor,
    remap: dict[int, int | None],
    keep_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply student-side event_remap to a teacher event distribution.

    Args:
        teacher: [N, E] teacher event probabilities.
        remap: {src_slot_idx: target_slot_idx_or_None}. None == 'drop'.
            Slots not in `remap` are identity.
        keep_mask: [E] bool. Slots that survive after drop-remap (the student
            still loses bits at drop'd indices). Caller is responsible for
            constructing this from the remap dict.

    Returns:
        (remapped, surviving_mask): the renormalised teacher distribution and
        the surviving-slot mask, suitable for passing into a divergence below.
    """
    if not remap:
        return _normalize_masked(teacher, keep_mask), keep_mask
    out = teacher.clone()
    # Aliases: fold source -> target; zero out source.
    for src, dst in remap.items():
        if dst is None:
            continue
        out[..., dst] = out[..., dst] + out[..., src]
        out[..., src] = 0
    # Drops: zero source columns.
    for src, dst in remap.items():
        if dst is None:
            out[..., src] = 0
    return _normalize_masked(out, keep_mask), keep_mask


def event_jsd(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between student and teacher, masked, mean over rows."""
    s = _normalize_masked(student, mask)
    t = _normalize_masked(teacher, mask)
    m = 0.5 * (s + t)
    log_m = (m + _EPS).log()
    kl_s = (s * ((s + _EPS).log() - log_m) * mask).sum(dim=-1)
    kl_t = (t * ((t + _EPS).log() - log_m) * mask).sum(dim=-1)
    return 0.5 * (kl_s + kl_t).mean()


def event_kl_fwd(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student), the standard distillation loss."""
    s = _normalize_masked(student, mask)
    t = _normalize_masked(teacher, mask)
    return (t * ((t + _EPS).log() - (s + _EPS).log()) * mask).sum(dim=-1).mean()


def event_kl_rev(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(student || teacher)."""
    s = _normalize_masked(student, mask)
    t = _normalize_masked(teacher, mask)
    return (s * ((s + _EPS).log() - (t + _EPS).log()) * mask).sum(dim=-1).mean()


def event_hellinger(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    s = _normalize_masked(student, mask)
    t = _normalize_masked(teacher, mask)
    return ((s.sqrt() - t.sqrt()) ** 2 * mask).sum(dim=-1).mul(0.5).sqrt().mean()


_REGISTRY = {
    "jsd": event_jsd,
    "kl_fwd": event_kl_fwd,
    "kl_rev": event_kl_rev,
    "hellinger": event_hellinger,
}


def get_event_loss(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"unknown event_loss {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build_remap_indices(vocab, remap_strings: dict[str, str]) -> tuple[dict[int, int | None], torch.Tensor]:
    """Translate a string remap (`{slot_name: 'drop'|other_slot_name}`) into
    (idx_map, keep_mask) suitable for `apply_event_remap`.

    Args:
        vocab: an EventVocab instance.
        remap_strings: as parsed from the student TOML.

    Returns:
        (idx_map, keep_mask):
          - idx_map: {src_idx: dst_idx or None}.  None means 'drop'.
          - keep_mask: torch.bool [E]; True at slots that survive drop-remap.
    """
    idx_map: dict[int, int | None] = {}
    drop_set: set[int] = set()
    for src, dst in remap_strings.items():
        src_i = vocab.index(src)
        if dst == "drop":
            idx_map[src_i] = None
            drop_set.add(src_i)
        else:
            dst_i = vocab.index(dst)
            idx_map[src_i] = dst_i
    keep = torch.ones(vocab.size, dtype=torch.bool)
    for i in drop_set:
        keep[i] = False
    return idx_map, keep
