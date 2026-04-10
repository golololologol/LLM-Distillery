import pytest
import torch
from conftest import requires_gpu


@requires_gpu
def test_vocab_size_even(byte_vocab):
    assert byte_vocab.vocab_size % 2 == 0


@requires_gpu
def test_byte_lens_positive(byte_vocab):
    assert (byte_vocab.token_byte_lens > 0).all()


@requires_gpu
def test_byte_matrix_d0_one_hot(byte_vocab):
    ones_per_row = (byte_vocab.byte_matrix_d0 == 1.0).sum(dim=1)
    assert (ones_per_row == 1).all()


@requires_gpu
def test_d1_boundaries_monotonic(byte_vocab):
    diffs = torch.diff(byte_vocab.d1_boundaries)
    assert (diffs >= 0).all()


@requires_gpu
def test_first_bytes_consistent(byte_vocab):
    expected = byte_vocab.token_byte_seqs[:, 0].long()
    torch.testing.assert_close(byte_vocab.first_bytes, expected)


@requires_gpu
def test_adjust_logits_wider(byte_vocab):
    V = byte_vocab.vocab_size
    logits = torch.randn(4, V + 100, device=byte_vocab.device)
    adjusted = byte_vocab._adjust_logits_to_vocab(logits)
    assert adjusted.shape == (4, V)


@requires_gpu
def test_adjust_logits_narrower(byte_vocab):
    V = byte_vocab.vocab_size
    logits = torch.randn(4, V - 100, device=byte_vocab.device)
    adjusted = byte_vocab._adjust_logits_to_vocab(logits)
    assert adjusted.shape == (4, V)
    assert adjusted[0, -1].item() == pytest.approx(torch.finfo(logits.dtype).min, abs=1e-6)


@requires_gpu
def test_adjust_logits_exact(byte_vocab):
    V = byte_vocab.vocab_size
    logits = torch.randn(4, V, device=byte_vocab.device)
    adjusted = byte_vocab._adjust_logits_to_vocab(logits)
    assert adjusted is logits


@requires_gpu
def test_content_positions_full_range(byte_vocab, tokenizer):
    text = "Hello, this is a test message."
    ids = tokenizer.encode(text)
    token_ids = torch.tensor(ids, device=byte_vocab.device)
    total_bytes = len(text.encode("utf-8"))
    content_byte_ranges = [(0, total_bytes)]
    active_idx, active_byte_lens, content_mask = byte_vocab._compute_content_positions(
        token_ids, content_byte_ranges, len(ids), byte_vocab.device
    )
    assert len(active_idx) > 0


@requires_gpu
def test_content_positions_no_content(byte_vocab, tokenizer):
    text = "Hello world"
    ids = tokenizer.encode(text)
    token_ids = torch.tensor(ids, device=byte_vocab.device)
    content_byte_ranges = []
    active_idx, active_byte_lens, content_mask = byte_vocab._compute_content_positions(
        token_ids, content_byte_ranges, len(ids), byte_vocab.device
    )
    assert len(active_idx) == 0


@requires_gpu
def test_content_positions_partial(byte_vocab, tokenizer):
    text = "Hello world test"
    ids = tokenizer.encode(text)
    token_ids = torch.tensor(ids, device=byte_vocab.device)
    total_bytes = len(text.encode("utf-8"))
    full_active, _, _ = byte_vocab._compute_content_positions(
        token_ids, [(0, total_bytes)], len(ids), byte_vocab.device
    )
    partial_active, _, _ = byte_vocab._compute_content_positions(
        token_ids, [(0, total_bytes // 3)], len(ids), byte_vocab.device
    )
    assert len(partial_active) <= len(full_active)


@requires_gpu
def test_cuda_kernel_tensors(byte_vocab):
    result = byte_vocab.cuda_kernel_tensors()
    assert len(result) == 6
    for i, t in enumerate(result[:5]):
        assert t.dtype == torch.int32
        assert t.is_cuda
    assert isinstance(result[5], int)
