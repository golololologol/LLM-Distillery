import pytest
import torch
from conftest import requires_gpu


def _naive_marginalize(probs_2d, token_ids, byte_vocab):
    T = probs_2d.shape[0]
    V = byte_vocab.vocab_size
    device = byte_vocab.device
    results = []

    for t in range(T - 1):
        next_tok = token_ids[t + 1].item()
        blen = byte_vocab.token_byte_lens[next_tok].item()
        bseq = byte_vocab.token_byte_seqs[next_tok, :blen]

        for d in range(blen):
            if d == 0:
                mask = torch.ones(V, dtype=torch.bool, device=device)
            else:
                prefix = bseq[:d]
                mask = (byte_vocab.token_byte_seqs[:, :d] == prefix.unsqueeze(0)).all(dim=1)
                mask &= byte_vocab.token_byte_lens > d

            matching_probs = probs_2d[t] * mask.float()
            total = matching_probs.sum()

            dist = torch.zeros(256, device=device, dtype=torch.float32)
            if total > 0:
                d_bytes = byte_vocab.token_byte_seqs[mask, d].long()
                d_probs = probs_2d[t, mask]
                dist.scatter_add_(0, d_bytes, d_probs)
                dist /= total
            results.append(dist)

    return torch.stack(results) if results else torch.empty(0, 256, device=device)


@requires_gpu
def test_output_non_negative(byte_vocab):
    T, V = 16, byte_vocab.vocab_size
    logits = torch.randn(T, V, device=byte_vocab.device, dtype=torch.float16)
    token_ids = torch.randint(0, V, (T,), device=byte_vocab.device)
    dists = byte_vocab.marginalize(logits, token_ids)
    assert (dists >= 0).all()


@requires_gpu
def test_output_sums_to_one(byte_vocab):
    T, V = 16, byte_vocab.vocab_size
    logits = torch.randn(T, V, device=byte_vocab.device, dtype=torch.float16)
    token_ids = torch.randint(0, V, (T,), device=byte_vocab.device)
    dists = byte_vocab.marginalize(logits, token_ids)
    sums = dists.float().sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=5e-3, rtol=0)


@requires_gpu
def test_output_shape(byte_vocab):
    T, V = 16, byte_vocab.vocab_size
    logits = torch.randn(T, V, device=byte_vocab.device, dtype=torch.float16)
    token_ids = torch.randint(0, V, (T,), device=byte_vocab.device)
    dists = byte_vocab.marginalize(logits, token_ids)
    expected_len = byte_vocab.token_byte_lens[token_ids[1:]].sum().item()
    assert dists.shape == (expected_len, 256)


@requires_gpu
def test_pytorch_vs_naive_oracle(byte_vocab):
    T, V = 32, byte_vocab.vocab_size
    torch.manual_seed(123)
    logits = torch.randn(T, V, device=byte_vocab.device, dtype=torch.float16)
    token_ids = torch.randint(0, V, (T,), device=byte_vocab.device)

    probs = torch.softmax(logits.float(), dim=-1)
    naive = _naive_marginalize(probs, token_ids, byte_vocab)
    pytorch = byte_vocab._marginalize_pytorch(logits, token_ids, T_CHUNK=256)

    max_diff = (naive - pytorch.float()).abs().max().item()
    assert max_diff < 1e-2, f"max diff {max_diff}"


@requires_gpu
def test_cuda_vs_pytorch_parity(byte_vocab):
    T, V = 32, byte_vocab.vocab_size
    torch.manual_seed(456)
    logits = torch.randn(T, V, device=byte_vocab.device, dtype=torch.float16)
    token_ids = torch.randint(0, V, (T,), device=byte_vocab.device)

    pytorch_out = byte_vocab._marginalize_pytorch(logits, token_ids, T_CHUNK=256)

    from kernels import get_inference_kernel
    if get_inference_kernel() is None:
        pytest.skip("CUDA inference kernel not available")

    cuda_out = byte_vocab._marginalize_cuda(logits, token_ids, T_CHUNK=512)
    max_diff = (pytorch_out.float() - cuda_out.float()).abs().max().item()
    assert max_diff < 5e-3, f"max diff {max_diff}"


@requires_gpu
def test_marginalize_train_gradient_flow(byte_vocab):
    T, V = 16, byte_vocab.vocab_size
    logits = torch.randn(T, V, device=byte_vocab.device, dtype=torch.float32, requires_grad=True)
    token_ids = torch.randint(0, V, (T,), device=byte_vocab.device)
    dists = byte_vocab.marginalize_train(logits, token_ids, T_CHUNK=256)
    loss = dists.sum()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


@requires_gpu
def test_marginalize_content_filters(byte_vocab, tokenizer):
    text = "Hello, this is a test."
    ids = tokenizer.encode(text)
    T = len(ids)
    V = byte_vocab.vocab_size
    token_ids = torch.tensor(ids, device=byte_vocab.device)
    logits = torch.randn(T, V, device=byte_vocab.device, dtype=torch.float16)

    full = byte_vocab.marginalize(logits, token_ids)
    total_bytes = len(text.encode("utf-8"))
    partial = byte_vocab.marginalize_content(
        logits, token_ids, [(0, total_bytes // 3)], length=T, T_CHUNK=256
    )
    assert partial.shape[0] <= full.shape[0]


@requires_gpu
def test_chunk_size_invariance(byte_vocab):
    T, V = 64, byte_vocab.vocab_size
    torch.manual_seed(789)
    logits = torch.randn(T, V, device=byte_vocab.device, dtype=torch.float16)
    token_ids = torch.randint(0, V, (T,), device=byte_vocab.device)

    out_32 = byte_vocab._marginalize_pytorch(logits, token_ids, T_CHUNK=32)
    out_256 = byte_vocab._marginalize_pytorch(logits, token_ids, T_CHUNK=256)

    max_diff = (out_32.float() - out_256.float()).abs().max().item()
    assert max_diff < 1e-3, f"max diff {max_diff}"
