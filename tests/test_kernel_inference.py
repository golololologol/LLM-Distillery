import pytest
import torch
from conftest import requires_gpu
from kernels import get_inference_kernel


@pytest.fixture(autouse=True)
def _skip_no_kernel():
    if get_inference_kernel() is None:
        pytest.skip("Inference CUDA kernel not available")


def _make_inference_data(byte_vocab, T1, device="cuda"):
    V = byte_vocab.vocab_size
    T = T1 + 1
    logits = torch.randn(T, V, device=device, dtype=torch.float16) * 0.5
    token_ids = torch.randint(0, V, (T,), device=device)
    return logits, token_ids


@requires_gpu
@pytest.mark.slow
def test_inference_cuda_vs_pytorch_parity(byte_vocab):
    logits, token_ids = _make_inference_data(byte_vocab, 32)
    cuda_out = byte_vocab._marginalize_cuda(logits, token_ids)
    pytorch_out = byte_vocab._marginalize_pytorch(logits, token_ids)
    diff = (cuda_out.float() - pytorch_out.float()).abs().max().item()
    assert diff < 5e-3, f"Max abs diff {diff:.6f} >= 5e-3"


@requires_gpu
@pytest.mark.slow
def test_inference_output_normalized(byte_vocab):
    logits, token_ids = _make_inference_data(byte_vocab, 64)
    out = byte_vocab._marginalize_cuda(logits, token_ids)
    row_sums = out.float().sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=5e-3), \
        f"Row sums not ~1: min={row_sums.min():.6f} max={row_sums.max():.6f}"


@requires_gpu
@pytest.mark.slow
def test_inference_output_non_negative(byte_vocab):
    logits, token_ids = _make_inference_data(byte_vocab, 64)
    out = byte_vocab._marginalize_cuda(logits, token_ids)
    assert (out >= 0).all(), f"Negative values found: min={out.min().item():.6f}"


@requires_gpu
@pytest.mark.slow
def test_inference_chunk_size_invariance(byte_vocab):
    logits, token_ids = _make_inference_data(byte_vocab, 32)
    out_32 = byte_vocab._marginalize_cuda(logits, token_ids, T_CHUNK=32)
    out_512 = byte_vocab._marginalize_cuda(logits, token_ids, T_CHUNK=512)
    diff = (out_32.float() - out_512.float()).abs().max().item()
    assert diff < 1e-3, f"Chunk size invariance failed: max diff {diff:.6f}"


@requires_gpu
@pytest.mark.slow
def test_inference_marginalize_dispatch(byte_vocab):
    logits, token_ids = _make_inference_data(byte_vocab, 64)
    dispatch_out = byte_vocab.marginalize(logits, token_ids)
    cuda_out = byte_vocab._marginalize_cuda(logits, token_ids)
    diff = (dispatch_out.float() - cuda_out.float()).abs().max().item()
    assert diff < 1e-6, f"Dispatch != CUDA: max diff {diff:.6f}"
