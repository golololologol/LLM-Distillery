import pytest
import torch
import numpy as np

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("test_data/tiny_tokenizer")

@pytest.fixture(scope="session")
def byte_vocab(tokenizer):
    if not torch.cuda.is_available():
        pytest.skip("ByteVocabIndex requires CUDA")
    from classes.byte_vocab import ByteVocabIndex
    return ByteVocabIndex(tokenizer, device="cuda")

@pytest.fixture
def random_byte_dists():
    def _make(N, device="cpu"):
        student = torch.softmax(torch.randn(N, 256, device=device), dim=-1)
        teacher = torch.softmax(torch.randn(N, 256, device=device), dim=-1)
        actual = torch.randint(0, 256, (N,), device=device)
        return student, teacher, actual
    return _make
