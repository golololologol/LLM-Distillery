import pytest
import torch
import numpy as np
from pathlib import Path

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-golden", action="store_true", default=False,
        help="Rewrite golden files instead of comparing.",
    )


@pytest.fixture(scope="session")
def regenerate_golden(request):
    return request.config.getoption("--regenerate-golden")

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture(scope="session")
def tokenizer():
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    root = Path("test_data/tiny_tokenizer")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(root / "tokenizer.json"),
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=[
            "<|im_start|>", "<|im_end|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
            "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>",
            "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        ],
        chat_template=(root / "chat_template.jinja").read_text(encoding="utf-8"),
    )
    tokenizer.name_or_path = str(root)
    return tokenizer

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
