import pytest
import torch
from conftest import requires_gpu
from classes.losses import (
    _skew_kl_loss, _akl_loss, _abomination_loss,
    calculate_divergence, _FUSED_LOSS_FUNCTIONS,
)


@requires_gpu
@pytest.mark.slow
@pytest.mark.parametrize("N", [100, 500])
def test_skew_kl_parity(N):
    from kernels import fused_skew_kl_forward_backward

    torch.manual_seed(42)
    device = "cuda"
    student = torch.softmax(torch.randn(N, 256, device=device), dim=-1).requires_grad_(True)
    teacher = torch.softmax(torch.randn(N, 256, device=device), dim=-1)
    actual = torch.randint(0, 256, (N,), device=device, dtype=torch.uint8)

    ref = _skew_kl_loss(student, teacher, actual, alpha=0.5)
    ref["train_loss"].backward()
    ref_grad = student.grad.clone()
    student.grad = None

    grad_cuda, cuda_dict = fused_skew_kl_forward_backward(student.detach(), teacher, actual, alpha=0.5)
    if grad_cuda is None:
        pytest.skip("Kernel compilation failed")

    loss_diff = abs(ref["train_loss"].item() - cuda_dict["train_loss"].item())
    assert loss_diff < 1e-4, f"loss diff {loss_diff}"

    cos = torch.nn.functional.cosine_similarity(
        grad_cuda.flatten().unsqueeze(0), ref_grad.flatten().unsqueeze(0)
    )
    assert cos.item() > 0.999, f"cosine sim {cos.item()}"


@requires_gpu
@pytest.mark.slow
@pytest.mark.parametrize("N", [100, 500])
def test_akl_parity(N):
    from kernels import fused_akl_forward_backward

    torch.manual_seed(42)
    device = "cuda"
    student = torch.softmax(torch.randn(N, 256, device=device), dim=-1).requires_grad_(True)
    teacher = torch.softmax(torch.randn(N, 256, device=device), dim=-1)
    actual = torch.randint(0, 256, (N,), device=device, dtype=torch.uint8)

    ref = _akl_loss(student, teacher, actual, alpha=0.5)
    ref["train_loss"].backward()
    ref_grad = student.grad.clone()
    student.grad = None

    grad_cuda, cuda_dict = fused_akl_forward_backward(student.detach(), teacher, actual, alpha=0.5)
    if grad_cuda is None:
        pytest.skip("Kernel compilation failed")

    loss_diff = abs(ref["train_loss"].item() - cuda_dict["train_loss"].item())
    assert loss_diff < 1e-3, f"loss diff {loss_diff}"

    cos = torch.nn.functional.cosine_similarity(
        grad_cuda.flatten().unsqueeze(0), ref_grad.flatten().unsqueeze(0)
    )
    assert cos.item() > 0.999, f"cosine sim {cos.item()}"


@requires_gpu
@pytest.mark.slow
@pytest.mark.parametrize("N", [100, 500])
def test_abomination_parity(N):
    from kernels import fused_abomination_forward_backward

    torch.manual_seed(42)
    device = "cuda"
    student = torch.softmax(torch.randn(N, 256, device=device), dim=-1).requires_grad_(True)
    teacher = torch.softmax(torch.randn(N, 256, device=device), dim=-1)
    actual = torch.randint(0, 256, (N,), device=device, dtype=torch.uint8)

    ref = _abomination_loss(student, teacher, actual, alpha=0.5)
    ref["train_loss"].backward()
    ref_grad = student.grad.clone()
    student.grad = None

    grad_cuda, cuda_dict = fused_abomination_forward_backward(student.detach(), teacher, actual, alpha=0.5)
    if grad_cuda is None:
        pytest.skip("Kernel compilation failed")

    loss_diff = abs(ref["train_loss"].item() - cuda_dict["train_loss"].item())
    assert loss_diff < 1e-3, f"loss diff {loss_diff}"

    cos = torch.nn.functional.cosine_similarity(
        grad_cuda.flatten().unsqueeze(0), ref_grad.flatten().unsqueeze(0)
    )
    assert cos.item() > 0.999, f"cosine sim {cos.item()}"


@requires_gpu
@pytest.mark.slow
def test_calculate_divergence_falls_back():
    device = "cuda"
    student = torch.softmax(torch.randn(50, 256, device=device), dim=-1).requires_grad_(True)
    teacher = torch.softmax(torch.randn(50, 256, device=device), dim=-1)
    actual = torch.randint(0, 256, (50,), device=device)

    import classes.losses as losses_mod
    saved = dict(losses_mod._FUSED_LOSS_FUNCTIONS)
    try:
        losses_mod._FUSED_LOSS_FUNCTIONS.clear()
        result = calculate_divergence(student, teacher, actual, alpha=0.5, loss_type="abomination")
        assert "train_loss" in result
        assert result["train_loss"].requires_grad
    finally:
        losses_mod._FUSED_LOSS_FUNCTIONS.update(saved)


@requires_gpu
@pytest.mark.slow
@pytest.mark.parametrize("loss_type", ["skew_kl", "akl", "abomination"])
def test_calculate_divergence_autograd(loss_type):
    device = "cuda"
    student = torch.softmax(torch.randn(50, 256, device=device), dim=-1).requires_grad_(True)
    teacher = torch.softmax(torch.randn(50, 256, device=device), dim=-1)
    actual = torch.randint(0, 256, (50,), device=device)

    result = calculate_divergence(student, teacher, actual, alpha=0.5, loss_type=loss_type)
    loss = result["train_loss"]
    loss.backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0
