import pytest
import torch
import math
from classes.losses import _abomination_loss, _skew_kl_loss, _akl_loss

ALL_LOSSES = [_abomination_loss, _skew_kl_loss, _akl_loss]
ALL_LOSS_IDS = ["abomination", "skew_kl", "akl"]

LOSSES_WITH_KL = ALL_LOSSES  # all return "kl_div"
KL_IDS = ALL_LOSS_IDS

# These have train_loss = kl + alpha * CE (akl uses adaptive weighting so kl_div != train_loss)
ADDITIVE_LOSSES = [_skew_kl_loss]
ADDITIVE_IDS = ["skew_kl"]


@pytest.fixture
def dists():
    def _make(N=64):
        student = torch.softmax(torch.randn(N, 256), dim=-1)
        teacher = torch.softmax(torch.randn(N, 256), dim=-1)
        actual = torch.randint(0, 256, (N,))
        return student, teacher, actual
    return _make


@pytest.mark.parametrize("fn", ALL_LOSSES, ids=ALL_LOSS_IDS)
def test_loss_non_negative(fn, dists):
    s, t, a = dists()
    result = fn(s, t, a, alpha=0.5)
    assert result["train_loss"].item() >= 0


@pytest.mark.parametrize("fn", LOSSES_WITH_KL, ids=KL_IDS)
def test_kl_non_negative(fn, dists):
    s, t, a = dists()
    result = fn(s, t, a, alpha=0.5)
    assert result["kl_div"].item() >= -1e-6


@pytest.mark.parametrize("fn", ALL_LOSSES, ids=ALL_LOSS_IDS)
def test_ce_non_negative(fn, dists):
    s, t, a = dists()
    result = fn(s, t, a, alpha=0.5)
    assert result["CE loss"].item() >= 0


@pytest.mark.parametrize("fn", LOSSES_WITH_KL, ids=KL_IDS)
def test_identical_dists_zero_kl(fn):
    t = torch.softmax(torch.randn(64, 256), dim=-1)
    s = t.clone()
    a = torch.randint(0, 256, (64,))
    result = fn(s, t, a, alpha=0.5)
    assert result["kl_div"].item() < 1e-5


@pytest.mark.parametrize("fn", LOSSES_WITH_KL, ids=KL_IDS)
def test_uniform_distributions(fn):
    u = torch.ones(64, 256) / 256
    a = torch.randint(0, 256, (64,))
    result = fn(u.clone(), u.clone(), a, alpha=0.5)
    assert result["kl_div"].item() < 1e-5


@pytest.mark.parametrize("fn", ALL_LOSSES, ids=ALL_LOSS_IDS)
def test_gradient_flows(fn):
    s = torch.softmax(torch.randn(32, 256), dim=-1).requires_grad_(True)
    t = torch.softmax(torch.randn(32, 256), dim=-1)
    a = torch.randint(0, 256, (32,))
    result = fn(s, t, a, alpha=0.5)
    result["train_loss"].backward()
    assert s.grad is not None
    assert not torch.isnan(s.grad).any()


@pytest.mark.parametrize("fn", ALL_LOSSES, ids=ALL_LOSS_IDS)
def test_output_dict_has_train_loss(fn, dists):
    s, t, a = dists()
    result = fn(s, t, a, alpha=0.5)
    assert "train_loss" in result
    assert "CE loss" in result


@pytest.mark.parametrize("fn", ADDITIVE_LOSSES, ids=ADDITIVE_IDS)
def test_alpha_zero_no_ce_contribution(fn):
    s = torch.softmax(torch.randn(64, 256), dim=-1)
    t = torch.softmax(torch.randn(64, 256), dim=-1)
    a = torch.randint(0, 256, (64,))
    result = fn(s, t, a, alpha=0.0)
    torch.testing.assert_close(result["train_loss"], result["kl_div"], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fn", ALL_LOSSES, ids=ALL_LOSS_IDS)
def test_batch_size_one(fn):
    s = torch.softmax(torch.randn(1, 256), dim=-1)
    t = torch.softmax(torch.randn(1, 256), dim=-1)
    a = torch.randint(0, 256, (1,))
    result = fn(s, t, a, alpha=0.5)
    assert torch.isfinite(result["train_loss"])
