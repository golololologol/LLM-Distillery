import pytest
import torch
from classes.losses import Losses


def test_add_losses():
    l = Losses(logger=None)
    l.add_losses({"a": torch.tensor(1.0), "b": torch.tensor(2.0)})
    l.add_losses({"a": torch.tensor(3.0), "c": torch.tensor(4.0)})
    assert l.num_steps_accumulated == 2
    assert l.loss_dict["a"].item() == pytest.approx(4.0)
    assert l.loss_dict["b"].item() == pytest.approx(2.0)
    assert l.loss_dict["c"].item() == pytest.approx(4.0)


def test_add_empty_raises():
    l = Losses(logger=None)
    with pytest.raises(ValueError):
        l.add_losses({})


def test_backward_scales():
    t = torch.tensor(8.0, requires_grad=True)
    l = Losses(logger=None)
    l.add_losses({"train_loss": t * 1.0})
    l.backward(divisor=4)
    assert t.grad is not None
    assert t.grad.item() == pytest.approx(0.25)


def test_empty_clears():
    l = Losses(logger=None)
    l.add_losses({"x": torch.tensor(5.0)})
    l.empty()
    assert l.loss_dict == {}
    assert l.num_steps_accumulated == 0


def test_division_operator():
    l = Losses(logger=None)
    l.add_losses({"a": torch.tensor(10.0), "b": torch.tensor(6.0)})
    result = l / 2
    assert isinstance(result, Losses)
    assert result.loss_dict["a"].item() == pytest.approx(5.0)
    assert result.loss_dict["b"].item() == pytest.approx(3.0)
    # original unchanged
    assert l.loss_dict["a"].item() == pytest.approx(10.0)


def test_multiplication_operator():
    l = Losses(logger=None)
    l.add_losses({"a": torch.tensor(2.0), "b": torch.tensor(4.0)})
    result = l * 3
    assert isinstance(result, Losses)
    assert result.loss_dict["a"].item() == pytest.approx(6.0)
    assert result.loss_dict["b"].item() == pytest.approx(12.0)
    # original unchanged
    assert l.loss_dict["a"].item() == pytest.approx(2.0)
