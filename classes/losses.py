from __future__ import annotations
from typing import TYPE_CHECKING
from torch import Tensor
import numpy as np
import torch


def _weighted_mean(x: Tensor, w: Tensor | None) -> Tensor:
    if w is None:
        return x.mean()
    return (x * w).sum() / w.sum().clamp(min=1e-8)


if TYPE_CHECKING:
    import wandb.wandb_run


class Losses:
    """
    A class to store and manage losses during training or validation.
    """
    def __init__(self, logger: "wandb.wandb_run.Run | None", validation: bool = False):
        self.loss_dict: dict[str, torch.Tensor] = {}
        self.validation = validation
        self.device = None
        self.logger = logger
        self.num_steps_accumulated = 0

    def add_losses(self, new_losses: dict[str, torch.Tensor]):
        """
        Adds new losses to the loss dictionary. 
        
        If new loss keys already exists in the dictionary, then they are summed with the existing ones accordingly.

        Args:
            new_losses (dict[str, torch.Tensor]): A dictionary containing the new losses to be added.
                The keys represent the names of the losses, and the values represent the corresponding loss tensors.

        Raises:
            ValueError: If no losses are provided to add to the loss dictionary.

        """
        if not new_losses:
            raise ValueError("No losses to add to the loss dictionary.")

        self.num_steps_accumulated += 1

        for key, value in new_losses.items():
            if key != "train_loss":
                new_losses[key] = value.clone().detach()

        if not self.loss_dict:
            self.loss_dict = new_losses
            return

        for key, value in new_losses.items():
            if key in self.loss_dict:
                self.loss_dict[key] += value
            else:
                self.loss_dict[key] = value

    def empty(self):
        """
        Clears the loss dictionary and resets the number of accumulated steps to zero.
        """
        self.loss_dict = {}
        self.num_steps_accumulated = 0
        
    def log(self, step: int, average: bool = True):
        if self.logger is None:
            return

        prefix = "Val Loss" if self.validation else "Loss"
        for key, value in self.loss_dict.items():
            if key == "train_loss":
                continue
            self.logger.log({f"{prefix}/{key}": ((value/self.num_steps_accumulated) if average else value)}, step=step)

    def backward(self, divisor: int = 1):
        """
        Does a backward pass on the training loss scaled by the divisor.

        Args:
            divisor (int): Divisor to scale the loss value. Default is 1.
        """
        if "train_loss" not in self.loss_dict:
            return
        self.loss_dict["train_loss"] /= divisor
        self.loss_dict["train_loss"].backward()
        del self.loss_dict["train_loss"]

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            new_losses = Losses(self.logger, self.validation)
            new_losses.loss_dict = {key: value / other for key, value in self.loss_dict.items()}
            return new_losses
        return NotImplemented
    
    def __itruediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            for key in self.loss_dict:
                self.loss_dict[key] /= other
            return self
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            new_losses = Losses(self.logger, self.validation)
            new_losses.loss_dict = {key: value * other for key, value in self.loss_dict.items()}
            return new_losses
        return NotImplemented
    
    def __imul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            for key in self.loss_dict:
                self.loss_dict[key] *= other
            return self
        return NotImplemented
    
    def __str__(self):
        loss_str = " Validation Losses:\n" if self.validation else " Training Losses:\n"
        for key, value in self.loss_dict.items():
            loss_str += f"{key}: {value.item():.4f}\n"
        return loss_str


def _abomination_loss(student_dists: Tensor, teacher_dists: Tensor, actual_bytes: Tensor, alpha: float, entropy_weights: Tensor | None = None) -> dict[str, Tensor]:
    eps = 1e-8
    s = student_dists + eps
    t = teacher_dists + eps
    s_log = s.log()
    t_log = t.log()

    fwd_kl = (t * (t_log - s_log)).sum(-1)
    rev_kl = (s * (s_log - t_log)).sum(-1)

    idx = actual_bytes.long()
    arange = torch.arange(len(idx), device=student_dists.device)
    student_ce = -(student_dists[arange, idx] + eps).log()
    teacher_ce = -(teacher_dists[arange, idx] + eps).log()
    ce_diff = torch.nn.functional.softplus(student_ce - teacher_ce, beta=5.0).clamp(max=5.0)

    def abomination(kl, ce_diff):
        norm = kl.mean().detach().clamp(min=eps)
        weights = ((kl / norm) + 1).pow(alpha) + ce_diff
        return _weighted_mean(kl * weights, entropy_weights)

    weighted_fwd = abomination(fwd_kl, ce_diff)
    weighted_rev = abomination(rev_kl, ce_diff)
    loss = (weighted_fwd + weighted_rev) / 2

    return {
        "train_loss": loss,
        "custom loss": loss,
        "CE loss": student_ce.mean(),
        "kl_div": fwd_kl.mean(),
        "reverse kl_div": rev_kl.mean(),
        "weighted kl_div": weighted_fwd,
        "weighted rev. kl_div": weighted_rev,
        "teacher CE loss": teacher_ce.mean(),
        "CE diff": ce_diff.mean(),
    }


def _skew_kl_loss(student_dists: Tensor, teacher_dists: Tensor, actual_bytes: Tensor, alpha: float, entropy_weights: Tensor | None = None) -> dict[str, Tensor]:
    eps = 1e-8
    lam = 0.1  # DistiLLM: lambda=0.1 recommended

    # SRKL: KL(S || (1-lam)*T + lam*S) - canonical DistiLLM formulation
    mix = (1.0 - lam) * teacher_dists + lam * student_dists
    s = student_dists + eps
    s_log = s.log()
    mix_log = (mix + eps).log()
    srkl = (s * (s_log - mix_log)).sum(-1)

    kl_loss = _weighted_mean(srkl, entropy_weights)

    # Ground truth CE
    idx = actual_bytes.long()
    arange = torch.arange(len(idx), device=student_dists.device)
    student_ce = -(student_dists[arange, idx] + eps).log()
    teacher_ce = -(teacher_dists[arange, idx] + eps).log()

    loss = kl_loss + alpha * _weighted_mean(student_ce, entropy_weights)

    return {
        "train_loss": loss,
        "custom loss": loss,
        "CE loss": student_ce.mean(),
        "kl_div": kl_loss,
        "teacher CE loss": teacher_ce.mean(),
    }


def _akl_loss(student_dists: Tensor, teacher_dists: Tensor, actual_bytes: Tensor, alpha: float, entropy_weights: Tensor | None = None) -> dict[str, Tensor]:
    eps = 1e-8
    s = student_dists + eps
    t = teacher_dists + eps
    s_log = s.log()
    t_log = t.log()

    fkl = (t * (t_log - s_log)).sum(-1)
    rkl = (s * (s_log - t_log)).sum(-1)

    sorted_t, sort_idx = teacher_dists.sort(dim=-1, descending=True)
    cumsum = sorted_t.cumsum(dim=-1)
    sorted_head = cumsum <= 0.5
    sorted_head[:, 0] = True
    head_mask = torch.zeros_like(sorted_head)
    head_mask.scatter_(1, sort_idx, sorted_head)

    gap = (teacher_dists - student_dists).abs()
    head_gap = (gap * head_mask).sum(-1)
    tail_gap = (gap * ~head_mask).sum(-1)

    w = (tail_gap / (head_gap + tail_gap + eps)).detach()
    kl_loss = _weighted_mean((1 - w) * fkl + w * rkl, entropy_weights)

    idx = actual_bytes.long()
    arange = torch.arange(len(idx), device=student_dists.device)
    student_ce = -(student_dists[arange, idx] + eps).log()
    teacher_ce = -(teacher_dists[arange, idx] + eps).log()

    loss = kl_loss + alpha * _weighted_mean(student_ce, entropy_weights)

    return {
        "train_loss": loss,
        "custom loss": loss,
        "CE loss": student_ce.mean(),
        "kl_div": fkl.mean(),
        "reverse kl_div": rkl.mean(),
        "teacher CE loss": teacher_ce.mean(),
        "adaptive_weight": w.mean(),
    }


_LOSS_FUNCTIONS = {
    "abomination": _abomination_loss,
    "skew_kl": _skew_kl_loss,
    "akl": _akl_loss,
}

from utils.kernel_utils import _FusedLossGrad

try:
    from kernels import (
        fused_skew_kl_forward_backward,
        fused_abomination_forward_backward,
        fused_akl_forward_backward,
    )
    _FUSED_LOSS_FUNCTIONS = {
        "skew_kl": fused_skew_kl_forward_backward,
        "abomination": fused_abomination_forward_backward,
        "akl": fused_akl_forward_backward,
    }
except (ImportError, Exception):
    _FUSED_LOSS_FUNCTIONS = {}


def calculate_divergence(student_dists: Tensor, teacher_dists: Tensor, actual_bytes: Tensor,
                         alpha: float, loss_type: str = "abomination", entropy_weights: Tensor | None = None) -> dict[str, Tensor]:
    fused_fn = _FUSED_LOSS_FUNCTIONS.get(loss_type)
    if fused_fn is not None and student_dists.requires_grad:
        grad_student, loss_dict = fused_fn(student_dists, teacher_dists, actual_bytes, alpha, entropy_weights=entropy_weights)
        if grad_student is not None:
            loss_dict["train_loss"] = _FusedLossGrad.apply(
                student_dists, grad_student, loss_dict["train_loss"]
            )
            return loss_dict

    fn = _LOSS_FUNCTIONS.get(loss_type)
    if fn is None:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Available: {list(_LOSS_FUNCTIONS)}")
    return fn(student_dists, teacher_dists, actual_bytes, alpha, entropy_weights=entropy_weights)
    