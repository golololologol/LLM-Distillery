from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler
from torch import Tensor
import torch.nn.functional as F
import torch
import math
import sys
import os

# shut the the hell up bnb with its `bin ..bitsandbytes\libbitsandbytes_cuda121.dll`
sys.stdout = open(os.devnull, 'w')
import bitsandbytes as bnb
sys.stdout = sys.__stdout__


class WarmupStableDecayLR(LRScheduler):
    """
    Implementation of Warmup Stable Decay learning rate scheduler from MiniCPM.\n
    Has custom cosine warmup instead of linear.\n
    Paper: https://arxiv.org/pdf/2404.06395
    """
    def __init__(self, optimizer, total_steps, warmup_steps, decay_start_percentage, final_lr, last_epoch=-1, verbose=False):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_start_step = int(total_steps * decay_start_percentage)
        self.decay_steps = self.total_steps - self.decay_start_step
        self.final_lr = final_lr
        self.constant_lr = optimizer.param_groups[0]['lr']

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / self.warmup_steps
            cosine = (0.5 * (1 + math.cos(math.pi + progress * math.pi)))
            return [self.constant_lr * cosine for group in self.optimizer.param_groups]
        elif self.warmup_steps <= self.last_epoch <= self.decay_start_step:
            return [self.constant_lr for group in self.optimizer.param_groups]
        elif self.decay_start_step < self.last_epoch < self.total_steps:
            lr_scale = 1 - ((self.last_epoch - self.decay_start_step) / self.decay_steps) * (1 - self.final_lr / self.constant_lr)
            return [self.constant_lr * lr_scale for group in self.optimizer.param_groups]
        else:
            return [self.final_lr for group in self.optimizer.param_groups]
        


def calculate_divergence(student_logits: Tensor, teacher_logits: Tensor, indices: Tensor|None, convo_CE_tokens: Tensor, alpha: Tensor, avoid_indices: Tensor) -> dict[str, Tensor]:

    def custom_kl_div(student_logprobs: Tensor, teacher_logprobs: Tensor, per_token: bool = False):
        kl_div_raw = F.kl_div(student_logprobs, teacher_logprobs, reduction='none', log_target=True)

        if per_token:
            kl_div = kl_div_raw.sum(dim=-1)
        else:
            kl_div = kl_div_raw.sum(dim=-1).mean()
        return kl_div

    def abomination_loss(kl_div_per_token: Tensor, alpha: Tensor, CE_loss_per_token: Tensor):
        if CE_loss_per_token.numel() < kl_div_per_token.numel():
            CE_loss_per_token = torch.cat((CE_loss_per_token, torch.zeros(1, device=CE_loss_per_token.device)))

        weights = ((kl_div_per_token / kl_div_per_token.max()) + 1).pow(alpha)
        weights = weights + CE_loss_per_token

        loss = (kl_div_per_token * weights).mean()
        return loss
    
    def custom_ce_loss(logprobs: Tensor, convo_CE_tokens: Tensor, indices: Tensor, ignore_indexes: Tensor) -> Tensor:
        crop = 0
        if convo_CE_tokens.size(-1) < logprobs.size(0):
            crop = -1

        topk_mask = convo_CE_tokens.unsqueeze(-1) == indices[:crop]
        topk_mask[ignore_indexes] = False
        valid_mask = topk_mask.any(dim=-1)
        
        valid_logprobs = torch.zeros_like(valid_mask, dtype=torch.float32)
        valid_logprobs[valid_mask] = logprobs[:crop][topk_mask].squeeze(-1)

        # compute the average of only valid tokens
        CE_valid_mean = valid_logprobs[valid_mask].mean()

        valid_logprobs[~valid_mask] = CE_valid_mean
        return -valid_logprobs, valid_mask


    min_len = min(student_logits.size(0), teacher_logits.size(0), indices.size(0) if indices is not None else student_logits.size(0))
    student_logprobs = F.log_softmax(student_logits[:min_len], dim=-1)

    if indices is not None:
        teacher_logprobs = F.log_softmax(teacher_logits[:min_len], dim=-1)
        student_logprobs = torch.gather(student_logprobs, 1, indices[:min_len])
    else:
        teacher_logprobs = F.log_softmax(teacher_logits[:min_len], dim=-1)

    convo_CE_tokens = convo_CE_tokens[:min_len]
    CE_loss, CE_valid_mask = custom_ce_loss(student_logprobs, convo_CE_tokens, indices, avoid_indices)
    teacher_CE_loss, teacher_CE_valid_mask = custom_ce_loss(teacher_logprobs, convo_CE_tokens, indices, avoid_indices)
        
    CE_diff = CE_loss - teacher_CE_loss

    corrected_CE_diff = torch.where(CE_diff < 0, torch.zeros_like(CE_diff), CE_diff)

    kl_div = custom_kl_div(student_logprobs, teacher_logprobs, per_token=True)
    reverse_kl_div = custom_kl_div(teacher_logprobs, F.log_softmax(student_logprobs, dim=-1), per_token=True)

    weighted_kl_div = abomination_loss(kl_div, alpha, corrected_CE_diff)
    weighted_r_kl_div = abomination_loss(reverse_kl_div, alpha, corrected_CE_diff)

    custom_loss = (weighted_kl_div + weighted_r_kl_div) / 2

    loss_dict = {
        "train_loss": custom_loss,
        "custom loss": custom_loss,
        "CE loss": CE_loss.mean(),
        "kl_div": kl_div.mean(),
        "reverse kl_div": reverse_kl_div.mean(),
        "weighted kl_div": weighted_kl_div,
        "weighted rev. kl_div": weighted_r_kl_div,
        "teacher CE loss": teacher_CE_loss.mean(),
        "CE diff": CE_diff.mean(),
    }

    return loss_dict


def set_optimizer(model_parameters, lr, betas, optimizer_name: str, weight_decay=1e-2, momentum=0.01, nesterov=False):
    optimizer_name = optimizer_name.lower()
    
    optimizer_classes = {
        "adam": bnb.optim.Adam,
        "adamw": bnb.optim.AdamW,
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit, 
        "paged_adamw": bnb.optim.PagedAdam,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
        "paged_adamw32bit": bnb.optim.PagedAdamW32bit,
        "sgd": bnb.optim.SGD,
        "rmsprop": bnb.optim.RMSprop,
        "rmsprop8bit": bnb.optim.RMSprop8bit,
        "rmsprop32bit": bnb.optim.RMSprop32bit,
        "adagrad": bnb.optim.Adagrad
    }

    if optimizer_name in optimizer_classes:
        if optimizer_name in ["adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        elif optimizer_name in ["sgd"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        elif optimizer_name in ["rmsprop", "rmsprop8bit", "rmsprop32bit"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay, alpha=0.9, eps=1e-10, centered=True)
        elif optimizer_name in ["adagrad"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}\nAvailable optimizers: {list(optimizer_classes.keys())}")
    

def set_lr_scheduler(optimizer, lr_scheduler_name: str, num_warmup_steps, num_training_steps, num_epoch_steps, decay_start=0.5, constant_lr=5e-5, final_lr=5e-7):
    lr_scheduler_name = lr_scheduler_name.lower()
    
    lr_scheduler_classes = {
        "wsd": WarmupStableDecayLR
    }
    
    if lr_scheduler_name in lr_scheduler_classes:
        if lr_scheduler_name == "wsd":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, num_training_steps, num_warmup_steps, decay_start, final_lr)
    else:
        return get_scheduler(lr_scheduler_name, optimizer, num_warmup_steps, num_training_steps)
        
