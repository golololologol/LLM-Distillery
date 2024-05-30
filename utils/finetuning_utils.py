from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler
import torch.nn.functional as F
import numpy as np
import subprocess
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
    Has custom cosine warmum instead of linear.\n
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
        

def calculate_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, indices: np.ndarray, convo_content_tokens) -> dict[str, torch.Tensor]:
    def custom_kl_div(student_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor, per_token: bool = False):
        kl_div_raw = F.kl_div(student_logprobs, teacher_logprobs, reduction='none', log_target=True)

        if per_token:
            kl_div = kl_div_raw.sum(dim=-1)
        else:
            kl_div = kl_div_raw.sum(dim=-1).mean()

        return kl_div
        
    def abomination_loss(kl_div_per_token: torch.Tensor, alpha: float):
        weights = ((kl_div_per_token / kl_div_per_token.max()) + 1).pow(alpha)

        loss = (kl_div_per_token * weights).mean()

        return loss

    min_len = min(student_logits.size(0), teacher_logits.size(0), indices.size(0) if indices is not None else student_logits.size(0))

    student_logprobs = F.log_softmax(student_logits[:min_len], dim=-1)

    if indices is not None:
        teacher_logits_padded = torch.zeros_like(student_logprobs) - 25

        teacher_logits_padded.scatter_(1, indices[:min_len], teacher_logits[:min_len])

        teacher_logprobs = F.log_softmax(teacher_logits_padded, dim=-1)
    else:
        teacher_logprobs = F.log_softmax(teacher_logits[:min_len], dim=-1)

    cross_entropy_loss = F.cross_entropy(student_logits[:min_len - 1], convo_content_tokens[1:min_len])

    kl_div = custom_kl_div(student_logprobs, teacher_logprobs, per_token=True)
    reverse_kl_div = custom_kl_div(teacher_logprobs, student_logprobs, per_token=True)

    alpha = 1.2

    weighted_kl_div = abomination_loss(kl_div, alpha)
    weighted_r_kl_div = abomination_loss(reverse_kl_div, alpha)

    custom_loss = (kl_div + reverse_kl_div)/2

    loss_dict = {
        "train_loss": custom_loss.mean(), # DO NOT REMOVE  This is the loss that will be backpropagated every batch at Losses.backward()
        "custom loss": custom_loss.mean(),
        "CE loss": cross_entropy_loss,
        "kl_div": kl_div.mean(),
        "reverse kl_div": reverse_kl_div.mean(),
        "weighted kl_div": weighted_kl_div,
        "weighted rev. kl_div": weighted_r_kl_div
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
        
