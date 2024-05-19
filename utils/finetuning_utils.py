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
            return [max((self.constant_lr * cosine), 1e-7) for group in self.optimizer.param_groups]
        elif self.warmup_steps <= self.last_epoch <= self.decay_start_step:
            return [self.constant_lr for group in self.optimizer.param_groups]
        elif self.decay_start_step < self.last_epoch < self.total_steps:
            lr_scale = 1 - ((self.last_epoch - self.decay_start_step) / self.decay_steps) * (1 - self.final_lr / self.constant_lr)
            return [self.constant_lr * lr_scale for group in self.optimizer.param_groups]
        else:
            return [self.final_lr for group in self.optimizer.param_groups]


def launch_tensorboard(log_dir):
    tensorboard = subprocess.Popen(['tensorboard', '--logdir', log_dir, '--bind_all'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tensorboard


def calculate_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, indices: np.ndarray, convo_content_tokens, custom=False):
    def total_kl_div(student_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor, custom: bool = False, per_token: bool = False):
        kl_div_raw = F.kl_div(student_logprobs, teacher_logprobs, reduction='none', log_target=True)

        if per_token:
            if custom:
                kl_div = kl_div_raw.sum(dim=-1)
            else:
                kl_div = kl_div_raw.mean(dim=-1)
        else:
            if custom:
                kl_div = kl_div_raw.sum(dim=-1).mean()
            else:
                kl_div = kl_div_raw.mean()

        return kl_div
    
    def inv_sum_func(student_logprobs: torch.Tensor):
        return (1 - student_logprobs.exp().sum(dim=-1).mean())

    def certainty_loss_func(inv_sum: torch.Tensor, N_remaining: int, epsilon=1.7e-6, zero_tensor: torch.Tensor = None):
        if inv_sum > 1e-5:
            return inv_sum * (inv_sum / (N_remaining * epsilon)).log()
        else:
            return zero_tensor
        
    def abomination_loss(kl_div_per_token: torch.Tensor, alpha: float):
        weights = ((kl_div_per_token / kl_div_per_token.max()) + 1).pow(alpha)

        variance = kl_div_per_token.var()

        loss = (kl_div_per_token * weights).mean()

        return loss, variance

    min_len = min(student_logits.size(0), teacher_logits.size(0), indices.size(0) if indices is not None else student_logits.size(0))
    inv_sum = torch.tensor(0.0, dtype=torch.float32).to(student_logits.device, non_blocking=True)
    certainty_loss = torch.tensor(0.0, dtype=torch.float32).to(student_logits.device, non_blocking=True)

    student_logprobs = F.log_softmax(student_logits[:min_len], dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits[:min_len], dim=-1)

    N_remaining = student_logprobs.size(-1) - teacher_logprobs.size(-1)

    cross_entropy_loss = F.cross_entropy(student_logits[:min_len - 1], convo_content_tokens[1:min_len])

    if indices is None:
        kl_div = total_kl_div(student_logprobs, teacher_logprobs, custom=custom, per_token=True)
        
    else:
        student_gathered = F.log_softmax(torch.gather(student_logprobs, dim=-1, index=indices[:min_len]), dim=-1)

        inv_sum = inv_sum_func(student_gathered)

        certainty_loss = certainty_loss_func(inv_sum, N_remaining, zero_tensor=certainty_loss)

        kl_div = total_kl_div(student_gathered, teacher_logprobs, custom=custom, per_token=True)

    custom_loss, variance = abomination_loss(kl_div, 2)

    custom_loss_divis = 1
    certainty_loss_divis = 1
    total_divis = 1

    combined_loss = (custom_loss/custom_loss_divis + certainty_loss/certainty_loss_divis) / total_divis

    return combined_loss, cross_entropy_loss, custom_loss, kl_div.mean(), variance, certainty_loss


def scale_temperature(current_step, num_training_steps, temperature):
    temp_scaling_steps = num_training_steps / 2
    if current_step < temp_scaling_steps and temperature > 1:
        decay_rate = -np.log(1e-2 / (temperature - 1)) / temp_scaling_steps
        return 1 + (temperature - 1) * np.exp(-decay_rate * current_step)
    else:
        return 1


def set_optimizer(model_parameters, lr, betas, optimizer_name: str, weight_decay=1e-2, momentum=0.9, nesterov=True):
    optimizer_name = optimizer_name.lower()
    
    optimizer_classes = {
        "adam": bnb.optim.Adam,
        "adamw": bnb.optim.AdamW,
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit, 
        "paged_adamw": bnb.optim.PagedAdam,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
        "paged_adamw32bit": bnb.optim.PagedAdamW32bit,
        "sgd": torch.optim.SGD,
        "rmsprop32bit": bnb.optim.RMSprop32bit,
    }

    if optimizer_name in optimizer_classes:
        if optimizer_name in ["adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        elif optimizer_name in ["sgd"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        elif optimizer_name in ["rmsprop32bit"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay, alpha=0.9, eps=1e-10, centered=True)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}\nAvailable optimizers: {list(optimizer_classes.keys())}")
    

def set_lr_scheduler(optimizer, lr_scheduler_name, num_warmup_steps, num_training_steps, num_epoch_steps, decay_start=0.5, constant_lr=5e-5, final_lr=5e-7):
    lr_scheduler_name = lr_scheduler_name.lower()
    
    lr_scheduler_classes = {
        "wsd": WarmupStableDecayLR
    }
    
    if lr_scheduler_name in lr_scheduler_classes:
        if lr_scheduler_name == "wsd":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, num_training_steps, num_warmup_steps, decay_start, final_lr)
    else:
        return get_scheduler(lr_scheduler_name, optimizer, num_warmup_steps, num_training_steps)
        
