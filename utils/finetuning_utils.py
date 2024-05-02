import numpy as np
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as lr
from transformers import get_scheduler
import subprocess
import os
import sys

# shut the fuck up bnb with its `bin ..bitsandbytes\libbitsandbytes_cuda121.dll`
sys.stdout = open(os.devnull, 'w')
import bitsandbytes as bnb
sys.stdout = sys.__stdout__

import math
import torch
from torch.optim.lr_scheduler import LRScheduler

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
            cosine = 0.5 * (1 + math.cos(math.pi + progress * math.pi))
            return [self.constant_lr * cosine for group in self.optimizer.param_groups]
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

def calculate_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, custom=False):
    assert teacher_logits[0].sum() != 0, "Teacher logprobs are all zeros"
    student_logprobs = F.log_softmax(student_logits, dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
    
    reduction = 'none' if custom else 'batchmean'
    kl_div = F.kl_div(student_logprobs, teacher_logprobs, reduction=reduction, log_target=True)
    if custom:
        kl_div = ((kl_div.exp() - 1).sum(dim=-1) + 1).mean().log()

    return kl_div

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
        
